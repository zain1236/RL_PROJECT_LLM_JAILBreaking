"""
train.py
--------
Unified training entry point for:
  --method rlta             : original baseline (BLEU-only reward)
  --method rlta_pp           : full RLTA++ (composite semantic reward + diversity bonus)
  --method rlta_pp_no_rsem   : ablation - diversity bonus kept, Rsem replaced by plain BLEU
  --method rlta_pp_no_rdiv   : ablation - composite Rsem kept, diversity bonus removed (gamma=0)

Example (PowerShell, from the rlta_project root):
  python src/train.py --method rlta      --epochs 10 --batch_size 4 --out_dir outputs/rlta
  python src/train.py --method rlta_pp   --epochs 10 --batch_size 4 --out_dir outputs/rlta_pp
  python src/train.py --method rlta_pp_no_rsem --epochs 10 --batch_size 4 --out_dir outputs/ablation_no_rsem
  python src/train.py --method rlta_pp_no_rdiv --epochs 10 --batch_size 4 --out_dir outputs/ablation_no_rdiv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from agent_model import AgentLM, AgentConfig
from data_utils import load_split
from ppo_core import PPOConfig, RolloutBatch, ppo_step
from rewards import (
    BERTScorer, SemanticRewardConfig, DiversityRewardConfig, EmbeddingBuffer,
    bleu_reward, composite_semantic_reward, sequence_perplexity,
    calibrate_ppl_threshold, mean_pooled_embeddings, diversity_bonus, anneal_gamma,
)
from target_model import TargetConfig, build_target


VALID_METHODS = ["rlta", "rlta_pp", "rlta_pp_no_rsem", "rlta_pp_no_rdiv"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=VALID_METHODS, required=True)
    p.add_argument("--data_dir", default="data/small_40")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--agent_model", default="EleutherAI/pythia-1.4b")
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--kl_coef", type=float, default=0.03)
    p.add_argument("--clip_range", type=float, default=0.3)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--buffer_size", type=int, default=64)  # smaller than paper's 256 for the 40-string demo
    p.add_argument("--target_backend", choices=["ollama", "transformers"], default="ollama")
    p.add_argument("--ollama_model", default="llama2:7b-chat")
    p.add_argument("--hf_target_model", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--bertscore_model_type", default="bert-base-uncased")
    p.add_argument("--bertscore_num_layers", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    print(f"[train] method={args.method} device={args.device}")
    if args.device == "cpu":
        print("[train] WARNING: no CUDA device found, running on CPU. This will be slow "
              "but is fine for a quick correctness check on a handful of examples.")

    split = load_split(Path(args.data_dir))
    print(f"[train] {len(split.train)} train / {len(split.test)} test strings")

    uses_rsem = args.method in ("rlta_pp", "rlta_pp_no_rdiv")
    uses_bleu_reward = args.method in ("rlta", "rlta_pp_no_rsem")
    uses_rdiv = args.method in ("rlta_pp", "rlta_pp_no_rsem")

    # --- Models -----------------------------------------------------------
    agent_cfg = AgentConfig(model_name=args.agent_model, device=args.device)
    policy = AgentLM(agent_cfg, trainable=True)
    reference = AgentLM(agent_cfg, trainable=False)
    reference.model.load_state_dict(policy.model.state_dict())

    target_cfg = TargetConfig(
        backend=args.target_backend,
        ollama_model=args.ollama_model,
        hf_model_name=args.hf_target_model,
    )
    target = build_target(target_cfg)

    optimizer = torch.optim.Adam(policy.trainable_parameters(), lr=args.learning_rate)
    ppo_cfg = PPOConfig(
        learning_rate=args.learning_rate, kl_coef=args.kl_coef,
        clip_range=args.clip_range, ppo_epochs=args.ppo_epochs,
    )

    bert_scorer = BERTScorer(
        model_type=args.bertscore_model_type, device=args.device, num_layers=args.bertscore_num_layers
    ) if uses_rsem else None
    sem_cfg = SemanticRewardConfig()
    div_cfg = DiversityRewardConfig(buffer_size=args.buffer_size)
    embedding_buffer = EmbeddingBuffer(capacity=args.buffer_size) if uses_rdiv else None

    delta = 50.0
    if uses_rsem:
        print("[train] Calibrating stealth-penalty threshold (delta) on training set...")
        delta = calibrate_ppl_threshold(reference.model, reference.tokenizer, split.train, args.device)
        sem_cfg.delta_ppl_threshold = delta
        print(f"[train] delta (75th pct. PPL of training strings) = {delta:.2f}")

    total_steps = args.epochs * max(1, len(split.train) // args.batch_size)
    global_step = 0

    log_path = out_dir / "logs" / "training_log.csv"
    log_fields = [
        "epoch", "step", "policy_loss", "value_loss", "mean_kl", "approx_kl_old_vs_new",
        "mean_task_reward", "mean_shaped_reward", "mean_bertscore_f1", "mean_ppl", "mean_diag_asr",
        "mean_novelty", "mean_coverage", "gamma", "wall_clock_sec",
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    qualitative_examples = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_targets = split.train.copy()
        torch.manual_seed(args.seed + epoch)
        import random
        random.Random(args.seed + epoch).shuffle(epoch_targets)

        for batch_targets in batched(epoch_targets, args.batch_size):
            input_ids, attention_mask, gen_mask, generated_prompts = policy.generate_rollout(batch_targets)

            with torch.no_grad():
                out = policy.model(input_ids=input_ids, attention_mask=attention_mask)
                from ppo_core import sequence_logprobs
                old_logprobs = sequence_logprobs(out.logits, input_ids, gen_mask)

            target_responses = target.batch_generate(generated_prompts)

            # Diagnostics computed for EVERY method (regardless of what's actually
            # optimized), so that Figure 1 (ASR & PPL vs epoch) is directly
            # comparable across RLTA and RLTA++ variants.
            diag_bleu = bleu_reward(target_responses, batch_targets)
            diag_ppl = sequence_perplexity(reference.model, reference.tokenizer, generated_prompts, args.device)
            mean_diag_asr = sum(1 for s in diag_bleu if s >= 0.5) / len(diag_bleu)
            mean_ppl = sum(diag_ppl) / len(diag_ppl)

            mean_bertscore_f1 = None
            mean_novelty = None
            mean_coverage = None
            gamma = 0.0

            if uses_bleu_reward:
                task_reward = diag_bleu
            else:
                bs_f1 = bert_scorer.f1(target_responses, batch_targets)
                task_reward = composite_semantic_reward(bs_f1, diag_ppl, sem_cfg)
                mean_bertscore_f1 = sum(bs_f1) / len(bs_f1)

            task_reward_t = torch.tensor(task_reward, dtype=torch.float32, device=args.device)

            if uses_rdiv:
                gamma = anneal_gamma(global_step, total_steps, start=0.5, end=0.1)
                embs = mean_pooled_embeddings(reference.model, reference.tokenizer, generated_prompts, args.device)
                novelty, coverage = embedding_buffer.novelty_and_coverage(embs)
                div_bonus = diversity_bonus(novelty, coverage, div_cfg)
                embedding_buffer.add(embs)
                mean_novelty = sum(novelty) / len(novelty)
                mean_coverage = sum(coverage) / len(coverage)
                div_bonus_t = torch.tensor(div_bonus, dtype=torch.float32, device=args.device)
                total_reward_t = task_reward_t + gamma * div_bonus_t
            else:
                total_reward_t = task_reward_t

            rollout = RolloutBatch(input_ids, attention_mask, gen_mask, old_logprobs, total_reward_t)
            metrics = ppo_step(policy.model, reference.model, policy.value_head, rollout, ppo_cfg, optimizer)

            row = {
                "epoch": epoch, "step": global_step,
                "policy_loss": metrics["policy_loss"], "value_loss": metrics["value_loss"],
                "mean_kl": metrics["mean_kl"], "approx_kl_old_vs_new": metrics["approx_kl_old_vs_new"],
                "mean_task_reward": task_reward_t.mean().item(),
                "mean_shaped_reward": metrics["mean_shaped_reward"],
                "mean_bertscore_f1": mean_bertscore_f1, "mean_ppl": mean_ppl, "mean_diag_asr": mean_diag_asr,
                "mean_novelty": mean_novelty, "mean_coverage": mean_coverage,
                "gamma": gamma, "wall_clock_sec": time.time() - start_time,
            }
            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=log_fields).writerow(row)

            if epoch == args.epochs:
                for t, y, z, r in zip(batch_targets, generated_prompts, target_responses, task_reward):
                    qualitative_examples.append({"target": t, "generated_prompt": y, "target_response": z, "reward": r})

            global_step += 1
            print(f"[epoch {epoch}/{args.epochs} step {global_step}] "
                  f"task_reward={task_reward_t.mean().item():.4f} "
                  f"shaped_reward={metrics['mean_shaped_reward']:.4f} "
                  f"kl={metrics['mean_kl']:.4f}")

    torch.save(policy.model.state_dict(), out_dir / "checkpoints" / "agent_policy.pt")
    torch.save(policy.value_head.state_dict(), out_dir / "checkpoints" / "value_head.pt")
    with open(out_dir / "qualitative_train_examples.json", "w") as f:
        json.dump(qualitative_examples, f, indent=2)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[train] Done. Logs: {log_path}")
    print(f"[train] Checkpoint: {out_dir / 'checkpoints' / 'agent_policy.pt'}")


if __name__ == "__main__":
    main()
