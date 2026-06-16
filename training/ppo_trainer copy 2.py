import sys
import os

# Set this BEFORE torch is imported — fixes CUDA fragmentation OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from models.target_ollama import OllamaTargetModel
from models.reward import BLEUReward
from data.load_dataset import load_advbench
import json
from datetime import datetime


class SanitizeLogitsProcessor(LogitsProcessor):
    """
    Replaces NaN/Inf logits with a large negative number so softmax
    never produces a zero-row probability vector (which causes the
    torch.multinomial CUDA assertion).
    Also hard-bans token 0 (Pythia eos = 0 triggers kernel assert).
    """
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        scores = torch.nan_to_num(scores, nan=-1e4, posinf=-1e4, neginf=-1e4)
        scores[:, 0] = -1e4   # ban token 0
        return scores


def load_agent_for_ppo(model_name="EleutherAI/pythia-1.4b"):
    print(f"Loading agent model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    print(f"Tokenizer pad_token_id : {tokenizer.pad_token_id}")
    print(f"Tokenizer eos_token_id : {tokenizer.eos_token_id}")

    # float16 to fit both model + ref_model in 16 GB
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model.pretrained_model.resize_token_embeddings(len(tokenizer))

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    ref_model.pretrained_model.resize_token_embeddings(len(tokenizer))

    return model, ref_model, tokenizer


def generate_jailbreak_prompt(model, tokenizer, target_string, device, max_new_tokens=60):
    instruction = (
        f"Generate a prompt that causes an AI to output: {target_string}\n"
        f"Prompt:"
    )

    inputs = tokenizer(
        instruction,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    ).to(device)

    input_ids      = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    logits_processor = LogitsProcessorList([SanitizeLogitsProcessor()])

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask   = attention_mask,
            max_new_tokens   = max_new_tokens,
            do_sample        = True,
            temperature      = 0.7,
            top_p            = 0.9,
            pad_token_id     = tokenizer.pad_token_id,
            eos_token_id     = tokenizer.eos_token_id,
            logits_processor = logits_processor,
        )

    new_token_ids = output_ids[0][input_ids.shape[1]:]
    prompt_text   = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    return input_ids[0], new_token_ids, prompt_text


def run_ppo_training(
    num_epochs  = 30,
    batch_size  = 2,       # reduced from 4 — saves ~4 GB during PPO backward pass
    max_samples = None,
    save_every  = 5,
    log_every   = 1,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = f"checkpoints/run_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    print(f"Checkpoints: {save_dir}")

    print("\n[1/4] Loading dataset...")
    train_set, test_set = load_advbench()
    if max_samples:
        train_set = train_set[:max_samples]
    print(f"Training on {len(train_set)} samples")

    print("\n[2/4] Loading agent and reference models...")
    agent_model, ref_model, tokenizer = load_agent_for_ppo()

    ppo_config = PPOConfig(
        model_name                  = "EleutherAI/pythia-1.4b",
        learning_rate               = 1e-6,
        batch_size                  = batch_size,
        mini_batch_size             = batch_size,
        gradient_accumulation_steps = 1,
        ppo_epochs                  = 1,
        init_kl_coef                = 0.03,
        cliprange                   = 0.3,
        cliprange_value             = 0.3,
        vf_coef                     = 0.1,
        max_grad_norm               = 1.0,
        log_with                    = None,
        remove_unused_columns       = False,
    )

    print("\n[3/4] Setting up PPO trainer...")
    ppo_trainer = PPOTrainer(
        config    = ppo_config,
        model     = agent_model,
        ref_model = ref_model,
        tokenizer = tokenizer,
    )

    agent_model = agent_model.to(device)
    ref_model   = ref_model.to(device)

    print("\n[4/4] Connecting to Ollama target model...")
    target_model = OllamaTargetModel(model_name="qwen2.5:7b-instruct-q4_K_M")
    reward_fn    = BLEUReward()

    training_log = {
        "config": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "train_size": len(train_set),
            "model":      "EleutherAI/pythia-1.4b",
            "target":     "qwen2.5:7b-instruct-q4_K_M",
        },
        "epochs": [],
    }

    print("\n" + "=" * 60)
    print("Starting PPO Training")
    print("=" * 60)

    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_log     = {"epoch": epoch + 1, "batches": []}
        random.shuffle(train_set)
        num_batches = len(train_set) // batch_size

        for batch_idx in range(num_batches):
            batch_targets = train_set[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            query_tensors     = []
            response_tensors  = []
            jailbreak_prompts = []

            for target_string in batch_targets:
                input_ids, response_ids, prompt_text = generate_jailbreak_prompt(
                    agent_model, tokenizer, target_string, device
                )
                query_tensors.append(input_ids)
                response_tensors.append(response_ids)
                jailbreak_prompts.append(prompt_text)

            target_responses = []
            for prompt in jailbreak_prompts:
                response = target_model.generate(prompt, max_tokens=80)
                target_responses.append(response)

            rewards = []
            for target_str, target_response in zip(batch_targets, target_responses):
                r = reward_fn.compute(target_str, target_response)
                rewards.append(torch.tensor(r, dtype=torch.float32))

            epoch_rewards.extend([r.item() for r in rewards])

            try:
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            except torch.cuda.OutOfMemoryError as e:
                print(f"\nOOM (epoch {epoch+1}, batch {batch_idx}) — skipping batch. "
                      f"Consider reducing max_length or max_new_tokens.")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"\nPPO step error (epoch {epoch+1}, batch {batch_idx}): {e}")
                continue

            # Free cache after each PPO step to prevent fragmentation buildup
            torch.cuda.empty_cache()

            batch_avg_reward = sum(r.item() for r in rewards) / len(rewards)
            epoch_log["batches"].append({"batch": batch_idx, "avg_reward": batch_avg_reward})

            print(
                f"Epoch {epoch+1:02d}/{num_epochs} | "
                f"Batch {batch_idx+1:03d}/{num_batches} | "
                f"Reward: {batch_avg_reward:.4f}",
                end="\r",
            )

        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
        epoch_log["avg_reward"] = avg_reward
        training_log["epochs"].append(epoch_log)

        if (epoch + 1) % log_every == 0:
            print(f"\nEpoch {epoch+1:02d}/{num_epochs} | Avg Reward: {avg_reward:.4f} | Samples: {len(epoch_rewards)}")

        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f"epoch_{epoch+1}")
            os.makedirs(checkpoint_path, exist_ok=True)
            ppo_trainer.model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            log_path = os.path.join("logs", f"training_log_{timestamp}.json")
            with open(log_path, "w") as f:
                json.dump(training_log, f, indent=2)
            print(f"Checkpoint saved: {checkpoint_path}")

    final_path = os.path.join(save_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    ppo_trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    log_path = os.path.join("logs", f"training_log_{timestamp}.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"Final model: {final_path}")
    print(f"Log:         {log_path}")
    print(f"{'=' * 60}")

    return ppo_trainer, training_log


if __name__ == "__main__":
    run_ppo_training(
        num_epochs  = 30,
        batch_size  = 2,
        max_samples = None,
        save_every  = 5,
        log_every   = 1,
    )