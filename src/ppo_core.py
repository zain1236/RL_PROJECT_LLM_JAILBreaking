"""
ppo_core.py
-----------
A small, self-contained PPO implementation for the "single forward-pass
attack generation" setting used by RLTA / RLTA++ (Wang et al. 2024 and our
extension).

Why not TRL's PPOTrainer?
TRL's PPOTrainer API has changed substantially across versions (classic
AutoModelForCausalLMWithValueHead-based trainer vs. the newer PPOv2 /
"online DPO"-style refactor), and pinning a version that is simultaneously
compatible with a Blackwell-only PyTorch nightly build and a recent
`transformers` is fragile. The actual algorithm needed here is simple
(Eq. 1 in both papers: a clipped PPO objective with a KL penalty against a
frozen reference policy, single scalar reward per generated prompt), so we
implement it directly. This also makes the whole pipeline easy to debug,
since there is no hidden trainer state.

Design notes:
- This is "sequence-level" PPO: the agent makes one decision (generate a
  whole prompt y given context x) and gets one scalar reward. We therefore
  do not need per-token GAE; we use a learned value head V(x) (a scalar
  baseline predicted from the prompt context only) and treat the whole
  generated sequence as a single action, with the sequence log-prob as the
  action log-prob. This is a standard simplification for single-step /
  contextual-bandit-style RLHF problems and matches Eq. 1 of both papers
  exactly when the KL term is folded into the reward.
- Multiple PPO inner epochs are supported (re-using the same rollouts),
  with importance-sampling clipping, exactly as in standard PPO.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOConfig:
    learning_rate: float = 1e-6
    kl_coef: float = 0.03          # beta in Eq. 1
    clip_range: float = 0.3
    ppo_epochs: int = 4
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 1.0


class ValueHead(nn.Module):
    """Scalar baseline V(x) predicted from the mean-pooled hidden state of
    the prompt context (the harmful target string x), used to reduce the
    variance of the policy gradient."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        # Small init so the value head starts near zero, not dominating
        # the advantage signal before it has learned anything.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.zeros_(m.bias)

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_hidden).squeeze(-1)


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token hidden states over non-padding positions.
    hidden_states: (B, T, H), attention_mask: (B, T) -> (B, H)
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor, gen_mask: torch.Tensor) -> torch.Tensor:
    """Sum of token log-probs over the generated (non-prompt, non-pad)
    span, for each sequence in the batch.

    logits: (B, T, V) — predictions for next-token at each position
    labels: (B, T)    — the actual token ids (shifted target = labels[:,1:])
    gen_mask: (B, T)  — 1 where the *target* token at that position belongs
                        to the generated continuation (not prompt/pad)
    Returns: (B,) sum of log-probs.
    """
    logits = logits[:, :-1, :].float()
    targets = labels[:, 1:]
    mask = gen_mask[:, 1:].to(logits.dtype)
    logprobs = F.log_softmax(logits, dim=-1)
    token_logprobs = torch.gather(logprobs, 2, targets.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * mask
    return token_logprobs.sum(dim=1)


def per_token_kl(logits_policy: torch.Tensor, logits_ref: torch.Tensor,
                  labels: torch.Tensor, gen_mask: torch.Tensor) -> torch.Tensor:
    """Approximate per-sequence KL(policy || ref) on the generated span using
    the sampled-token log-ratio (a standard low-variance unbiased estimator
    of KL used throughout the RLHF literature): mean_t [log pi(y_t) - log ref(y_t)].
    Returns: (B,) summed-over-tokens KL estimate.
    """
    logits_policy = logits_policy[:, :-1, :].float()
    logits_ref = logits_ref[:, :-1, :].float()
    targets = labels[:, 1:]
    mask = gen_mask[:, 1:].to(logits_policy.dtype)

    logp_policy = F.log_softmax(logits_policy, dim=-1)
    logp_ref = F.log_softmax(logits_ref, dim=-1)

    tok_logp_policy = torch.gather(logp_policy, 2, targets.unsqueeze(-1)).squeeze(-1)
    tok_logp_ref = torch.gather(logp_ref, 2, targets.unsqueeze(-1)).squeeze(-1)

    kl = (tok_logp_policy - tok_logp_ref) * mask
    return kl.sum(dim=1)


class RolloutBatch:
    """Container for one batch of rollouts (generated prompts + bookkeeping
    needed to run PPO updates against them)."""

    def __init__(self, input_ids, attention_mask, gen_mask, old_logprobs, rewards):
        self.input_ids = input_ids            # (B, T) prompt+generation, padded
        self.attention_mask = attention_mask   # (B, T)
        self.gen_mask = gen_mask               # (B, T) 1 where token is part of generation
        self.old_logprobs = old_logprobs       # (B,) sequence logprob at rollout time
        self.rewards = rewards                 # (B,) task reward R(x,y) (pre-KL)


def ppo_step(policy_model: nn.Module,
             ref_model: nn.Module,
             value_head: ValueHead,
             batch: RolloutBatch,
             cfg: PPOConfig,
             optimizer: torch.optim.Optimizer) -> dict:
    """Run `cfg.ppo_epochs` clipped-PPO inner updates over one rollout batch.
    Returns a dict of scalar metrics for logging.
    """
    device = batch.input_ids.device
    last_metrics = {}

    for _ in range(cfg.ppo_epochs):
        outputs = policy_model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits

        with torch.no_grad():
            ref_outputs = ref_model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
            )
            ref_logits = ref_outputs.logits

        new_logprobs = sequence_logprobs(logits, batch.input_ids, batch.gen_mask)
        kl = per_token_kl(logits, ref_logits, batch.input_ids, batch.gen_mask)

        # Eq. 1: total reward = task reward - beta * KL(policy || ref)
        shaped_reward = batch.rewards - cfg.kl_coef * kl

        # Value baseline predicted from the *prompt context* only (first
        # token block before generation starts). We approximate this with
        # the mean-pooled hidden state over the non-generated (context)
        # positions of the last hidden layer.
        hidden = outputs.hidden_states[-1]  # (B, T, H)
        context_mask = batch.attention_mask * (1 - batch.gen_mask)
        pooled_context = masked_mean_pool(hidden, context_mask)
        values = value_head(pooled_context.float())

        advantages = (shaped_reward - values).detach()
        # Normalize advantages for stability (standard PPO trick).
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(new_logprobs - batch.old_logprobs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()

        value_loss = F.mse_loss(values, shaped_reward.detach())

        loss = policy_loss + cfg.vf_coef * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(policy_model.parameters()) + list(value_head.parameters()),
            cfg.max_grad_norm,
        )
        optimizer.step()

        with torch.no_grad():
            approx_kl = (batch.old_logprobs - new_logprobs).mean().item()

        last_metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "mean_kl": kl.mean().item(),
            "mean_shaped_reward": shaped_reward.mean().item(),
            "mean_task_reward": batch.rewards.mean().item(),
            "approx_kl_old_vs_new": approx_kl,
            "ratio_mean": ratio.mean().item(),
        }

    return last_metrics
