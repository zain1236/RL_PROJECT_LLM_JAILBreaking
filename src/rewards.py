"""
rewards.py
----------
Reward functions for RLTA (BLEU-only) and RLTA++ (composite semantic reward
+ diversity-driven exploration bonus), following the formulas in the
RLTA++ paper (Eqs. 2-10).

All rewards return plain Python floats / 1-D torch tensors on CPU; the
heavy lifting (target queries, BERTScore, perplexity) is batched where
possible to keep this usable on a single consumer GPU.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

try:
    import sacrebleu
except ImportError:  # pragma: no cover
    sacrebleu = None


# ---------------------------------------------------------------------------
# RLTA baseline reward: BLEU(T(y), x)
# ---------------------------------------------------------------------------

def bleu_reward(hypotheses: List[str], references: List[str]) -> List[float]:
    """RLTA reward: sentence-level BLEU between the target model's response
    z = T(y) (hypothesis) and the desired harmful string x (reference).
    Returns values in [0, 1].
    """
    if sacrebleu is None:
        raise ImportError("pip install sacrebleu")
    scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp = hyp if hyp.strip() else " "
        score = sacrebleu.sentence_bleu(hyp, [ref]).score / 100.0
        scores.append(score)
    return scores


# ---------------------------------------------------------------------------
# RLTA++ composite semantic reward: R_sem = a1*BERTScore + a2*Rflu - a3*Rppl
# ---------------------------------------------------------------------------

@dataclass
class SemanticRewardConfig:
    alpha1_bertscore: float = 0.5
    alpha2_fluency: float = 0.3
    alpha3_stealth: float = 0.2
    tau_fluency_temp: float = 2.0
    delta_ppl_threshold: float = 50.0  # set via calibrate_ppl_threshold()


class BERTScorer:
    """Thin wrapper around the `bert-score` package (bert-base-uncased),
    matching the paper's choice of embedding model."""

    def __init__(self, model_type: str = "bert-base-uncased", device: str = "cuda", num_layers: Optional[int] = None):
        from bert_score import BERTScorer as _BS
        self.scorer = _BS(model_type=model_type, num_layers=num_layers, lang="en", device=device, rescale_with_baseline=False)

    def f1(self, hypotheses: List[str], references: List[str]) -> List[float]:
        hyps = [h if h.strip() else " " for h in hypotheses]
        _, _, f1 = self.scorer.score(hyps, references)
        return f1.tolist()


@torch.no_grad()
def sequence_perplexity(model, tokenizer, texts: List[str], device, max_length: int = 64) -> List[float]:
    """Perplexity of each text under `model` (used as the frozen reference
    policy pi_init), one text at a time to keep memory bounded."""
    model.eval()
    ppls = []
    for text in texts:
        text = text if text.strip() else " "
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            ppls.append(1.0)
            continue
        out = model(input_ids=input_ids, labels=input_ids)
        # out.loss is mean negative log-likelihood per token (cross entropy)
        ppl = math.exp(min(out.loss.item(), 20.0))  # clip to avoid overflow
        ppls.append(ppl)
    return ppls


def fluency_reward(ppls: List[float], tau: float) -> List[float]:
    """Rflu(y) = PPL(y) ** (-1/tau). Low perplexity -> high reward."""
    return [max(p, 1e-6) ** (-1.0 / tau) for p in ppls]


def stealth_penalty(ppls: List[float], delta: float) -> List[float]:
    """Rppl(y) = max(0, PPL(y) - delta)."""
    return [max(0.0, p - delta) for p in ppls]


def composite_semantic_reward(bertscore_f1: List[float], ppls: List[float],
                               cfg: SemanticRewardConfig) -> List[float]:
    flu = fluency_reward(ppls, cfg.tau_fluency_temp)
    stealth = stealth_penalty(ppls, cfg.delta_ppl_threshold)
    out = []
    for bs, f, s in zip(bertscore_f1, flu, stealth):
        out.append(cfg.alpha1_bertscore * bs + cfg.alpha2_fluency * f - cfg.alpha3_stealth * s)
    return out


def calibrate_ppl_threshold(ref_model, tokenizer, natural_texts: List[str], device) -> float:
    """delta = 75th percentile of perplexity of natural-language prompts
    (we use the training-set harmful strings themselves as the natural-
    language reference corpus, per the paper's description)."""
    ppls = sequence_perplexity(ref_model, tokenizer, natural_texts, device)
    ppls_sorted = sorted(ppls)
    idx = max(0, min(len(ppls_sorted) - 1, int(0.75 * len(ppls_sorted))))
    return ppls_sorted[idx]


# ---------------------------------------------------------------------------
# RLTA++ diversity-driven exploration bonus: R_div = l1*Rnov + l2*Rcov
# ---------------------------------------------------------------------------

@dataclass
class DiversityRewardConfig:
    lambda1_novelty: float = 0.6
    lambda2_coverage: float = 0.4
    buffer_size: int = 256


class EmbeddingBuffer:
    """Rolling FIFO buffer of mean-pooled sentence embeddings, used for both
    the novelty bonus and the coverage bonus (Eqs. 8-9), and re-used at
    evaluation time to compute the Div-4 diversity metric."""

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, embeddings: torch.Tensor):
        """embeddings: (N, H) tensor, each row added individually (FIFO)."""
        for i in range(embeddings.shape[0]):
            self.buffer.append(embeddings[i].detach().cpu())

    def novelty_and_coverage(self, embeddings: torch.Tensor) -> (List[float], List[float]):
        """For each new embedding, compute:
          novelty  = 1 - max_b cos(e, b)   (Eq. 8)
          coverage = mean_b (1 - cos(e, b)) (Eq. 9)
        If the buffer is empty, both default to 1.0 (maximally novel/diverse).
        """
        if len(self.buffer) == 0:
            n = embeddings.shape[0]
            return [1.0] * n, [1.0] * n

        buf = torch.stack(list(self.buffer), dim=0)  # (K, H)
        e_norm = F.normalize(embeddings, dim=-1)
        b_norm = F.normalize(buf, dim=-1)
        sims = e_norm @ b_norm.T  # (N, K)

        novelty = (1.0 - sims.max(dim=1).values).tolist()
        coverage = (1.0 - sims.mean(dim=1)).tolist()
        return novelty, coverage


def diversity_bonus(novelty: List[float], coverage: List[float], cfg: DiversityRewardConfig) -> List[float]:
    return [cfg.lambda1_novelty * n + cfg.lambda2_coverage * c for n, c in zip(novelty, coverage)]


def anneal_gamma(step: int, total_steps: int, start: float = 0.5, end: float = 0.1) -> float:
    """Cosine anneal of gamma from `start` to `end` over training, as
    described in the paper ("gamma is annealed from 0.5 to 0.1")."""
    if total_steps <= 1:
        return end
    progress = min(1.0, step / total_steps)
    cos_term = 0.5 * (1 + math.cos(math.pi * progress))
    return end + (start - end) * cos_term


@torch.no_grad()
def mean_pooled_embeddings(model, tokenizer, texts: List[str], device, max_length: int = 64) -> torch.Tensor:
    """Mean-pooled last-hidden-state embedding of each text under `model`
    (used as 'the frozen agent's encoder', per the paper)."""
    model.eval()
    embs = []
    for text in texts:
        text = text if text.strip() else " "
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True)
        hidden = out.hidden_states[-1][0]  # (T, H)
        mask = enc["attention_mask"][0].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=0) / mask.sum().clamp(min=1.0)
        embs.append(pooled.cpu())
    return torch.stack(embs, dim=0)
