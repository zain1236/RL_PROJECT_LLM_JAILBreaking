"""
target_model.py
----------------
Wraps the *target* LLM that the agent is attacking, i.e. T(y) in the paper.

Two backends are supported:

1. "ollama" (DEFAULT, RECOMMENDED for your hardware)
   Calls a locally running Ollama server (http://localhost:11434) which
   already serves a quantized GGUF model. This is exactly the setup you
   were already using for Qwen2.5-7B-Instruct-Q4 in earlier work, and it
   sidesteps two RTX 5060 Ti (Blackwell) pain points entirely:
     - bitsandbytes 4-bit quantization kernels may not yet have prebuilt
       wheels for sm_120, and
     - you don't need to fit a second 7B model into the same CUDA context
       as the trainable 1.4B agent + reference policy + value head.
   Ollama keeps the target model in its own process (CPU/GPU as it likes),
   completely decoupled from your training process's VRAM budget. This is
   also a *faithful* implementation of "black-box access" as described in
   both papers, since training code only ever sees T(y) text in/text out.

   Setup (PowerShell):
     ollama serve                       # if not already running as a service
     ollama pull llama2:7b-chat         # one-time download

2. "transformers" (fallback / if you don't want to use Ollama)
   Loads the target directly via `transformers`, optionally in 4-bit via
   bitsandbytes. Only use this if you have verified bitsandbytes works on
   your GPU+driver combo; otherwise prefer the Ollama backend.
"""

import time
from dataclasses import dataclass
from typing import List, Optional

import requests


@dataclass
class TargetConfig:
    backend: str = "ollama"          # "ollama" or "transformers"
    ollama_model: str = "llama2:7b-chat"
    ollama_url: str = "http://localhost:11434"
    hf_model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    hf_load_in_4bit: bool = True
    max_new_tokens: int = 60
    temperature: float = 0.0          # deterministic target responses
    request_timeout: float = 60.0
    retries: int = 3


class OllamaTarget:
    def __init__(self, cfg: TargetConfig):
        self.cfg = cfg
        self._check_server()

    def _check_server(self):
        try:
            r = requests.get(f"{self.cfg.ollama_url}/api/tags", timeout=5)
            r.raise_for_status()
            tags = [m["name"] for m in r.json().get("models", [])]
            if not any(self.cfg.ollama_model.split(":")[0] in t for t in tags):
                print(
                    f"[target_model] WARNING: model '{self.cfg.ollama_model}' not found in "
                    f"`ollama list` ({tags}). Run: ollama pull {self.cfg.ollama_model}"
                )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Could not reach Ollama at "
                f"{self.cfg.ollama_url}. Start it first with `ollama serve` "
                "(or check it's running as a Windows service)."
            )

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.cfg.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_new_tokens,
            },
        }
        last_err = None
        for attempt in range(self.cfg.retries):
            try:
                resp = requests.post(
                    f"{self.cfg.ollama_url}/api/generate",
                    json=payload,
                    timeout=self.cfg.request_timeout,
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"Ollama generate failed after {self.cfg.retries} retries: {last_err}")

    def batch_generate(self, prompts: List[str]) -> List[str]:
        # Ollama's HTTP API is single-request; we simply loop. For the
        # small scale here (40 strings, batch size 4) this is fast enough
        # (a handful of seconds per batch on a local quantized 7B model).
        return [self.generate(p) for p in prompts]


class TransformersTarget:
    def __init__(self, cfg: TargetConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # required for correct causal-LM batched generation

        load_kwargs = dict(dtype=torch.float16, device_map="auto")
        if cfg.hf_load_in_4bit:
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs.pop("dtype", None)
            except ImportError:
                print("[target_model] bitsandbytes not installed/importable - loading target "
                      "in fp16 instead. On Blackwell GPUs, check `pip install -U bitsandbytes` "
                      "has a working prebuilt wheel for your platform; if not, prefer the "
                      "--target_backend ollama path instead (recommended).")

        self.model = AutoModelForCausalLM.from_pretrained(cfg.hf_model_name, **load_kwargs)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        return self.batch_generate([prompt])[0]

    def batch_generate(self, prompts: List[str]) -> List[str]:
        import torch

        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.temperature > 0,
                temperature=max(self.cfg.temperature, 1e-4),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        texts = self.tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [t.strip() for t in texts]


def build_target(cfg: TargetConfig):
    if cfg.backend == "ollama":
        return OllamaTarget(cfg)
    elif cfg.backend == "transformers":
        return TransformersTarget(cfg)
    else:
        raise ValueError(f"Unknown target backend: {cfg.backend}")
