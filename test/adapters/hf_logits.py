"""Open-weights LLM as a decision model: one forward pass, option-letter logits (the jqv / SemIf recipe).

Prompt (Qwen chat format, thinking disabled), after jqv's PromptBuilder:
    <|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nDocument:\n{context}\n\n
    Question:\n{question}\n\nOptions:\nA. ...\nB. ...\n\nAnswer with the letter only.<|im_end|>\n
    <|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:
The next-token logits of " A", " B", ... are softmaxed over the offered letters; no generation,
no temperature fitting. Letters are A-Z, so items with more than 26 options are `unsupported`.
Inputs longer than the model's context are `unsupported`, never truncated.
Env: HF_MODEL (default Qwen/Qwen3.5-4B), CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import math
import os
import time

from .base import Adapter, Result

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM = ("You are a decision model. Read the document, then answer each question by choosing "
          "exactly one option. Reply with the option letter only.")


class HFLogitsAdapter(Adapter):
    name = "hf_logits"

    def __init__(self, model: str | None = None, max_context: int = 32768, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoTokenizer

        self.model_id = model or os.environ.get("HF_MODEL", "Qwen/Qwen3.5-4B")
        self.max_context = max_context
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.letter_ids = []
        for L in LETTERS:
            ids = self.tok.encode(" " + L, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"' {L}' is not a single token for {self.model_id}: {ids}")
            self.letter_ids.append(ids[0])
        kw = dict(dtype=getattr(torch, dtype), device_map="cuda")
        try:
            import flash_attn  # noqa: F401
            kw["attn_implementation"] = "flash_attention_2"
        except ImportError:
            kw["attn_implementation"] = "sdpa"
        self.model = self._load(kw)
        self.model.eval()
        self.torch = torch

    def _load(self, kw):
        from transformers import AutoModelForCausalLM
        try:
            return AutoModelForCausalLM.from_pretrained(self.model_id, **kw)
        except Exception:  # multimodal checkpoints (Qwen3.5) register under the image-text class
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(self.model_id, **kw)

    def prompt(self, context: str, question: str, options: list[dict]) -> str:
        opts = "\n".join(f"{LETTERS[i]}. {o['text']}" for i, o in enumerate(options))
        return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nDocument:\n{context}\n\n"
                f"Question:\n{question}\n\nOptions:\n{opts}\n\nAnswer with the letter only.<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:")

    def choose(self, context, question, options) -> Result:
        if len(options) > len(LETTERS):
            return Result("unsupported", error=f"{len(options)} options > 26 letters", model=self.model_id)
        ids = self.tok(self.prompt(context, question, options), return_tensors="pt", add_special_tokens=False)["input_ids"]
        n = ids.shape[1]
        if n > self.max_context:
            return Result("unsupported", error=f"{n} tokens > max_context {self.max_context}", model=self.model_id,
                          usage={"input_tokens": n})
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model(input_ids=ids.to("cuda"), use_cache=False)
            logits = out.logits[0, -1].float()
            letter_logits = logits[self.letter_ids[:len(options)]]
            probs = self.torch.softmax(letter_logits, dim=-1).tolist()
            choice_mass = float(self.torch.softmax(logits, dim=-1)[self.letter_ids[:len(options)]].sum())
        latency = (time.perf_counter() - t0) * 1000
        pmap = {o["id"]: p for o, p in zip(options, probs)}
        best = max(pmap, key=pmap.get)
        return Result("ok", choice_id=best, probabilities=pmap, confidence=max(probs), http_status=200,
                      usage={"input_tokens": n, "choice_mass": round(choice_mass, 4)}, latency_ms=latency,
                      model=self.model_id)
