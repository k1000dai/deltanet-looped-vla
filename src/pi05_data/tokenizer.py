"""Exact PaliGemma SentencePiece tokenizer matching `mme_vla_suite/training/config.py:PaligemmaTokenizer`."""

from __future__ import annotations

import os
from pathlib import Path
import urllib.request
from typing import Sequence

import numpy as np
import sentencepiece
import torch

DEFAULT_MIYABI_TOKENIZER_PATH = "/work/gj26/b20098/checkpoints/pi05_init/paligemma_tokenizer.model"
GCS_PALIGEMMA_TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"


def resolve_paligemma_tokenizer_model(path: str | Path | None = None) -> Path:
    """Resolve local or Miyabi `paligemma_tokenizer.model` path (downloading from GCS only if absent)."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    env_p = os.getenv("PALIGEMMA_TOKENIZER_PATH")
    if env_p:
        candidates.append(Path(env_p))
    candidates.extend(
        [
            Path(DEFAULT_MIYABI_TOKENIZER_PATH),
            Path.home() / ".cache" / "openpi" / "big_vision" / "paligemma_tokenizer.model",
            Path("/tmp/paligemma_tokenizer.model"),
        ]
    )
    for c in candidates:
        if c.exists() and c.stat().st_size > 100_000:
            return c

    target = Path.home() / ".cache" / "openpi" / "big_vision" / "paligemma_tokenizer.model"
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(GCS_PALIGEMMA_TOKENIZER_URL, str(target))
    return target


class Pi05Tokenizer:
    """SentencePiece tokenizer reproducing `mme_vla_suite/training/config.py:PaligemmaTokenizer` verbatim.

    Modes:
      - `discrete_state_input=False, subgoal=None` (**exact `Yinpei/pi05_baseline` & `MME-VLA` default**, `max_len=64`):
        `encode(cleaned_text, add_bos=True) + encode("\\n")`
      - `discrete_state_input=False, subgoal="..."` (**exact `MME-VLA` Symbolic / GroundSG format**, `max_len=128`):
        `encode(f"Task: {cleaned_text};\\nCurrent Subgoal: {subgoal};\\nAction: ", add_bos=True)`
      - `discrete_state_input=True, subgoal=None`:
        `encode(f"Task: {cleaned_text}; State: {state_str};\\nAction: ", add_bos=True)`
    """

    def __init__(
        self,
        max_len: int = 64,
        model_path: str | Path | None = None,
    ) -> None:
        self.max_len = int(max_len)
        resolved = resolve_paligemma_tokenizer_model(model_path)
        self.model_path = str(resolved)
        self._sp = sentencepiece.SentencePieceProcessor(model_proto=resolved.read_bytes())

    @property
    def vocab_size(self) -> int:
        return int(self._sp.vocab_size())

    @staticmethod
    def clean_text(text: str, *, lower: bool = True) -> str:
        s = (text or "").strip().replace("_", " ").replace("\n", " ")
        return s.lower() if lower else s

    def tokenize_one(
        self,
        prompt: str,
        *,
        state_norm: np.ndarray | None = None,
        subgoal: str | None = None,
        max_len: int | None = None,
        lower_prompt: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        limit = self.max_len if max_len is None else int(max_len)
        cleaned = self.clean_text(prompt, lower=lower_prompt)

        if state_norm is not None:
            if subgoal is None:
                st = np.clip(np.asarray(state_norm, dtype=np.float32), -1.0, 1.0)
                disc = np.digitize(st, bins=np.linspace(-1.0, 1.0, 256 + 1)[:-1]) - 1
                state_str = " ".join(map(str, disc.tolist()))
                full_prompt = f"Task: {cleaned}; State: {state_str};\nAction: "
                toks = self._sp.encode(full_prompt, add_bos=True)
            else:
                sg = self.clean_text(subgoal, lower=True)
                full_prompt = f"Task: {cleaned}\nCurrent Subgoal: {sg}.\nAction: "
                toks = self._sp.encode(full_prompt, add_bos=True)
        elif subgoal is not None:
            sg = self.clean_text(subgoal, lower=True)
            full_prompt = f"Task: {cleaned};\nCurrent Subgoal: {sg};\nAction: "
            toks = self._sp.encode(full_prompt, add_bos=True)
        else:
            toks = self._sp.encode(cleaned, add_bos=True) + self._sp.encode("\n")

        n = len(toks)
        if n < limit:
            pad_len = limit - n
            tok_arr = np.asarray(toks + [0] * pad_len, dtype=np.int32)
            mask_arr = np.asarray([True] * n + [False] * pad_len, dtype=np.bool_)
        else:
            tok_arr = np.asarray(toks[:limit], dtype=np.int32)
            mask_arr = np.ones((limit,), dtype=np.bool_)
        return tok_arr, mask_arr

    def tokenize_batch(
        self,
        prompts: Sequence[str],
        *,
        max_len: int | None = None,
        lower_prompt: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        limit = self.max_len if max_len is None else int(max_len)
        toks_out = np.zeros((len(prompts), limit), dtype=np.int32)
        masks_out = np.zeros((len(prompts), limit), dtype=np.bool_)
        for i, p in enumerate(prompts):
            t, m = self.tokenize_one(p, max_len=limit, lower_prompt=lower_prompt)
            toks_out[i] = t
            masks_out[i] = m
        return torch.from_numpy(toks_out), torch.from_numpy(masks_out)

    def tokenize_subgoals(
        self,
        subgoals: Sequence[str],
        *,
        max_len: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deduplicated fast batch tokenization of standalone `grounded_subgoal` strings `(N, max_len)`."""
        limit = int(max_len)
        n_total = len(subgoals)
        toks_out = np.zeros((n_total, limit), dtype=np.int32)
        masks_out = np.zeros((n_total, limit), dtype=np.bool_)

        cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for i, raw in enumerate(subgoals):
            s = self.clean_text(raw, lower=True)
            if not s or s in ("none", "no record", "null"):
                continue
            cached = cache.get(s)
            if cached is None:
                ids = self._sp.encode(s, add_bos=False, add_eos=True)
                k = len(ids)
                if k < limit:
                    t_arr = np.asarray(ids + [0] * (limit - k), dtype=np.int32)
                    m_arr = np.asarray([True] * k + [False] * (limit - k), dtype=np.bool_)
                else:
                    t_arr = np.asarray(ids[:limit], dtype=np.int32)
                    m_arr = np.ones((limit,), dtype=np.bool_)
                cached = (t_arr, m_arr)
                cache[s] = cached
            toks_out[i], masks_out[i] = cached

        return torch.from_numpy(toks_out), torch.from_numpy(masks_out)
