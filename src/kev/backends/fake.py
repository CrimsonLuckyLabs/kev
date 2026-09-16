"""Injectable logits engine for tests. No GPU. No generate()."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from kev.infer import PrefillCache


class FakeLogitsEngine:
    """Deterministic next-token logits keyed by prefix+suffix. Never decodes."""

    name = "fake"
    vocab_size = 128

    def __init__(
        self,
        token_table: dict[str, list[int]] | None = None,
        logit_overrides: dict[str, dict[int, float]] | None = None,
        continuation_scores: dict[str, float] | None = None,
    ) -> None:
        self.token_table = {
            "YES": [1],
            "Yes": [2],
            "yes": [3],
            "NO": [4],
            "No": [5],
            "no": [6],
            **(token_table or {}),
        }
        self.logit_overrides = logit_overrides or {}
        self.continuation_scores = continuation_scores or {}
        self.prefill_calls = 0
        self.sequential_calls = 0
        self.batch_calls = 0
        self.continuation_calls = 0

    def encode(self, text: str) -> list[int]:
        if text in self.token_table:
            return list(self.token_table[text])
        digest = hashlib.sha256(text.encode()).digest()
        return [10 + digest[0] % 80]

    def prefill(self, prefix_text: str) -> PrefillCache:
        self.prefill_calls += 1
        return PrefillCache(
            prefix_text=prefix_text,
            n_tokens=max(1, len(prefix_text) // 4),
            payload=None,
        )

    def next_logits(self, cache: PrefillCache, suffix_text: str) -> list[float]:
        self.sequential_calls += 1
        return self._logits_for(cache.prefix_text, suffix_text)

    def next_logits_batch(self, cache: PrefillCache, suffixes: Sequence[str]) -> list[list[float]]:
        self.batch_calls += 1
        return [self._logits_for(cache.prefix_text, suffix) for suffix in suffixes]

    def continuation_logprob(
        self, cache: PrefillCache, suffix_text: str, continuation: str
    ) -> float:
        self.continuation_calls += 1
        for key in (f"{suffix_text}\0{continuation}", continuation):
            if key in self.continuation_scores:
                return self.continuation_scores[key]
        key = f"{cache.prefix_text}\0{suffix_text}\0{continuation}"
        digest = hashlib.sha256(key.encode()).digest()
        return -((digest[0] + 1) / 64.0)

    def _logits_for(self, prefix_text: str, suffix_text: str) -> list[float]:
        vec = [0.0] * self.vocab_size
        digest = hashlib.sha256(f"{prefix_text}\0{suffix_text}".encode()).digest()
        vec[1] = 1.5 + digest[0] / 255.0
        vec[2] = 0.4
        vec[3] = 0.2
        vec[4] = 1.0 + digest[1] / 255.0
        vec[5] = 0.3
        vec[6] = 0.1
        for index in range(10, self.vocab_size):
            vec[index] = (digest[index % 32] / 255.0) - 0.5
        overrides = self.logit_overrides.get(suffix_text) or self.logit_overrides.get("default")
        if overrides:
            for token_id, value in overrides.items():
                if 0 <= token_id < self.vocab_size:
                    vec[token_id] = value
        return vec
