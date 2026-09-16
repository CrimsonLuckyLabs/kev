"""Labeled atomic decisions. One JSONL row is one Noul, Choice, or Score.

This is not chat SFT. Rows never contain generated JSON answers.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kev.encode import state_to_text
from kev.primitives import Choice, Noul, Score, parse_question
from kev.prompt import ChatFormat, Qwen25ChatFormat
from kev.types import State

YES_LABEL = "YES"
NO_LABEL = "NO"


class QuestionSpec(BaseModel):
    """Question as stored on disk. Closed world for choice/score lives in criteria."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["noul", "choice", "score"]
    instructions: str
    criteria: dict[str, str | None] | list[str] | None = None

    @field_validator("id", "instructions")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question id and instructions must be non-empty")
        return value

    def as_primitive(self) -> Noul | Choice | Score:
        payload: dict[str, object] = {
            "type": self.type,
            "instructions": self.instructions,
        }
        if self.criteria is not None:
            payload["criteria"] = self.criteria
        parsed = parse_question(payload)
        return parsed


class LabelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    noul: float | None = Field(default=None, ge=0.0, le=1.0)
    choice: str | None = None
    level: str | None = None
    index: int | None = None


class DecisionRow(BaseModel):
    """One atomic labeled decision."""

    model_config = ConfigDict(extra="forbid")

    id: str
    state: State
    question: QuestionSpec
    label: LabelSpec
    teacher_p: float | None = Field(default=None, ge=0.0, le=1.0)
    teacher_probs: dict[str, float] | None = None
    source: str = "toy"

    @model_validator(mode="after")
    def _label_matches_type(self) -> DecisionRow:
        kind = self.question.type
        if kind == "noul":
            if self.label.noul is None:
                raise ValueError("noul rows require label.noul")
            if self.teacher_probs is not None:
                raise ValueError("noul rows use teacher_p, not teacher_probs")
        elif kind == "choice":
            if not self.label.choice:
                raise ValueError("choice rows require label.choice")
            options = self.option_labels()
            if self.label.choice not in options:
                raise ValueError("label.choice must be a supplied option")
            self._check_teacher_probs(options)
        else:
            if not self.label.level:
                raise ValueError("score rows require label.level")
            levels = self.option_labels()
            if self.label.level not in levels:
                raise ValueError("label.level must be a supplied level")
            expected_index = levels.index(self.label.level)
            if self.label.index is None:
                self.label.index = expected_index
            elif self.label.index != expected_index:
                raise ValueError("label.index must match the level position")
            self._check_teacher_probs(levels)
        return self

    def _check_teacher_probs(self, keys: Sequence[str]) -> None:
        if self.teacher_probs is None:
            return
        if set(self.teacher_probs) != set(keys):
            raise ValueError("teacher_probs keys must match the option/level set")
        if any(value < 0.0 for value in self.teacher_probs.values()):
            raise ValueError("teacher_probs must be non-negative")
        total = float(sum(self.teacher_probs.values()))
        if abs(total - 1.0) > 1e-6:
            raise ValueError("teacher_probs must sum to 1")

    def option_labels(self) -> list[str]:
        primitive = self.question.as_primitive()
        if isinstance(primitive, Noul):
            return [YES_LABEL, NO_LABEL]
        if isinstance(primitive, Choice):
            return primitive.options
        return primitive.levels

    def target_probs(self) -> list[float]:
        """Soft target aligned with option_labels(). Never a generated string."""
        labels = self.option_labels()
        if self.question.type == "noul":
            p_yes = self.teacher_p if self.teacher_p is not None else float(self.label.noul or 0.0)
            return [p_yes, 1.0 - p_yes]
        if self.teacher_probs is not None:
            return [float(self.teacher_probs[name]) for name in labels]
        winner = self.label.choice if self.question.type == "choice" else self.label.level
        return [1.0 if name == winner else 0.0 for name in labels]

    def hard_label(self) -> str:
        if self.question.type == "noul":
            return YES_LABEL if float(self.label.noul or 0.0) >= 0.5 else NO_LABEL
        if self.question.type == "choice":
            assert self.label.choice is not None
            return self.label.choice
        assert self.label.level is not None
        return self.label.level

    def prompt_text(self, chat_format: ChatFormat | None = None) -> str:
        fmt = chat_format or Qwen25ChatFormat()
        primitive = self.question.as_primitive()
        prefix = fmt.prefix(state_to_text(self.state))
        suffix = fmt.suffix_for(self.question.id, primitive)
        return prefix + suffix


@dataclass
class HashTokenizer:
    """Deterministic tokenizer for collate tests. Not a Qwen vocab."""

    pad_token_id: int = 0
    _atom: dict[str, int] = field(default_factory=dict)
    _next_atom: int = 16

    def __post_init__(self) -> None:
        self._atom.setdefault(YES_LABEL, 11)
        self._atom.setdefault(NO_LABEL, 12)
        self._atom.setdefault("Yes", 11)
        self._atom.setdefault("yes", 11)
        self._atom.setdefault("No", 12)
        self._atom.setdefault("no", 12)

    def register(self, text: str) -> int:
        if text not in self._atom:
            self._atom[text] = self._next_atom
            self._next_atom += 1
        return self._atom[text]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in self._atom:
            return [self._atom[text]]
        data = text.encode()
        if not data:
            return [1]
        ids: list[int] = []
        for index in range(0, len(data), 3):
            chunk = data[index : index + 3].ljust(3, b"\0")
            ids.append(1 + (int.from_bytes(chunk, "little") % 8000))
        return ids

    @classmethod
    def from_rows(cls, rows: Sequence[DecisionRow]) -> HashTokenizer:
        tokenizer = cls()
        for row in rows:
            for label in row.option_labels():
                tokenizer.register(label)
        return tokenizer


@dataclass
class CollatedBatch:
    """One training batch of option-token CE inputs. No generation labels."""

    row_ids: list[str]
    kinds: list[str]
    prompts: list[str]
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    last_index: list[int]
    option_labels: list[list[str]]
    option_token_ids: list[list[int]]
    target_probs: list[list[float]]

    def __len__(self) -> int:
        return len(self.row_ids)

    @property
    def seq_len(self) -> int:
        return len(self.input_ids[0]) if self.input_ids else 0


class DecisionDataset:
    """JSONL of DecisionRow. Iterable, indexable, collatable."""

    def __init__(self, rows: Sequence[DecisionRow]) -> None:
        if not rows:
            raise ValueError("DecisionDataset requires at least one row")
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> DecisionRow:
        return self.rows[index]

    def __iter__(self) -> Iterator[DecisionRow]:
        return iter(self.rows)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> DecisionDataset:
        rows: list[DecisionRow] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                text = raw.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                    rows.append(DecisionRow.model_validate(payload))
                except Exception as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
        return cls(rows)

    def to_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(row.model_dump_json() + "\n")

    def collate(
        self,
        rows: Sequence[DecisionRow] | None = None,
        tokenizer: Any | None = None,
        chat_format: ChatFormat | None = None,
        pad_token_id: int | None = None,
    ) -> CollatedBatch:
        selected = list(rows) if rows is not None else list(self.rows)
        if not selected:
            raise ValueError("cannot collate an empty row list")
        tok = tokenizer or HashTokenizer.from_rows(selected)
        pad_id = pad_token_id
        if pad_id is None:
            pad_id = int(getattr(tok, "pad_token_id", 0) or 0)
        encode = _bind_encode(tok)
        prompts: list[str] = []
        raw_ids: list[list[int]] = []
        option_labels: list[list[str]] = []
        option_token_ids: list[list[int]] = []
        target_probs: list[list[float]] = []
        kinds: list[str] = []
        row_ids: list[str] = []
        for row in selected:
            prompt = row.prompt_text(chat_format)
            ids = encode(prompt)
            if not ids:
                raise ValueError(f"tokenizer produced no ids for row {row.id}")
            labels = row.option_labels()
            token_ids = [_first_token(encode, label) for label in labels]
            prompts.append(prompt)
            raw_ids.append(ids)
            option_labels.append(labels)
            option_token_ids.append(token_ids)
            target_probs.append(row.target_probs())
            kinds.append(row.question.type)
            row_ids.append(row.id)
        max_len = max(len(ids) for ids in raw_ids)
        input_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in raw_ids]
        attention_mask = [
            [1] * len(ids) + [0] * (max_len - len(ids)) for ids in raw_ids
        ]
        last_index = [len(ids) - 1 for ids in raw_ids]
        return CollatedBatch(
            row_ids=row_ids,
            kinds=kinds,
            prompts=prompts,
            input_ids=input_ids,
            attention_mask=attention_mask,
            last_index=last_index,
            option_labels=option_labels,
            option_token_ids=option_token_ids,
            target_probs=target_probs,
        )


def _bind_encode(tokenizer: Any) -> Any:
    def encode(text: str) -> list[int]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return [int(token) for token in ids]

    return encode


def _first_token(encode: Any, text: str) -> int:
    ids = encode(text)
    if not ids:
        raise ValueError(f"tokenizer produced no ids for option {text!r}")
    return int(ids[0])
