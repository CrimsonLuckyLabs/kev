# Why Kev scores option tokens instead of generating JSON

Kev is System One: unstructured state plus typed questions in, distributions out.
The model never writes the answer. Python does.

## The rejected design

A common wrapper is:

1. Prompt the model to emit JSON (`{"tone": "angry", ...}`).
2. Call `model.generate()`.
3. Parse, retry, or regex the text until it looks valid.

That fights the contract:

- **Invalid labels become possible.** The model can invent `"furious"` when the option set is `{calm, frustrated, angry}`. Constrained decoding still generates; it only filters tokens. Kev does not generate at all.
- **Control flow leaves Python.** Retries, “fix your JSON” loops, and schema repair are chat behavior. Code must own routing; Kev only judges.
- **Many questions become many generate loops.** Shared state is re-encoded, `output_tokens` is not zero, and adding a Noul starts a new decode.
- **Training becomes chat SFT.** The student learns to emit documents. That is the opposite of scoring a closed option set.

Teacher distillation may use JSON. The teacher is not Kev. `rationale_internal` is discarded. The student is still trained and served as option-token classification.

## What Kev does instead

1. Encode the state once (ChatML prefix). Prefill once.
2. For each question, attach a suffix that ends where the option token would be.
3. Read **next-token logits**. Softmax only over the developer-supplied option ids:
   - **Noul:** YES/NO mass, renormalized to P(true) in `[0, 1]`.
   - **Choice / Score:** first tokenizer id of each label. If two labels share a first token, teacher-force the **full option string** and softmax length-normalized logprobs. Still not decode.
4. Python assembles `NoulAnswer` / `ChoiceAnswer` / `ScoreAnswer`. `output_tokens` is always `0`.

Invalid labels are structurally impossible: the argmax is taken over the supplied keys, never over the vocabulary.

## Confidence

Confidence is a number (`1 - H(p) / log(K)` for Choice/Score). Application code thresholds it. Kev does not write an essay about uncertainty.

## `generate()` is a bug

If a code path calls `model.generate()` to produce an answer, delete it. Hugging Face and SFT backends monkey-patch `generate` to raise. Mock never had it.
