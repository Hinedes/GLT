# Issue 3 Results

## A. Generation-Path Diagnosis

Manual shared-prefix logit diagnosis across six fixed prompts and two repeats classified the cache discrepancy as numerical greedy instability. BF16 current SDPA had 2/384 argmax flips; BF16 eager had 4/384; FP32 eager had 0/384, with maximum absolute logit error `0.000381` and no flips. The BF16 flips occurred at near-tied top-1/top-2 logits, with cosine similarity above `0.99994`. This is not evidence of a semantic position/mask/cache defect. Use `use_cache=False` as the authoritative path for capability comparisons.

Evidence: `experiments/kyrgyz_generation_authority/cache_logit_parity.json`, `CACHE_DIAGNOSIS.md`.

## B. Existing Axis Results Under Authoritative Decoding

The 60-record no-cache audit covered frozen base, Axis step 200, Axis step 2000, and LoRA step 200 with five heldout Kyrgyz, five FLORES Kyrgyz, and five English OOD prompts each. Base and LoRA-200 remained repetitive or malformed on Kyrgyz/FLORES. Axis-2000 produced some more locally grammatical heldout Kyrgyz continuations, but FLORES remained inconsistent and English collapsed to punctuation in all five examples. The previous degeneration therefore remains under the authoritative full-sequence path; it is not explained by cache mode alone.

Evidence: `experiments/kyrgyz_generation_authority/no_cache_generations.jsonl`, `no_cache_metrics.json`, `NO_CACHE_AUDIT.md`.

## C. LoRA Control Result

The conventional FFN LoRA control ran continuously to 5000 steps, seeing `2,555,000` supervised target tokens. Heldout Kyrgyz PPL improved from `11.99` at step 200 to `6.11` at step 5000; FLORES improved from `13.75` to `8.09`; English OOD worsened from `13.81` to `16.94`.

Raw no-cache generations did not show robust coherent Kyrgyz acquisition. Some step-2000/5000 samples were locally grammatical, but the set still contained copied web text, loops, malformed continuations, and no consistent capability across the three prompts. The LoRA gate is not passed. This isolates the leading remaining problem to shared corpus, packing, tokenizer, objective, exposure, or generalization quality; LoRA remains diagnostic only and is not a Grafting proposal.

Evidence: `experiments/kyrgyz_lora_exposure/metrics.json`, `generations.jsonl`, `EXPOSURE.md`.

## D. Conditional Matched Axis Result

Not run. The Issue 3 gate requires clearly coherent LoRA generation before matched Axis exposure. That condition was not met, so no new Axis training was started. Existing Axis artifacts remain unchanged.

## Corpus Audit

The CPU audit found zero exact duplicate 512-token rows, but `35/512` heldout first-64 prompts (`6.84%`) matched a training 64-token window; FLORES had `0/155`. Kyrgyz train had multiple EOS in `1.17%` of sequences, while FLORES had multiple EOS in `100%` of sequences. Sampled Kyrgyz fragmentation was `4.657` tokens/word versus `1.273` for English. These are plausible contributors to low teacher-forced PPL and poor free generation, but do not identify one dominant factor.

Evidence: `experiments/kyrgyz_corpus_audit/corpus_audit.json`, `CORPUS_AUDIT.md`.

## Runtime and Artifacts

- Phase 1 runtime: `132.075s`.
- Phase 2 runtime: `92.504s` after the autocast fix; the initial standalone script failed on BF16/FP32 hook dtype mismatch and the exact failure is documented in `NO_CACHE_AUDIT.md`.
- Phase 3 runtime: `3659.866s`; peak allocated VRAM `12761 MiB`, reserved `18742 MiB`.
- Train SHA-256: `c63f76b1f870525c651f2743df54b0b6f75f7e26d02fc359219a88f309c03d72`.
- Heldout SHA-256: `fe5de6d922da889064f3ef22d10951e61d7387015142257f1a0ab60ffe66783e`.
- FLORES SHA-256: `4439348a87c824cbd32bb38ddb3968251bc4251051d9a22518a5ebb20101d393`.
- English OOD SHA-256: `79cabc3f05659bcb5a7b5e33e5260b1a40759c1b590dc604cb295116d1d51ea6`.
- LoRA checkpoint and optimizer hashes are recorded per checkpoint in `experiments/kyrgyz_lora_exposure/metrics.json`; large tensors remain outside Git.
