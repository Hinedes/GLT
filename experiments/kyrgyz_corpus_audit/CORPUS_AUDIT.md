# Kyrgyz Corpus Audit

CPU-only structural audit; statistics are diagnostic and not capability claims.

```json
{
  "model": "/workspace/model/real_SmolLM3-3B",
  "sequence_lengths": {
    "kyrgyz_train": 512,
    "kyrgyz_heldout": 512,
    "kyrgyz_flores": 512,
    "english_ood": 512
  },
  "binary_sha256": {
    "kyrgyz_train": "c63f76b1f870525c651f2743df54b0b6f75f7e26d02fc359219a88f309c03d72",
    "kyrgyz_heldout": "fe5de6d922da889064f3ef22d10951e61d7387015142257f1a0ab60ffe66783e",
    "kyrgyz_flores": "4439348a87c824cbd32bb38ddb3968251bc4251051d9a22518a5ebb20101d393",
    "english_ood": "79cabc3f05659bcb5a7b5e33e5260b1a40759c1b590dc604cb295116d1d51ea6"
  },
  "duplicate_stats": {
    "kyrgyz_train": {
      "sequences": 4096,
      "unique_sequences": 4096,
      "duplicate_rows": 0,
      "duplicate_rate": 0.0,
      "max_duplicate_count": 1
    },
    "kyrgyz_heldout": {
      "sequences": 512,
      "unique_sequences": 512,
      "duplicate_rows": 0,
      "duplicate_rate": 0.0,
      "max_duplicate_count": 1
    },
    "kyrgyz_flores": {
      "sequences": 155,
      "unique_sequences": 155,
      "duplicate_rows": 0,
      "duplicate_rate": 0.0,
      "max_duplicate_count": 1
    },
    "english_ood": {
      "sequences": 4096,
      "unique_sequences": 4096,
      "duplicate_rows": 0,
      "duplicate_rate": 0.0,
      "max_duplicate_count": 1
    }
  },
  "boundary_stats": {
    "kyrgyz_train": {
      "eos_token_id": 128001,
      "sequences": 4096,
      "mean_eos_per_sequence": 0.287109375,
      "sequences_with_eos": 1128,
      "sequences_with_eos_rate": 0.275390625,
      "sequences_with_multiple_eos": 48,
      "multiple_eos_rate_proxy_for_multiple_documents": 0.01171875,
      "max_eos_per_sequence": 2
    },
    "kyrgyz_heldout": {
      "eos_token_id": 128001,
      "sequences": 512,
      "mean_eos_per_sequence": 0.28515625,
      "sequences_with_eos": 139,
      "sequences_with_eos_rate": 0.271484375,
      "sequences_with_multiple_eos": 7,
      "multiple_eos_rate_proxy_for_multiple_documents": 0.013671875,
      "max_eos_per_sequence": 2
    },
    "kyrgyz_flores": {
      "eos_token_id": 128001,
      "sequences": 155,
      "mean_eos_per_sequence": 6.483870967741935,
      "sequences_with_eos": 155,
      "sequences_with_eos_rate": 1.0,
      "sequences_with_multiple_eos": 155,
      "multiple_eos_rate_proxy_for_multiple_documents": 1.0,
      "max_eos_per_sequence": 10
    },
    "english_ood": {
      "eos_token_id": 128001,
      "sequences": 4096,
      "mean_eos_per_sequence": 0.552734375,
      "sequences_with_eos": 1778,
      "sequences_with_eos_rate": 0.43408203125,
      "sequences_with_multiple_eos": 431,
      "multiple_eos_rate_proxy_for_multiple_documents": 0.105224609375,
      "max_eos_per_sequence": 4
    }
  },
  "exact_prompt_overlap": {
    "kyrgyz_heldout": {
      "sequences_checked": 512,
      "exact_first_64_prompt_matches_in_train_windows": 35,
      "exact_first_64_prompt_match_rate": 0.068359375
    },
    "kyrgyz_flores": {
      "sequences_checked": 155,
      "exact_first_64_prompt_matches_in_train_windows": 0,
      "exact_first_64_prompt_match_rate": 0.0
    }
  },
  "fragmentation": {
    "kyrgyz_train": {
      "sample_sequences": 128,
      "sample_tokens": 32768,
      "decoded_chars": 56816,
      "decoded_words": 7036,
      "tokens_per_word": 4.657191586128482,
      "tokens_per_char": 0.5767389467755561,
      "replacement_char_count": 24
    },
    "english_ood": {
      "sample_sequences": 128,
      "sample_tokens": 32768,
      "decoded_chars": 154468,
      "decoded_words": 25748,
      "tokens_per_word": 1.2726425353425508,
      "tokens_per_char": 0.21213455214024912,
      "replacement_char_count": 0
    }
  },
  "selection": {
    "overlap_window_tokens": 64,
    "fragmentation_sample_tokens_per_sequence": 256
  }
}
```

## Interpretation

- Full 512-token sequence duplicates were zero in all four bins.
- `35/512` heldout first-64 prompts (`6.84%`) exactly matched a 64-token window somewhere in training; FLORES had `0/155` such matches. This is nontrivial heldout leakage for the Kyrgyz crawl and can inflate heldout PPL without proving generation capability.
- Kyrgyz train had EOS in `27.54%` of sequences and multiple EOS in `1.17%`; English had EOS in `43.41%` and multiple EOS in `10.52%`. FLORES had multiple EOS in `100%` of sequences, confirming that its fixed bins contain many sentence/document boundaries.
- The sampled Kyrgyz tokenizer fragmentation was `4.657` tokens/decoded word versus `1.273` for English, and `0.577` tokens/character versus `0.212`. Kyrgyz therefore receives substantially more token steps per surface unit, making fixed-step exposure and PPL comparisons asymmetric.
- The audit supports shared corpus/packing/tokenizer concerns as plausible contributors to low PPL plus poor free generation. It does not by itself establish which factor dominates.
