# Readability review and regression check

Date: 2026-11-23. Baseline source commit: `3aff052`.

## Changes

- Separated ONNX metadata collection, call-record construction, and dependency linking. Names now identify tensors, operands, and producers directly, and packed assignments use explicit statements.
- Avoided creating unknown `TensorMeta` defaults for operands that already have metadata. Missing operands still receive separate unknown records, preserving their original behavior.
- Reused a node-list snapshot during topological sorting and cached layer information during serialization.
- Reused layer-count and total-work results in replay checks, while retaining the original methods and independent validation paths.
- Separated Markdown formatting from batch analysis and JSON/CSV output. Kept the original text, file names, field ordering, and the distinct `.4f` and `.4g` formats.
- Retained the original optional-dependency error order in `OnnxParser.parse()` and added a regression test for it.

The static estimation rules, CLI arguments, schemas, input conditions, report data, and error messages were not changed. The review does not claim a measured execution-time speedup.

## Verification

| Check | Result |
|---|---|
| Baseline test suite | 73 passed |
| Final suite, including the added compatibility test | 74 passed |
| Real local small-HF-model weights to ONNX, JSON, and reports | Passed in the final suite |
| Report outputs before and after cleanup | All 15 files byte-for-byte identical |
| Seven-model archive regression | All 14 forwards and 26,074 paper call estimates/states unchanged |
| Independent paper audit | Table 2/3 values, Figure 4 counts, 936 groups, and Mistral's 32 repeated blocks matched |
| Reparse both retained Qwen ONNX exports | All 4,840 call records and full reports matched the pre-refactor snapshots |
| Core-only wheel installation and English CLI reports | See `verification.json` |

Qwen graph comparisons exclude only the serialization timestamp. Fresh exports of all seven large models were not rerun in this review. The Qwen check used the retained ONNX files from the preceding real-weight reproduction; the final suite reran actual export from small local HF weights.

The original source repository, paper files, CSV data, and archived graphs retain their baselines. Local detailed logs and comparison hashes are under `output/readability_review/`; the summary is recorded in `verification.json`.
