# Readability and behavior-preserving cleanup

**Goal:** Make the implementation easier to review for the paper artifact without changing public behavior, numerical rules, report fields, or input conditions.

## Constraints

- Keep comments, docstrings, documentation, and report prose in English.
- Preserve the local HF weights-to-JSON-and-complete-reports workflow.
- Preserve static-v2.0 rules, unavailable values, known zeros, raw types, operand order, and graph dependencies.
- Keep CLI arguments, JSON schemas, report filenames, labels, and error behavior stable.
- Do not modify the original parser repository, paper, or archived inputs.

## Work items

- [x] Record the current source commit, full test baseline, and Qwen ONNX parser/report snapshots.
- [x] Review parser, serialization, reporting, and export helpers for duplication, compressed expressions, redundant comments, and unclear responsibilities.
- [x] Apply small, focused refactorings. Prefer names and explicit control flow over cosmetic churn or new abstractions.
- [x] Review each changed module against its previous behavior and existing regression cases.
- [x] Run the complete suite, local HF integration test, seven-model archive regression, paper table/figure audit, and retained Qwen ONNX comparison.
- [x] Build and install a wheel in the core-only environment; check English report generation and source parity.
- [x] Record exact outcomes and commit the reviewed changes locally.

No speedup or general compatibility claim will be made without a corresponding measurement or test.
