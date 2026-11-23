# OperatorGraph migration implementation plan

> Execute the user-approved local migration with separate implementation tasks and review. Original repository and paper artifacts are read-only baselines.

**Goal:** Deliver an independently installable operator-graph project aligned with the current manuscript.

**Architecture:** A standard-library data/analysis core with optional ONNX parsing and PyTorch export. Keep static-v2.0 formulas stable and read legacy schema 1.0, while writing a versioned schema with explicit unavailable estimates and input conditions. Preserve original provenance.

**Tech stack:** Python >=3.10, setuptools, pytest; optional ONNX and PyTorch/Transformers.

## Constraints

- New directory OperatorGraph and import package operator_graph; no old Git remote.
- No changes to LLM-Graph-Parser, original outputs, manuscript, Chinese manuscript or Overleaf package.
- Inputs remain full_prompt and independent single_token_probe with KV cache disabled; no energy measurement claims.
- Operations/access are static estimates, known zero differs from unavailable, and weighted paths preserve zero.
- No cached-decode, dynamic-shape or zero-length-tensor support claimed as manuscript-validated.

## Tasks

- [x] 1. Write focused failing contract tests in tests/test_paper_contract.py. Run pytest with src import path to confirm missing implementation.
- [x] 2. Migrate static estimation and split analysis/reports into estimation/rules.py, analysis/{aggregation,topology,workloads,precision,engine}.py, reports/{writer,batch}.py. Adapt quantitative/workload regression tests without changing expectations except obsolete run.py integration. Verify current formula values.
- [x] 3. Implement model records/graph, schema-1 reader and schema-2 writer, ONNX metadata and call construction. Cover repeated operands, optional inputs, unknown shape/dtype, static scope, known zero/None and invalid graphs.
- [x] 4. Implement lazy PyTorch export, model-loading adapter and CLI parse/export/analyze/validate. Verify export failure state restoration and local ONNX examples; no model downloads needed for migration.
- [x] 5. Implement independent validation and optional layer grouping; migrate registry/layer tests where applicable. Keep legacy calculators, FX hooks and hardware assumptions outside the default core; document their disposition.
- [x] 6. Compare all seven archived graphs: new/old engine reports, all 26,074 node estimates, statuses and aggregates; verify source hashes and original paper artifacts unchanged.
- [x] 7. Package/install in an isolated environment, run tests, execute CLI parse/analyze/validate example, inspect wheel, review architecture and paper alignment. Record exact results and limitations, then update README and migration report.

## Acceptance

`python -m pytest` passes; core imports/JSON analysis work without torch, transformers or onnx; hand-checked ONNX graph keeps duplicate operand uses but deduplicates edges. Unknown estimates serialize as null, incomplete totals remain unavailable. Cycles and dangling edges are rejected. Archive regression reports no unapproved formula drift. Original source and publication artifacts match inventory baselines. New Git repository has no remote until the new GitHub destination is chosen.

## Validation record

See docs/verification.json and docs/migration_verification.json. The local HF-weight pipeline additionally creates complete reports automatically and has a real safetensors/config/tokenizer integration test. Independent core and cross-module review findings were addressed with failing-then-passing regressions. No original repository or paper artifact changed.
