# Local migration and paper alignment

Date: 2026-11-23. The project directory is `OperatorGraph` and the Python package is `operator_graph`. A separate local Git repository has been initialized, with no remote configured and no GitHub push.

## Migrated workflows

1. **Local Hugging Face model directory to JSON and complete reports**: load standard weights/config/tokenizer files, tokenize, and separately export the full prompt and independent single-token probe. Save ONNX, external-weight provenance hashes, graph JSON, and the original model configuration. Automatically produce per-call, operator-type, layer-label, configuration, precision, and topology reports for both conditions, plus combined summaries and validation results.
2. Existing ONNX to call-level graph JSON.
3. Existing new or legacy JSON to static estimates and complete offline reports.
4. JSON to completeness, serialization, aggregation, independent path-reference, and optional layer-partition replay checks.

See the [README](../README.md) for commands and output directories. `export` is the end-to-end weights-to-reports entry point. It does not generate text or download models by default.

## Code organization

Paths in the new-location column are relative to `src/operator_graph/`.

| Original implementation | New location |
|---|---|
| Export orchestration in `run.py` and package entry points | `exporters/torch.py`, `exporters/prompt.py`, `cli.py` |
| `core/operator_node.py` and basic graph operations | `model/records.py`, `model/graph.py` |
| `core/operator_registry.py` | `model/registry.py` (legacy category strings remain display labels, not measured bottleneck classifications) |
| `parser/onnx_parser.py` | `parsers/onnx.py`, `parsers/op_names.py` |
| Static estimation rules in `metrics.py` | `estimation/rules.py` |
| Aggregation and structural/configuration/precision analysis in `metrics.py` | `analysis/aggregation.py`, `analysis/topology.py`, `analysis/workloads.py`, `analysis/precision.py`, `analysis/engine.py` |
| Reporting and batch processing in `metrics.py` | `reports/writer.py`, `reports/batch.py` |
| `core/serialization.py` | `serialization/__init__.py` |
| `offline_validation.py` | `validation/replay.py` |
| `LayerPartitioner` | `extensions/layers.py` |

The original `.git` directory, legacy `utils` estimators, hardware collection, energy fitting, and historical manuscript/plot scripts were not copied into the new core. Migrated code retains source attribution. A publication license has not yet been selected.

## Intentional behavior corrections

- Unavailable estimates use `None`/`null`; complete totals and known-part totals remain separate.
- Explicit states in legacy `static_cost` records take precedence over numeric zero placeholders. Legacy records without states are not assigned semantics from their numeric values alone; offline analysis explicitly recomputes estimates.
- Weighted paths retain true zero weights; unavailable weights do not produce a falsely complete path.
- Validated graph interfaces reject cycles, dangling dependencies, asymmetric parent/child links, and duplicate IDs. Analysis of raw dictionaries can still produce diagnostics.
- Scalars, unknown rank, and zero dimensions remain distinct. The paper's evaluated scope is unchanged.
- Input names, optional slots, repeated operand uses, and deduplicated dependency edges are preserved together.
- Custom ONNX domains do not inherit rules from same-named standard operators. Estimates for the seven standard-domain archives are unchanged.
- Schema 2 rejects writes that omit metadata needed to preserve raw types, attributes, and other provenance, preventing display aliases from changing recomputation.
- Layer validation handles unavailable totals, and legacy tensor records normalize optional default fields correctly.
- Directory analysis recognizes both export output filenames and gives each condition its own report directory.
- Temporary ONNX cleanup removes stale source paths while retaining ONNX hashes and `source_retained` declarations. Model training/cache states are restored after success or failure. Export does not globally modify environment variables, warnings, or pytree registration.

The `static-v2.0` arithmetic and logical-access formulas are preserved. These corrections concern input scope, unavailable states, graph validity, and interface organization; they do not change archived paper results. See [paper alignment](paper_alignment.md) for the detailed comparison.

## Validation evidence

- [verification.json](verification.json): latest tests, wheel build, installation without optional dependencies, environment versions, and Git-state checks.
- [migration_verification.json](migration_verification.json): complete reports for seven archives and fourteen forwards agree with the original implementation. Arithmetic work, logical data access, and states agree call by call for all 26,074 paper calls.
- [test_hf_weights_pipeline.py](../tests/test_hf_weights_pipeline.py): creates and saves a small local GPT-2 model with config, tokenizer, and safetensors, then uses public model-loading interfaces for two actual ONNX exports, JSON, and complete reports. It downloads no model.
- [test_export_contract.py](../tests/test_export_contract.py): exports a real small PyTorch module and checks success/failure state restoration, external-file cleanup, default local loading, input conditions, and automatic report generation.
- [test_paper_contract.py](../tests/test_paper_contract.py): regression cases for core paper semantics and migration fixes.
- [test_report_language.py](../tests/test_report_language.py): verifies English summary and validation reports while checking unchanged numerical fields. All project comments, docstrings, documentation, and generated report prose use English; source-path and provenance values remain data and are preserved.
- After wheel installation into an environment without optional dependencies, JSON `analyze` and `validate` commands both passed without `torch`, `onnx`, or `transformers` installed.
- A separate check reran both forwards from complete local Qwen3-0.6B HF weights: a 7-token full prompt and a 1-token independent probe, each with 2,420 calls. All 4,840 calls matched the paper archives in raw types, ordered input/output shapes and dtypes, parent/child dependencies, estimates, formulas, states, and all stage statistics. Legacy probe IDs were aligned by removing the `dc_` prefix.
- Seven-model archive validation also matched all 42 numeric cells in Table 2, 28 cells in Table 3, 50 counts in Figure 4, 936 configuration groups, and 32 Mistral repeated blocks. Reproducible scripts and results are retained in the original workspace's analysis verification directory.

Weights-to-reports validation covers the locally generated small HF model and actual Qwen3-0.6B weights. The other six models were checked using archived JSON and were not re-exported from weights. Models must load with the installed dependency versions and support ONNX export; validation does not cover arbitrary custom models.

## Readability review

The behavior-preserving cleanup and its 74-test regression are recorded in [readability_review.md](readability_review.md).

## Additional usage

### Python API

```python
from operator_graph import parse_onnx, graph_to_json
from operator_graph.analysis.engine import analyze_graph
from operator_graph.reports.writer import write_reports

graph = parse_onnx("model.onnx", model_name="example")
graph_to_json(graph, output_path="output/graph.json")
write_reports(analyze_graph(graph.to_dict()), "output/reports")
```

Run the complete workflow from model weights:

```python
from operator_graph.exporters import export_prompt

graphs = export_prompt("path/to/local-hf-model", "What's the capital of France?", "output/model")
```

### ONNX example without model weights

After installing `.[onnx]`, generate a three-call graph with branching and merging:

```text
python examples/make_branch_graph.py output/branch.onnx
operator-graph parse output/branch.onnx --output output/graph.json
operator-graph analyze output/graph.json --output output/reports
operator-graph validate output/graph.json
```

### Legacy JSON and local archive regression

New output uses schema 2.0; reading legacy schema 1.0 is supported. Legacy JSON retains its `prefill`/`decode` labels. Use `analyze --decode-is-probe` to declare probe semantics only after confirming that the records came from two independent cache-disabled forwards.

```text
python tools/verify_migration.py <original-paper-workspace> --output output/migration.json
```

This regression tool requires the original workspace's seven archives, paper CSV files, and legacy implementation as references. Normal graph construction and analysis in the new project do not depend on the original repository.

### Package layout

```text
src/operator_graph/
  model/          # Graphs, calls, tensors, display registry
  parsers/        # ONNX metadata and two-pass graph construction
  exporters/      # PyTorch export and HF model input adaptation
  estimation/     # Static estimation rules
  analysis/       # Aggregation, topology, configuration, precision
  serialization/  # Versioned JSON reading and writing
  reports/        # Reports and batch analysis
  validation/     # Completeness and replay checks
  extensions/     # Optional layer partitioning
```

## Environment notes

Local validation used Python 3.13.9, PyTorch 2.12.1, ONNX 1.22.0, ONNXScript 0.7.1, Transformers 5.13.0, and pytest 8.4.2. Consult [verification.json](verification.json) for exact versions. Export requires `torch>=2.6` for the modern export API. Other declared minimum versions were not each tested in this run.

Optional torch tests on this machine's Anaconda installation require `KMP_DUPLICATE_LIB_OK=TRUE` in the test process, as in the original environment. Production code does not set this variable. Upstream copyreg/Swig deprecation warnings and pyreadline3 exit messages remain in the test record; the run is not described as warning-free. Core installation and analysis without optional dependencies need no such setting.

## Publication steps

The local migration is complete. Repository name, visibility, and license remain to be selected before configuring a new remote and pushing. The original repository's remote is unchanged. The original English/Chinese manuscripts, PDF, ZIP, CSV files, seven archives, and source code retain their hash baseline.
