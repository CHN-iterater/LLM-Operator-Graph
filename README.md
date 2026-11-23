# OperatorGraph

Build call-level operator graphs from local Hugging Face weights or ONNX. Preserve tensor configurations, dependencies, and static estimation rules, and produce JSON, CSV, and Markdown reports.

## Install and quick start

Requires Python >=3.10. Run the following commands from the project root.

```shell
python -m pip install ".[export]"
operator-graph export "D:\Models\YourModel" --prompt "What's the capital of France?" --output output/your-model
```

The model directory must contain weights, `config.json`, and tokenizer files. One run loads the model, tokenizes the prompt, exports ONNX, saves graph JSON, and generates complete reports. Some models require additional tokenizer dependencies.

The program uses token IDs for two independent forwards: the full prompt (`full_prompt`) and a single-token probe (`single_token_probe`). Both run in eval mode with KV cache disabled. The probe receives only the original prompt's last token, with no past key/value states; it does not represent cached decoding.

Loading uses local files by default. Use `--allow-download` to download model files and `--trust-remote-code` to execute custom model code. See `operator-graph export --help` for all options.

## Other entry points

Choose dependencies for your input:

| Purpose | Install command |
|---|---|
| Offline analysis and validation of existing JSON | `python -m pip install .` |
| Graph construction from existing ONNX | `python -m pip install ".[onnx]"` |
| Model weights to graphs and complete reports | `python -m pip install ".[export]"` |

```shell
operator-graph parse model.onnx --output output/graph.json
operator-graph analyze examples/graph.json --output output/reports
operator-graph validate examples/graph.json
```

`analyze` also accepts directories containing `graph.json` or `*_graph.json`. Offline JSON analysis needs no model weights, GPU, ONNX, or PyTorch. You can use `python -m operator_graph` in place of `operator-graph`.

## Outputs and estimation conventions

Main outputs from the model-weights entry point:

```text
output/your-model/
  model_config.json
  full_prompt.onnx
  single_token_probe.onnx
  full_prompt_graph.json
  single_token_probe_graph.json
  reports/
    summary.md / model_metrics.csv / shape_signature_catalog.csv
    validation_summary.json / validation_summary.md
    <separate directory for each input condition>/
      metrics.json / metrics.md
      node_metrics.csv / operator_metrics.csv / layer_metrics.csv
      stage_metrics.csv / level_widths.csv
      shape_signature_metrics.csv / cast_metrics.csv / dtype_metrics.csv
```

Large models may also produce ONNX external-weight files; graphs record retained files and hashes. Call records preserve raw operator types and ordered input/output shapes and dtypes, distinguishing repeated operand uses from deduplicated call-dependency edges.

Arithmetic work, logical data access, and arithmetic intensity (AI) follow the declared `static-v2.0` rules, with formulas, assumptions, and availability states retained. `null`/`NA` means unavailable; complete totals and known-part totals are reported separately. These are static estimates, not measured energy consumption or device traffic.

Graph construction covers top-level ONNX calls without expanding nested control flow. Topological width describes logical dependencies; layer labels are heuristic.

## Validation and detailed documentation

Current validation results:

- 74 tests pass, covering construction, estimation, serialization, the CLI, and local HF weights through report generation.
- Archived recomputation for seven models and fourteen forwards matches the paper, including 26,074 calls and key figure/table statistics.
- Two graphs re-exported from local Qwen3-0.6B weights match the archives in configurations, dependencies, and estimates for all 4,840 calls. The other six models have not been re-exported from weights.

Run the full suite, including export dependencies:

```shell
python -m pip install ".[test,export]"
python -m pytest
```

Further details:

- [Usage notes, project structure, and migration validation](docs/migration_report.md)
- [Paper-to-code alignment](docs/paper_alignment.md)
- [Test and installation verification record](docs/verification.json)

Migrated from the research working tree of [LLM-Graph-Parser](https://github.com/CHN-iterater/LLM-Graph-Parser) for the paper *Operator Graph Construction for Large Language Model Inference*.
