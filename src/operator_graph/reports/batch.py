"""Analyze graph.json archives and write per-model reports and a joint catalog."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from operator_graph.estimation.rules import COST_MODEL_VERSION
from operator_graph.analysis.engine import analyze_graph
from .writer import write_reports, _write_csv, _flat_stages

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="graph.json or directory containing graph.json files")
    parser.add_argument("--output", type=Path, default=Path("metrics_output"))
    parser.add_argument("--decode-is-probe", action="store_true", help="Explicitly identify legacy decode stage as a no-cache probe")
    parser.add_argument("--validate-replay", action="store_true", help="Test structural roundtrip, aggregation and layer-code replay without weights")
    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    files = (sorted(set(args.input.rglob('graph.json')) | set(args.input.rglob('*_graph.json')))
             if args.input.is_dir() else [args.input])
    if not files:
        parser.error("No graph.json or *_graph.json files found")
    all_rows, destinations, validations = [], set(), []
    catalog = {}
    for src in files:
        label = src.parent.name if src.name == 'graph.json' else src.parent.name + '__' + src.stem
        dest = args.output / label
        if dest in destinations:
            parser.error(f"Duplicate output folder {dest}; analyze these inputs separately")
        destinations.add(dest)
        source_bytes = src.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        data = json.loads(source_bytes.decode("utf-8-sig"))
        if args.decode_is_probe:
            data["execution"] = {**data.get("execution", {}), "decode_stage_semantics": "single_token_forward_probe", "use_cache": False}
        report = analyze_graph(data)
        report["source_graph"] = str(src.resolve())
        report["source_sha256"] = source_hash
        if args.validate_replay:
            from operator_graph.validation.replay import validate_replay
            validation = validate_replay(data, report)
            validation["checks"]["source_file_unchanged"] = hashlib.sha256(src.read_bytes()).hexdigest() == source_hash
            validation["passed"] = all(validation["checks"].values())
            report["offline_validation"] = validation
            validations.append({"model_name": report["model_name"], "source_graph": str(src.resolve()),
                                "sha256": source_hash, **validation})
        write_reports(report, dest)
        all_rows.extend(_flat_stages(report))
        for group in report["shape_signatures"]:
            key = (group["stage"], group["signature_id"])
            if key not in catalog:
                catalog[key] = {k: group[k] for k in ("stage", "signature_id", "domain", "raw_op_type",
                                                     "input_specs", "output_specs", "signature_complete")}
                catalog[key].update(num_nodes=0, model_names=set())
            catalog[key]["num_nodes"] += group["num_nodes"]
            catalog[key]["model_names"].add(report["model_name"])
        print(f"{report['model_name']}: {len(report['nodes'])} nodes -> {dest}")
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "model_metrics.csv", all_rows)
    catalog_rows = [{**group, "model_names": sorted(group["model_names"]), "num_models": len(group["model_names"])}
                    for _, group in sorted(catalog.items())]
    _write_csv(args.output / "shape_signature_catalog.csv", catalog_rows,
               ("stage", "signature_id", "signature_complete", "num_nodes", "model_names", "num_models"))
    def fmt(x):
        return "NA" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)
    summary = ["# Model workload summary", "",
        f"Estimation model: {COST_MODEL_VERSION}. AI is FLOP equivalents per logical byte, not measured DRAM intensity.",
        "Coverage includes convention-based, approximate and known-zero estimates, not graph extraction accuracy. Consult execution metadata for legacy decode semantics.", "",
        "| Model | Original stage label | Calls | Known work | Known logical bytes | Work-rule coverage | Complete AI | Mean level width | Path calls |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    keys = ["num_nodes", "known_flops", "known_logical_io_bytes", "flop_model_coverage", "arithmetic_intensity", "mean_level_width", "critical_path_nodes"]
    for row in all_rows:
        summary.append("| " + row["model_name"] + " | " + row["stage"] + " | " + " | ".join(fmt(row[k]) for k in keys) + " |")
    summary += ["", "## Workloads and dependencies", "",
                "Shape reuse is the number of calls with complete shape/dtype records divided by the number of complete configurations. It describes geometric configuration reuse.",
                "Top-10% shares select ceil(0.1 * call count) calls ranked by estimated work. Work/span is total estimated work divided by the maximum path work.", "",
                "| Model | Original stage label | Complete configurations | Shape reuse | Matrix calls/configurations | Top-10% work share | Work/span | Cast calls | Cast logical-byte share |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in all_rows:
        values = [row["num_shape_signatures"], row["shape_signature_reuse_factor"],
                  f"{row['matrix_call_count']}/{row['matrix_shape_signature_count']}", row["top10pct_node_flop_share"],
                  row["flop_work_over_span"], row["cast_count"], row["cast_logical_io_fraction"]]
        summary.append("| " + row["model_name"] + " | " + row["stage"] + " | " + " | ".join(fmt(v) for v in values) + " |")
    summary += ["", "## Cross-model configuration catalog", "",
                "Pool matching signatures separately by original stage label. Full prompts and single-token probes remain separate conditions.", "",
                "| Original stage label | Complete-config calls | Complete configurations | Matrix calls | Matrix configurations |", "|---|---:|---:|---:|---:|"]
    for stage in sorted({r["stage"] for r in all_rows}):
        groups = [g for g in catalog_rows if g["stage"] == stage and g["signature_complete"]]
        matrices = [g for g in groups if str(g["raw_op_type"]).upper() in {"MATMUL", "GEMM", "BMM", "LINEAR"}]
        summary.append(f"| {stage} | {sum(g['num_nodes'] for g in groups)} | {len(groups)} | {sum(g['num_nodes'] for g in matrices)} | {len(matrices)} |")
    summary += ["", "See shape_signature_catalog.csv. Descriptors exclude attributes, layouts and constant values and do not establish semantic or performance equivalence.",
                "See each model's metrics.json, stage_metrics.csv and detail CSVs for precision payloads, workload concentration and dependency statistics."]
    (args.output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    if validations:
        (args.output / "validation_summary.json").write_text(json.dumps(validations, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = ["# Offline graph validation", "",
                 "Checks use archived graph JSON without loading model weights: structural serialization, DAG integrity, estimate aggregation and layer-code replay.",
                 "These checks do not exercise model loading, PyTorch-to-ONNX export, source ONNX fidelity, numerical inference or hardware measurements.", "",
                 "| Model | Passed checks | Total checks | Result |", "|---|---:|---:|---|"]
        for v in validations:
            lines.append(f"| {v['model_name']} | {sum(v['checks'].values())} | {len(v['checks'])} | {'PASS' if v['passed'] else 'FAIL'} |")
        lines += ["", "Layer-code replay is not a layer-boundary accuracy measure. The source_file_unchanged check and SHA-256 records verify that the analysis preserved its inputs.",
                  "See validation_summary.json for individual checks and per-stage layer statistics, and summary.md / model_metrics.csv for numerical results."]
        (args.output / "validation_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Aggregate table: {args.output / 'model_metrics.csv'}")
    return 1 if any(not v["passed"] for v in validations) else 0


if __name__ == "__main__":
    raise SystemExit(main())
