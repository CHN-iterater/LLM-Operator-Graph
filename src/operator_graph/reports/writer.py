"""Write the versioned per-graph JSON, CSV and Markdown reports."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from operator_graph.estimation.rules import COST_MODEL_VERSION

def _write_csv(path, rows, empty_fields=()):
    rows = [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
             for k,v in row.items()} for row in rows]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        if rows or empty_fields:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else empty_fields)
            writer.writeheader()
            writer.writerows(rows)


def _flat_stages(report):
    return [{"model_name": report["model_name"], "stage": stage,
             **{k:v for k,v in stats.items() if k != "level_widths"}}
            for stage, stats in report["stages"].items()]


def write_reports(report, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _write_csv(out / "stage_metrics.csv", _flat_stages(report))
    _write_csv(out / "operator_metrics.csv", report["operators"])
    _write_csv(out / "layer_metrics.csv", report["layers"])
    _write_csv(out / "node_metrics.csv", report["nodes"])
    _write_csv(out / "shape_signature_metrics.csv", report["shape_signatures"],
               ("stage", "signature_id", "signature_complete", "num_nodes"))
    _write_csv(out / "cast_metrics.csv", report["cast_transitions"],
               ("stage", "input_dtypes", "output_dtypes", "num_nodes", "logical_io_bytes"))
    _write_csv(out / "dtype_metrics.csv", report["output_dtypes"],
               ("stage", "dtype", "num_output_tensors", "output_payload_bytes", "known_output_payload_bytes", "payload_coverage"))
    _write_csv(out / "level_widths.csv", [{"stage": s, "level": lv, "num_nodes": n}
        for s, stats in report["stages"].items() for lv,n in stats["level_widths"].items()])
    def fmt(x):
        return "NA" if x is None else f"{x:.4g}" if isinstance(x, float) else str(x)
    lines = [f"# Quantitative metrics: {report['model_name']}", "", f"Cost model: {COST_MODEL_VERSION}", "",
        "| Stage | Nodes | Edges | Known FLOPs | Logical bytes | FLOP coverage | Full AI | Paired modeled AI | Mean width | Path nodes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    keys = ["num_nodes", "num_edges", "known_flops", "known_logical_io_bytes", "flop_model_coverage",
            "arithmetic_intensity", "modeled_arithmetic_intensity", "mean_level_width", "critical_path_nodes"]
    for stage, stats in report["stages"].items():
        lines.append("| " + stage + " | " + " | ".join(fmt(stats[k]) for k in keys) + " |")
    lines += ["", "## Workload structure", "",
              "| Stage | Complete shapes | Shape coverage | Shape reuse | Matrix calls/shapes | Top-10% FLOP share | FLOP span | Work/span | Cast nodes | Cast I/O share |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for stage, stats in report["stages"].items():
        values = [stats["num_shape_signatures"], stats["shape_signature_coverage"], stats["shape_signature_reuse_factor"],
                  f"{stats['matrix_call_count']}/{stats['matrix_shape_signature_count']}", stats["top10pct_node_flop_share"],
                  stats["weighted_flop_span"], stats["flop_work_over_span"], stats["cast_count"], stats["cast_logical_io_fraction"]]
        lines.append("| " + stage + " | " + " | ".join(fmt(v) for v in values) + " |")
    lines += ["", "AI units: scalar FLOP equivalents / logical byte. NA means unavailable.", "",
        "Execution metadata: " + json.dumps(report["execution"], ensure_ascii=False), "",
        "Integrity checks: " + json.dumps(report["integrity"]), "", "## Definitions and assumptions", ""]
    lines += ["- " + s for s in report["assumptions"]]
    (out / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
