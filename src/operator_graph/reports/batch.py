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
    summary = ["# 模型定量指标汇总", "",
        f"计量版本：{COST_MODEL_VERSION}。AI 是 FLOPs / 逻辑读写字节；不是实测 DRAM 计算强度。",
        "覆盖率包含精确、近似和已知零计数，不代表图解析准确率。decode 标签的含义见各报告 execution 字段。", "",
        "| 模型 | 原始阶段标签 | 节点 | 已建模 FLOPs | 逻辑字节 | FLOP 规则覆盖率 | 完整 AI | 平均层宽 | 最长路径节点数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    keys = ["num_nodes", "known_flops", "known_logical_io_bytes", "flop_model_coverage", "arithmetic_intensity", "mean_level_width", "critical_path_nodes"]
    for row in all_rows:
        summary.append("| " + row["model_name"] + " | " + row["stage"] + " | " + " | ".join(fmt(row[k]) for k in keys) + " |")
    summary += ["", "## 工作负载与依赖结构", "",
                "形状复用倍数 = 形状/dtype 完整的节点数 / 完整形状配置数；只表示几何配置复用。",
                "Top 10% 按 FLOPs 选取 ceil(0.1×节点数) 个节点；Work/span 为总 FLOPs / 最长依赖路径 FLOPs。", "",
                "| 模型 | 原始阶段标签 | 完整形状配置 | 形状复用倍数 | 矩阵调用/配置 | Top 10% FLOP 占比 | Work/span | Cast 节点 | Cast 逻辑字节占比 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in all_rows:
        values = [row["num_shape_signatures"], row["shape_signature_reuse_factor"],
                  f"{row['matrix_call_count']}/{row['matrix_shape_signature_count']}", row["top10pct_node_flop_share"],
                  row["flop_work_over_span"], row["cast_count"], row["cast_logical_io_fraction"]]
        summary.append("| " + row["model_name"] + " | " + row["stage"] + " | " + " | ".join(fmt(v) for v in values) + " |")
    summary += ["", "## 跨模型形状配置目录", "",
                "按原始阶段标签分别合并相同签名，不将完整 prompt 与单 token 探针混成一次执行。", "",
                "| 原始阶段标签 | 完整配置实例 | 完整配置数 | 矩阵实例 | 矩阵配置数 |", "|---|---:|---:|---:|---:|"]
    for stage in sorted({r["stage"] for r in all_rows}):
        groups = [g for g in catalog_rows if g["stage"] == stage and g["signature_complete"]]
        matrices = [g for g in groups if str(g["raw_op_type"]).upper() in {"MATMUL", "GEMM", "BMM", "LINEAR"}]
        summary.append(f"| {stage} | {sum(g['num_nodes'] for g in groups)} | {len(groups)} | {sum(g['num_nodes'] for g in matrices)} | {len(matrices)} |")
    summary += ["", "明细见 shape_signature_catalog.csv；配置不含属性、布局和常量取值，不能直接作为语义等价或性能等价证明。",
                "精度载荷、代价集中度与依赖统计见各模型 metrics.json、stage_metrics.csv 及三个新增明细 CSV。"]
    (args.output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    if validations:
        (args.output / "validation_summary.json").write_text(json.dumps(validations, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = ["# 图后端离线验证", "",
                 "输入为既有 graph.json，不加载权重。检查结构读写、DAG、计量聚合和层划分代码重放。",
                 "模型加载、PyTorch→ONNX 导出、原始 ONNX 忠实性、数值推理及硬件测量未在本次验证中覆盖。", "",
                 "| 模型 | 检查通过数 | 检查总数 | 结果 |", "|---|---:|---:|---|"]
        for v in validations:
            lines.append(f"| {v['model_name']} | {sum(v['checks'].values())} | {len(v['checks'])} | {'PASS' if v['passed'] else 'FAIL'} |")
        lines += ["", "层标签重放不等于边界准确率；source_file_unchanged 与 SHA-256 记录保证本次分析未改写输入文件。",
                  "各项检查及每阶段层统计见 validation_summary.json，定量结果见 summary.md / model_metrics.csv。"]
        (args.output / "validation_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Aggregate table: {args.output / 'model_metrics.csv'}")
    return 1 if any(not v["passed"] for v in validations) else 0


if __name__ == "__main__":
    raise SystemExit(main())
