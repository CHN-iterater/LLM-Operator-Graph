"""Explicit local archive regression against a read-only source workspace."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from operator_graph.analysis.engine import analyze_graph
from operator_graph.estimation.rules import estimate_node_cost
from operator_graph.validation.replay import validate_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    spec = importlib.util.spec_from_file_location('reference_metrics', root/'LLM-Graph-Parser/llm_graph_parser/metrics.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    baseline = json.loads((root/'analysis/新项目代码梳理_2026-11-23/baseline.json').read_text(encoding='utf-8'))
    sources, graphs = {}, []
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for entry in baseline['archived_graphs']:
        path = root/entry['path']
        assert sha(path) == entry['sha256'], f'Archive changed: {path}'
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        report = analyze_graph(data)
        assert report == reference.analyze_graph(data), f'Report changed: {path}'
        validation = validate_replay(data, report)
        assert validation['passed'], (path, validation['checks'])
        sources[entry['sha256']] = {n['op_id']: n for n in data['nodes']}
        graphs.append({'model': data.get('model_name'), 'sha256': entry['sha256'],
                       'nodes': len(data['nodes']), 'stages': len(report['stages']),
                       'report_matches_reference': True, 'replay_passed': True})
    with (root/'ACM_Journals_Primary_Article_Template/data/all_call_attributes.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        node = sources[row['source_sha256']][row['call_id']]
        cost = estimate_node_cost(node)
        for old, new in [('arithmetic_work_flop_equivalents','flops'), ('logical_bytes','logical_io_bytes')]:
            assert cost[new] is not None and math.isclose(float(row[old]), cost[new], rel_tol=1e-12, abs_tol=1e-12), (row['call_id'], old)
        assert cost['flops_status'] == row['arithmetic_status']
    for path, digest in {**baseline['paper_artifact_hashes'], **baseline['paper_csv_hashes']}.items():
        assert sha(root/path) == digest, f'Protected artifact changed: {path}'
    inventory = json.loads((root/'analysis/新项目代码梳理_2026-11-23/inventory.json').read_text(encoding='utf-8'))
    for entry in inventory['files']:
        assert sha(root/'LLM-Graph-Parser'/entry['path']) == entry['sha256']
    result = {'archive_graphs': graphs, 'forwards': sum(g['stages'] for g in graphs),
              'paper_call_records_checked': len(rows), 'static_cost_model': 'static-v2.0',
              'original_sources_and_paper_artifacts_unchanged': True,
              'reference_reports_identical': True, 'runtime_energy_measured': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
