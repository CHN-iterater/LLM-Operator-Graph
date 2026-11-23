import json
import os
from pathlib import Path
import subprocess
import sys


def run_cli(*args):
    return subprocess.run([sys.executable, '-m', 'operator_graph', *map(str, args)],
        env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')},
        text=True, capture_output=True)


def test_cli_help_and_export_defaults():
    result = run_cli('--help')
    assert result.returncode == 0, result.stderr
    assert all(command in result.stdout for command in ('parse', 'analyze', 'validate', 'export'))
    from operator_graph.cli import build_parser
    args = build_parser().parse_args(['export', 'local-model', '--prompt', 'hello', '--output', 'output'])
    assert not args.allow_download
    assert not args.trust_remote_code


def test_parse_command_creates_json_and_validate_reports_failure(tmp_path):
    import onnx
    from onnx import TensorProto, helper
    model = helper.make_model(helper.make_graph(
        [helper.make_node('Relu', ['x'], ['y'])], 'tiny',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2])]))
    source = tmp_path / 'tiny.onnx'
    onnx.save(model, source)
    output = tmp_path / 'nested' / 'graph.json'
    result = run_cli('parse', source, '--output', output)
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text(encoding='utf-8'))
    assert len(data['nodes']) == 1
    validation = run_cli('validate', output)
    assert validation.returncode == 0, validation.stderr
    assert json.loads(validation.stdout)['passed']
    data['nodes'][0]['parents'] = ['missing']
    output.write_text(json.dumps(data), encoding='utf-8')
    invalid = run_cli('validate', output)
    assert invalid.returncode != 0
    assert not json.loads(invalid.stdout)['passed']


def test_analyze_export_directory_finds_both_conditions(tmp_path):
    source = Path(__file__).resolve().parents[1]/'examples/graph.json'
    target = tmp_path/'exports'
    target.mkdir()
    for condition in ['full_prompt', 'single_token_probe']:
        data = json.loads(source.read_text(encoding='utf-8'))
        for node in data['nodes']:
            node['stage'] = condition
        (target/(condition+'_graph.json')).write_text(json.dumps(data), encoding='utf-8')
    output = tmp_path/'reports'
    result = run_cli('analyze', target, '--output', output)
    assert result.returncode == 0, result.stderr
    reports = list(output.glob('*/metrics.json'))
    assert len(reports) == 2
    stages = {next(iter(json.loads(p.read_text(encoding='utf-8'))['stages'])) for p in reports}
    assert stages == {'full_prompt', 'single_token_probe'}
