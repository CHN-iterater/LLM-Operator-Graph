"""Export safety contracts; ONNX export is isolated to avoid model downloads."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _exporter():
    assert importlib.util.find_spec('operator_graph.exporters') is not None
    return importlib.import_module('operator_graph.exporters.torch')


@pytest.mark.parametrize('fail', [False, True])
def test_export_restores_state_and_cleans_external_files(monkeypatch, fail):
    torch = pytest.importorskip('torch')
    module = _exporter()
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model.config = SimpleNamespace(use_cache=None)
    model.train()
    model[0].eval()
    original = [m.training for m in model.modules()]
    paths = []

    def fake_export(actual_model, args, path, **kwargs):
        paths.append(Path(path))
        assert not actual_model.training
        assert actual_model.config.use_cache is False
        assert kwargs['verbose'] is False
        assert kwargs['external_data'] is True
        Path(path).write_bytes(b'onnx')
        Path(path).with_name('weights.bin').write_bytes(b'external')
        if fail:
            raise RuntimeError('export failed')

    monkeypatch.setattr(torch.onnx, 'export', fake_export)
    monkeypatch.setattr(module, '_parse_export', lambda path, *args, **kw: SimpleNamespace(execution={'source_onnx':str(path)}))
    if fail:
        with pytest.raises(RuntimeError, match='export failed'):
            module.parse_model(model, torch.ones(1, 2))
    else:
        graph = module.parse_model(model, torch.ones(1, 2))
        assert graph.execution['use_cache'] is False
        assert graph.execution['retained_source_onnx'] is None
        assert graph.execution['source_onnx'] is None
        assert len(graph.execution['source_onnx_sha256']) == 64
    assert [m.training for m in model.modules()] == original
    assert model.config.use_cache is None
    assert paths and not paths[0].parent.exists()


def test_retained_export_has_truthful_provenance(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    module = _exporter()
    model = torch.nn.Linear(2, 2)
    target = tmp_path / 'retained.onnx'
    monkeypatch.setattr(torch.onnx, 'export', lambda model, args, path, **kwargs: Path(path).write_bytes(b'onnx'))
    monkeypatch.setattr(module, '_parse_export', lambda *args, **kw: SimpleNamespace(execution={}))
    monkeypatch.setattr(module, '_artifacts', lambda path: [{'path': str(path)}])
    graph = module.parse_model(model, torch.ones(1, 2), onnx_path=target, opset=19)
    assert target.exists()
    assert graph.execution['retained_source_onnx'] == str(target.resolve())
    assert graph.execution['export_opset'] == 19
    assert graph.execution['use_cache'] is None


def test_exporter_import_does_not_import_heavy_dependencies():
    import os
    import subprocess
    result = subprocess.run([sys.executable, '-c',
        "import sys; import operator_graph.exporters; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
        env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_actual_tiny_torch_export_retains_external_data_and_parses(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('onnxscript')
    from operator_graph import parse_model
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU())
    model.train()
    path = tmp_path / 'tiny.onnx'
    graph = parse_model(model, torch.ones(1, 2), onnx_path=path)
    assert model.training
    assert graph.num_nodes > 0
    assert graph.execution['retained_source_onnx'] == str(path.resolve())
    assert graph.execution['retained_artifacts']
    assert all(len(item['sha256']) == 64 for item in graph.execution['retained_artifacts'])
    assert all(n.stage == 'forward' for n in graph.nodes)


def test_prompt_exports_two_no_cache_conditions_with_safe_loading(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    import types
    module = importlib.import_module('operator_graph.exporters.prompt')
    from operator_graph.model.graph import ComputationGraph
    calls = []
    forwards = []

    class Model(torch.nn.Module):
        config = SimpleNamespace(use_cache=True)

        def forward(self, **kwargs):
            forwards.append(kwargs)
            return (kwargs['input_ids'],)

    model = Model()
    def load_model(source, **kwargs):
        calls.append(kwargs)
        return model

    def load_tokenizer(source, **kwargs):
        calls.append(kwargs)
        return lambda text, **kw: {'input_ids': torch.tensor([[1, 2, 3]]),
                                  'attention_mask': torch.ones(1, 3)}

    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer)))

    def parse(wrapper, tokens, model_name, onnx_path):
        wrapper(tokens)
        from operator_graph.model.records import OperatorNode
        graph = ComputationGraph(model_name)
        graph.add_node(OperatorNode(op_id='x', op_name='x', op_type='Mul'))
        return graph

    monkeypatch.setattr(module, 'parse_model', parse)
    graphs = module.export_prompt('local/model', 'text', tmp_path)
    assert set(graphs) == {'full_prompt', 'single_token_probe'}
    assert all(c == {'trust_remote_code': False, 'local_files_only': True} for c in calls)
    assert [list(f['input_ids'].shape) for f in forwards] == [[1, 3], [1, 1]]
    assert all(f['use_cache'] is False and f['return_dict'] is False for f in forwards)
    assert graphs['single_token_probe'].prompt_tokens == 1
    assert graphs['full_prompt'].execution['use_cache'] is False
    assert len(list(tmp_path.glob('*_graph.json'))) == 2
    assert (tmp_path/'reports/model_metrics.csv').exists()
    assert (tmp_path/'reports/summary.md').exists()
    assert (tmp_path/'reports/validation_summary.json').exists()
    assert len(list((tmp_path/'reports').glob('*/node_metrics.csv'))) == 2
