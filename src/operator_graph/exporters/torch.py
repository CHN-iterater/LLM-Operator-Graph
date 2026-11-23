"""Lazy PyTorch to ONNX export with scoped model state and file ownership."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory


def _parse_export(path, model_name, registry):
    from operator_graph import parse_onnx
    return parse_onnx(path, model_name=model_name, registry=registry)


def software_versions():
    result = {}
    for package in ('torch', 'onnx', 'transformers'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _artifacts(path):
    """Only inspect referenced external files; never scan unrelated user files."""
    import onnx
    artifacts = [path]
    model = onnx.load(str(path), load_external_data=False)
    from onnx.external_data_helper import _get_all_tensors
    for tensor in _get_all_tensors(model):
        for entry in tensor.external_data:
            if entry.key == 'location':
                artifact = (path.parent / entry.value).resolve()
                if artifact not in artifacts:
                    artifacts.append(artifact)
    result = []
    for artifact in artifacts:
        result.append({'path': str(artifact), 'sha256': _file_hash(artifact)})
    return result


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_model(model, *example_args, model_name='model', registry=None,
                onnx_path='', opset=18):
    """Export an evaluation forward and parse it, restoring model state on failure.

    Cache is disabled when exposed by ``model.config.use_cache``. Models without
    that setting have unknown cache semantics. A temporary directory owns all
    transient external weights, rather than guessing their filenames.
    Requires the modern torch ONNX exporter (torch >= 2.6).
    """
    import torch
    retained = bool(onnx_path)
    context = nullcontext(None) if retained else TemporaryDirectory(prefix='operator_graph_')
    states = [(module, module.training) for module in model.modules()]
    config = getattr(model, 'config', None)
    has_cache = config is not None and hasattr(config, 'use_cache')
    original_cache = config.use_cache if has_cache else None
    with context as temporary:
        path = Path(onnx_path).resolve() if retained else Path(temporary) / 'model.onnx'
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            model.eval()
            if has_cache:
                config.use_cache = False
            with torch.no_grad():
                torch.onnx.export(model, example_args, str(path), opset_version=opset,
                                  external_data=True, verbose=False, dynamo=True)
            graph = _parse_export(path, model_name, registry)
            # ONNX is parsed before recording provenance, so malformed exports fail.
            artifacts = _artifacts(path) if retained else []
            graph.execution = {**graph.execution, 'graph_kind': 'onnx_forward',
                'source_onnx': str(path) if retained else None,
                'source_onnx_sha256': _file_hash(path), 'source_retained': retained,
                'export_opset': opset, 'use_cache': False if has_cache else None,
                'cache_control': 'model.config.use_cache=False' if has_cache else 'not_available',
                'retained_source_onnx': str(path) if retained else None,
                'retained_artifacts': artifacts, 'software_versions': software_versions()}
            return graph
        finally:
            if has_cache:
                config.use_cache = original_cache
            for module, training in states:
                module.training = training
