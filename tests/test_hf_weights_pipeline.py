"""Actual local HF config/tokenizer/safetensors -> JSON and complete reports."""
import json
import pytest


def test_local_huggingface_weights_to_json_and_reports(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('onnxscript')
    pytest.importorskip('transformers')
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from operator_graph.exporters import export_prompt

    weights = tmp_path/'hf_weights'
    vocab = {'[UNK]':0, '[EOS]':1, '[PAD]':2, 'What':3, 's':4, 'the':5,
             'capital':6, 'of':7, 'France':8, '?':9, "'":10}
    raw = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]',
                                       eos_token='[EOS]', pad_token='[PAD]')
    model = GPT2LMHeadModel(GPT2Config(n_layer=1,n_head=1,n_embd=8,n_positions=32,
        vocab_size=len(vocab),bos_token_id=1,eos_token_id=1,pad_token_id=2,use_cache=True))
    model.save_pretrained(weights, safe_serialization=True)
    tokenizer.save_pretrained(weights)
    assert (weights/'model.safetensors').exists()
    output = tmp_path/'analysis'
    graphs = export_prompt(weights, "What's the capital of France?", output)
    assert set(graphs) == {'full_prompt','single_token_probe'}
    for condition, graph in graphs.items():
        assert graph.num_nodes > 0
        assert graph.execution['use_cache'] is False
        assert all(n.stage == condition for n in graph.nodes)
        assert (output/(condition+'_graph.json')).exists()
        assert (output/(condition+'.onnx')).exists()
    assert (output/'model_config.json').exists()
    reports = output/'reports'
    assert (reports/'model_metrics.csv').exists()
    assert (reports/'shape_signature_catalog.csv').exists()
    assert (reports/'summary.md').exists()
    checks = json.loads((reports/'validation_summary.json').read_text(encoding='utf-8'))
    assert len(checks) == 2 and all(x['passed'] for x in checks)
    assert len(list(reports.glob('*/node_metrics.csv'))) == 2
