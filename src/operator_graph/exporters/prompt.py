"""Two independent token-ID-only forwards; no generation or KV-cache decoding."""
from pathlib import Path

from .torch import parse_model


def export_prompt(model_source, prompt, output_dir, trust_remote_code=False,
                  local_files_only=True):
    """Load local HF weights and write graphs plus complete analysis reports.

    Returns a mapping from actual input condition to graph. Downloading weights
    and running model-provided code each require their own explicit opt-in.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    options = {'trust_remote_code': trust_remote_code, 'local_files_only': local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(str(model_source), **options)
    model = AutoModelForCausalLM.from_pretrained(str(model_source), **options)

    class TokenIdsForward(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.config = base.config

        def forward(self, input_ids):
            return self.base(input_ids=input_ids, use_cache=False, return_dict=False)

    ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
        raise ValueError('Prompt must tokenize to a nonempty batch-one token tensor')
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = ids.device
    ids = ids.to(device)
    wrapper = TokenIdsForward(model)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    label = Path(str(model_source).replace('\\', '/')).name
    graphs = {}
    for condition, tokens in [('full_prompt', ids), ('single_token_probe', ids[:, -1:])]:
        graph = parse_model(wrapper, tokens, model_name=label,
                            onnx_path=target / (condition + '.onnx'))
        graph.tag_unassigned_as(condition)
        graph.prompt_text = prompt
        graph.prompt_tokens = int(tokens.shape[1])
        graph.execution.update(graph_kind=condition, condition=condition, use_cache=False,
            cache_control='explicit_forward_keyword_and_model_config',
            batch_size=1, input_interface='input_ids_only', input_tokens=int(tokens.shape[1]),
            model_source=str(model_source), independent_forward=True,
            trust_remote_code=trust_remote_code, local_files_only=local_files_only)
        graph.save_to_json(target, name=condition)
        graphs[condition] = graph
    # Keep the original loading configuration separate from per-forward overrides.
    if hasattr(model.config, 'to_json_file'):
        model.config.to_json_file(str(target/'model_config.json'))
    from operator_graph.reports.batch import main as analyze_archives
    status = analyze_archives([str(target), '--output', str(target/'reports'), '--validate-replay'])
    if status:
        raise RuntimeError('Graphs exported, but analysis validation failed; inspect reports/validation_summary.json')
    return graphs
