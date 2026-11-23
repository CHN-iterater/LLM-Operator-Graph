"""Command line access to independent parse, analysis, validation and export steps."""
import argparse
import json
from pathlib import Path
import sys


def build_parser():
    parser = argparse.ArgumentParser(prog='operator-graph', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    parse = commands.add_parser('parse', help='Parse an existing ONNX forward')
    parse.add_argument('input', type=Path)
    parse.add_argument('--output', required=True, type=Path, help='Output graph JSON file')
    parse.add_argument('--model-name', default='')
    analyze = commands.add_parser('analyze', help='Analyze archived graph JSON without model weights')
    analyze.add_argument('input', type=Path)
    analyze.add_argument('--output', required=True, type=Path)
    analyze.add_argument('--decode-is-probe', action='store_true')
    analyze.add_argument('--validate-replay', action='store_true')
    validate = commands.add_parser('validate', help='Run graph replay and internal consistency checks')
    validate.add_argument('input', type=Path)
    validate.add_argument('--output', type=Path, help='Also write validation JSON to this file')
    export = commands.add_parser('export', help='Load HF weights and produce independent no-cache graphs plus full reports')
    export.add_argument('model')
    export.add_argument('--prompt', required=True)
    export.add_argument('--output', required=True, type=Path)
    export.add_argument('--allow-download', action='store_true', help='Allow downloading model/tokenizer files')
    export.add_argument('--trust-remote-code', action='store_true', help='Allow execution of model-provided Python code')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'parse':
            from . import parse_onnx
            from .serialization import graph_to_json
            graph = parse_onnx(args.input, model_name=args.model_name)
            graph_to_json(graph, output_path=args.output)
            print(args.output)
        elif args.command == 'analyze':
            from .reports.batch import main as batch_main
            arguments = [str(args.input), '--output', str(args.output)]
            if args.decode_is_probe:
                arguments.append('--decode-is-probe')
            if args.validate_replay:
                arguments.append('--validate-replay')
            return batch_main(arguments)
        elif args.command == 'validate':
            from .analysis.engine import analyze_graph
            from .validation.replay import validate_replay
            try:
                data = json.loads(args.input.read_text(encoding='utf-8-sig'))
                result = validate_replay(data, analyze_graph(data))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                result = {'passed': False, 'checks': {'readable_valid_graph': False}, 'error': str(exc)}
            payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(payload + '\n', encoding='utf-8')
            print(payload)
            return 0 if result['passed'] else 1
        elif args.command == 'export':
            from .exporters import export_prompt
            export_prompt(args.model, args.prompt, args.output,
                trust_remote_code=args.trust_remote_code, local_files_only=not args.allow_download)
            print(args.output)
        return 0
    except (ImportError, ValueError, OSError, RuntimeError) as exc:
        print(f'operator-graph: {exc}', file=sys.stderr)
        return 1
