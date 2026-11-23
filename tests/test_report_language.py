"""The public report interface must use English with stable numerical fields."""
import json
import re
from pathlib import Path

from operator_graph.reports.batch import main


def test_generated_report_text_is_english(tmp_path):
    source = Path(__file__).resolve().parents[1]/'examples/graph.json'
    target = tmp_path/'reports'
    assert main([str(source), '--output', str(target), '--validate-replay']) == 0
    summary = (target/'summary.md').read_text(encoding='utf-8')
    validation = (target/'validation_summary.md').read_text(encoding='utf-8')
    assert '# Model workload summary' in summary
    assert '# Offline graph validation' in validation
    for file in target.rglob('*.md'):
        assert not re.search(r'[\u3400-\u9fff]', file.read_text(encoding='utf-8')), file.name
    metrics = json.loads(next(target.glob('*/metrics.json')).read_text(encoding='utf-8'))
    assert metrics['stages']['forward']['estimated_flops'] == 18
    assert metrics['stages']['forward']['logical_io_bytes'] == 216
