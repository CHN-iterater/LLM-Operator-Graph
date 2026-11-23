# OperatorGraph

调用级算子图构建与可追溯静态工作负载分析。独立迁移自 [LLM-Graph-Parser](https://github.com/CHN-iterater/LLM-Graph-Parser) 的本地研究工作树，与论文 *Operator Graph Construction for Large Language Model Inference* 的当前方法及计量口径对齐。

## 能力

- 从ONNX顶层调用收集原始类型、数值属性、有序输入输出名称、形状及dtype，通过生产者索引连接依赖。
- 保留重复操作数使用，同时对同一调用对的依赖边去重。
- 派生static-v2.0运算量、逻辑访问量和AI，保留公式、假设、状态；未知值为null，完整总量与已知部分总量分开。
- 分析拓扑层级、分叉汇合、配置分组与精度，输出CSV/JSON/Markdown。
- 读取旧schema 1.0图；新图使用schema 2.0，未知rank与标量形状明确区分。
- 可选PyTorch前向导出；可选启发式层划分。核心JSON分析不依赖ONNX、PyTorch或Transformers。

构图范围是导出ONNX的顶层调用，不展开嵌套控制流。估算不是实测能耗、DRAM流量或硬件指令数。逻辑依赖宽度不是设备并发度。

## 安装

Python >=3.10。离线分析仅安装本包：

```text
python -m pip install .
```

ONNX解析及模型导出分别选择可选依赖：

```text
python -m pip install ".[onnx]"
python -m pip install ".[export]"
python -m pip install ".[test]"
```

可选模型可能还需要各自tokenizer依赖。测试环境和已完成核查见[迁移核查](docs/migration_report.md)。

## 快速开始

无模型权重的三调用分叉汇合示例：

```text
python examples/make_branch_graph.py output/branch.onnx
operator-graph parse output/branch.onnx --output output/graph.json
operator-graph analyze output/graph.json --output output/reports
operator-graph validate output/graph.json
```

也可以用 `python -m operator_graph` 代替安装后的命令。已有JSON直接分析，无需GPU或模型加载。`examples/graph.json` 提供可直接分析的小型图。

Python接口：

```python
from operator_graph import parse_onnx, graph_to_json
from operator_graph.analysis.engine import analyze_graph
from operator_graph.reports.writer import write_reports

graph = parse_onnx("model.onnx", model_name="example")
graph_to_json(graph, output_path="output/graph.json")
report = analyze_graph(graph.to_dict())
write_reports(report, "output/reports")
```

## 从Hugging Face权重到JSON及完整报告

输入为下载到本地的标准Hugging Face模型目录，包含模型权重、`config.json`及对应tokenizer文件。一次命令完成加载、分词、两次前向构图、JSON保存和完整分析报告：

```text
operator-graph export "D:\Models\YourModel" --prompt "What's the capital of France?" --output output/your-model
```

输出包括：

```text
output/your-model/
  model_config.json
  full_prompt.onnx
  single_token_probe.onnx
  full_prompt_graph.json
  single_token_probe_graph.json
  reports/
    summary.md
    model_metrics.csv
    shape_signature_catalog.csv
    validation_summary.json
    validation_summary.md
    <每种输入条件独立的报告目录>/
      metrics.json / metrics.md
      node_metrics.csv / operator_metrics.csv / layer_metrics.csv
      stage_metrics.csv / level_widths.csv
      shape_signature_metrics.csv / cast_metrics.csv / dtype_metrics.csv
```

较大模型导出时还可能产生ONNX外部权重文件，保留文件及其哈希会写入图的执行来源信息。报告中的NA/null表示相应估算不可用，不会伪装成零。

参数通过 `operator-graph export --help` 查看。模型输入适配使用token IDs；默认仅加载本地文件。需要联网获取模型时显式加 `--allow-download`，需要自定义模型代码时显式加 `--trust-remote-code`。

## 输入条件与旧数据

新模型适配输出 `full_prompt` 和 `single_token_probe` 两个独立前向，均在eval模式下关闭KV cache。探针只接收同一prompt的最后一个token，没有历史key/value。它们不等同于Prefill和cached Decode。

旧JSON的prefill/decode标签保持原样读取。只有明确掌握数据来源时，才在分析命令中使用 `--decode-is-probe`。该开关声明输入语义，不会补造cache或执行历史。

## 项目结构

```text
src/operator_graph/
  model/          # 图、调用、张量、显示注册表
  parsers/        # ONNX元数据及两轮调用级构图
  exporters/      # 可选PyTorch和模型输入适配
  estimation/     # 唯一有效的静态计量规则
  analysis/       # 汇总、拓扑、配置和精度统计
  serialization/  # 旧图读取及版本化JSON输出
  reports/        # 报告与批量离线分析
  validation/     # 图完整性与重放核查
  extensions/     # 可选层划分
```

旧calculator、FX hooks、自动阶段推断、硬件瓶颈判断和能耗测量脚本不进入默认核心。具体去向和论文对齐差异见[论文与代码对照](docs/paper_alignment.md)。

## 测试与本地归档回归

```text
python -m pytest
python tools/verify_migration.py <原论文工作区> --output output/migration.json
```

归档回归需要原工作区中的七份JSON及论文CSV，但运行本包不依赖原仓库。回归工具显式读取旧实现作为参照，只用于本地迁移核查。

本项目尚未配置GitHub远端。发布前需确定新仓库名称及许可证；本次迁移保留源项目归属，不默认附加未经确认的开源许可证。
