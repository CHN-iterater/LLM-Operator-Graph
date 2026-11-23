# OperatorGraph

从本地 Hugging Face 模型权重或 ONNX 构建调用级算子图，保存张量配置、依赖及静态估算依据，生成 JSON、CSV 和 Markdown 分析报告。

## 安装与快速开始

需要 Python ≥3.10。以下命令均在项目根目录运行。

```shell
python -m pip install ".[export]"
operator-graph export "D:\Models\YourModel" --prompt "What's the capital of France?" --output output/your-model
```

模型目录应包含权重、`config.json` 和对应 tokenizer 文件。一次运行完成模型加载、分词、ONNX 导出、图 JSON 保存及完整报告生成。部分模型还需要各自的 tokenizer 依赖。

程序使用 token IDs，分别执行完整 prompt（`full_prompt`）和独立单 token 探针（`single_token_probe`）。两次前向均采用 eval 模式并关闭 KV cache；探针只接收原 prompt 的最后一个 token，不带历史 key/value，因此不能作为 cached decode。

默认加载本地文件。联网获取模型用 `--allow-download`，执行模型自定义代码用 `--trust-remote-code`。完整参数见 `operator-graph export --help`。

## 其他使用入口

按输入选择依赖：

| 用途 | 安装命令 |
|---|---|
| 已有 JSON 的离线分析与核查 | `python -m pip install .` |
| 已有 ONNX 的构图 | `python -m pip install ".[onnx]"` |
| 模型权重到图及完整报告 | `python -m pip install ".[export]"` |

```shell
operator-graph parse model.onnx --output output/graph.json
operator-graph analyze examples/graph.json --output output/reports
operator-graph validate examples/graph.json
```

`analyze` 也接受包含 `graph.json` 或 `*_graph.json` 的目录。离线 JSON 分析无需模型权重、GPU、ONNX 或 PyTorch。可用 `python -m operator_graph` 代替 `operator-graph`。

## 输出与计量说明

模型权重入口的主要输出：

```text
output/your-model/
  model_config.json
  full_prompt.onnx
  single_token_probe.onnx
  full_prompt_graph.json
  single_token_probe_graph.json
  reports/
    summary.md / model_metrics.csv / shape_signature_catalog.csv
    validation_summary.json / validation_summary.md
    <每种输入条件的独立目录>/
      metrics.json / metrics.md
      node_metrics.csv / operator_metrics.csv / layer_metrics.csv
      stage_metrics.csv / level_widths.csv
      shape_signature_metrics.csv / cast_metrics.csv / dtype_metrics.csv
```

较大模型还可能产生 ONNX 外部权重文件；图中会记录保留文件及哈希。调用记录保留原始类型、有序输入输出形状和 dtype，并区分重复操作数使用与去重后的调用依赖边。

运算量、逻辑数据访问量和 AI 使用 `static-v2.0` 声明规则，保留公式、假设与可用性状态。`null`/`NA` 表示估算不可用，完整总量与已知部分总量分别报告。这些数值属于静态估算，不是实测能耗或设备流量。

构图只处理 ONNX 顶层调用，不展开嵌套控制流。拓扑宽度描述逻辑依赖；层标签属于启发式分析。

## 验证与详细文档

当前核查结果：

- 72项测试通过，覆盖构图、估算、读写、CLI及本地 HF 权重到报告的流程。
- 七模型十四次前向的归档复算与论文一致，核对26,074条调用及论文关键图表统计。
- Qwen3-0.6B从本地权重重新导出的两张图，共4,840条调用的配置、依赖及估算与归档一致；其余六模型尚未从权重重新导出。

运行完整测试（含导出依赖）：

```shell
python -m pip install ".[test,export]"
python -m pytest
```

详细信息：

- [使用补充、项目结构与迁移核查](docs/migration_report.md)
- [论文与代码对照](docs/paper_alignment.md)
- [测试与安装验证记录](docs/verification.json)

代码迁移自 [LLM-Graph-Parser](https://github.com/CHN-iterater/LLM-Graph-Parser) 的研究工作树，服务于论文 *Operator Graph Construction for Large Language Model Inference*。
