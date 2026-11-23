# 本地迁移与论文对齐核查

日期：2026-11-23。新项目目录OperatorGraph，包名operator_graph。新Git仓库独立建立，未配置远端，未推送GitHub。

## 已迁移的完整工作流

1. **本地Hugging Face模型目录 → JSON与完整报告**：加载标准权重/config/tokenizer，分词，分别导出完整prompt和独立单token探针，保存ONNX、外部权重来源哈希、图JSON、原模型配置，再自动生成两种条件的逐调用/类型/层标签/配置/精度/拓扑报告及联合汇总、验证结果。
2. 已有ONNX → 调用级图JSON。
3. 已有新旧JSON → 静态估算及完整离线分析报告。
4. JSON → 完整性、读写、聚合、独立路径参照和可选层划分重放核查。

命令与输出目录见根README。export是权重到报告的端到端入口，默认不执行文本生成，也不下载模型。

## 结构迁移

| 原实现 | 新位置 |
|---|---|
| run.py及包入口中的导出编排 | exporters/torch.py、exporters/prompt.py、cli.py |
| core/operator_node.py与基础图操作 | model/records.py、model/graph.py |
| core/operator_registry.py | model/registry.py（历史category字符串仅为显示兼容标签，不表示实测瓶颈） |
| parser/onnx_parser.py | parsers/onnx.py及op_names.py |
| metrics.py计量规则 | estimation/rules.py |
| metrics.py汇总与结构/配置/精度分析 | analysis/aggregation.py、topology.py、workloads.py、precision.py、engine.py |
| metrics.py报告与批处理 | reports/writer.py、batch.py |
| core/serialization.py | serialization/__init__.py |
| offline_validation.py | validation/replay.py |
| LayerPartitioner | extensions/layers.py |

没有把旧仓库.git、旧utils估算、硬件采集、能耗拟合及历史改稿/绘图脚本复制为新核心。迁移代码保留原项目来源说明，尚未代选发布许可证。

## 有意修正的行为

- 未知估算用None/null，完整总量与已知部分总量分开。
- 旧数据已有static_cost时，它的显式状态优先于旧数值零占位。未携带状态的旧记录不凭数值擅自推断其语义；离线分析显式重算。
- 加权路径保留真实零权重，未知权重不返回伪完整路径。
- 环、悬空依赖、父子不对称及重复ID不再被有效图接口静默接受；原始字典分析仍可输出诊断。
- 标量、未知rank和零尺寸区分；论文已评估范围保持不变。
- 输入名称/可选槽位/重复操作数和去重依赖边同时保留。
- 自定义ONNX domain不继承同名标准算子的规则；不影响七份标准domain归档的估算。
- schema2拒绝丢弃原始类型、属性等必要来源的无metadata写出，避免显示别名改变复算结果。
- 未知总量的层核查、旧tensor可选默认字段归一处理正确。
- 两种export输出文件均可被目录批量分析识别，并使用独立报告目录。
- 临时ONNX清理后不保留失效来源路径；保留ONNX哈希及source_retained声明。模型training/cache状态在成功或失败后恢复，不全局修改环境、warning或pytree注册。

static-v2.0的算术及逻辑访问公式保持。上述修正集中于输入范围、未知状态、图合法性及接口组织，不改变论文归档结果。详细逐项对照见paper_alignment.md。

## 验证证据

- `verification.json`：最新测试、wheel构建、无可选依赖安装、环境版本及Git状态核对。
- `migration_verification.json`：七份归档、十四次前向的完整报告与原实现一致，论文26,074条调用的运算量、逻辑访问量及状态逐项一致。
- `test_hf_weights_pipeline.py`：真实创建并保存本地GPT-2小模型config、tokenizer、safetensors，再使用公开模型加载接口完成两次实际ONNX导出、JSON及全套报告。测试不下载模型。
- `test_export_contract.py`：真实小型PyTorch模块导出，以及失败/成功状态恢复、外部文件清理、默认本地加载、输入条件及报告自动生成。
- `test_paper_contract.py`：论文核心语义及迁移修复的回归用例。
- 无依赖虚拟环境安装wheel后，已有JSON的analyze和validate命令均通过，环境未安装torch、onnx或transformers。

此处不声称七个大模型均已重新从权重导出。七模型核查使用原归档JSON；权重加载端到端测试使用本地生成的小型HF模型。实际模型须能被其当前依赖版本加载并导出为ONNX；任意自定义模型兼容性不由上述小模型测试保证。

## 环境说明

本地验证环境使用Python 3.13.9、PyTorch 2.12.1、ONNX 1.22.0、ONNXScript 0.7.1、Transformers 5.13.0及pytest 8.4.2；准确版本以verification.json为准。模型导出依赖下限按现代torch导出API设为torch>=2.6。其他声明下限不代表本轮逐版本验证。

本机Anaconda的可选torch测试需要在测试进程设置KMP_DUPLICATE_LIB_OK=TRUE，与原运行环境一致；生产代码没有设置该变量。上游copyreg/Swig弃用提示及pyreadline3退出提示已保留于测试记录，不掩盖为零warning。无依赖核心安装与分析不需要此处理。

## 下一步发布

本地迁移完成后，可确定新的GitHub仓库名称、可见性和许可证，再配置新远端及推送。旧仓库远端未改动。原论文英文/中文/PDF/ZIP、CSV、七份归档和原源码保持哈希基线。
