# PepDesign

多肽与环肽序列设计的大模型微调平台。基座为 LOGOS-3B，仓库内实现了 LoRA 注入与合并、
SFT 训练循环、校准集预热和 GRPO，奖励由可复算的规则引擎给出，并提供 OpenAI 兼容的推理服务。

## 包含内容

- 手写 LoRA（`models/lora_layer.py`），训练路径不依赖 peft；另有一组与 peft 的数值对照测试。
- 两阶段训练：SFT 学习输出格式与约束，校准集预热后进入 GRPO。
- 四维奖励：热稳定性、酶解稳定性、靶点亲和力、结构合理性。ESM-2 作为可选审计通道，默认权重为 0。
- 数据管线：下载、清洗、去重、同源切分、语料构造，输出 `dataset_card.json` 记录数据版本。
- 推理服务：FastAPI + OpenAI 兼容接口，支持 transformers 与 vLLM 两种后端，带批调度。
- 无 GPU 时可跑纯逻辑测试与 CPU 冒烟测试，CUDA 缺失时自动回退 fp32。

## 目录结构

```
configs/                 训练与服务配置
scripts/                 训练、评估、服务、诊断脚本
src/pepdesign/
  bio/                   氨基酸表、序列性质、酶切、环化（纯 CPU）
  data/                  多源下载、清洗、切分、语料构造
  models/                基座加载、LoRA 注入与合并
  rewards/               四维奖励与预测器
  train/                 SFT、校准预热、GRPO、掩码、checkpoint
  serve/                 FastAPI 服务、批调度、推理后端
  eval/                  离线评估与奖励审计
  prompts/               prompt 模板与输出解析
  utils/                 配置、日志、IO、指标记录
tests/                   单元测试与 CPU 冒烟测试
```

依赖方向向下：`bio → data → rewards → train → serve`。奖励层和数据层不 import torch，
可以脱离 GPU 单独运行。

## 安装

需要 Python 3.10–3.12。

```bash
pip install -e .                 # 核心依赖：数据、奖励、离线评估
pip install -e '.[train]'        # 训练
pip install -e '.[infer]'        # 推理服务（transformers 后端）
pip install -e '.[infer,serve-vllm]'   # vLLM 后端，仅 GPU 机器
pip install -e '.[dev]'          # 开发依赖
```

`huggingface-hub` 固定为 `<2.0`，`torch` 下限为 2.6，原因见 `pyproject.toml` 的依赖注释。

## 使用

### 数据

没有网络时，先用内置的公开多肽序列生成样例数据：

```bash
python scripts/00_make_sample_data.py --out data/raw/sample_peptides.jsonl
python -m pepdesign.data.pipeline
```

需要完整数据源时：

```bash
python -m pepdesign.data.pipeline --download
```

产出文件：

```
data/processed/sft_train.jsonl
data/processed/sft_valid.jsonl
data/processed/grpo_prompts.jsonl
data/calibration/calibration_pairs.jsonl
data/processed/dataset_card.json
```

### 监督微调

```bash
python scripts/train_sft.py --config configs/sft_lora.yaml

# 只校验配置与语料，不加载模型
python scripts/train_sft.py --config configs/sft_lora.yaml --dry-run
```

### 校准预热与 GRPO

```bash
python scripts/train_grpo.py --config configs/grpo_lora.yaml
```

校准预热由 `grpo.calibration_enabled` 控制，默认在 GRPO 前执行。

### 推理服务

```bash
python scripts/serve.py --config configs/serve.yaml
```

默认监听 `0.0.0.0:8000`，接口如下：

| 接口 | 说明 |
|---|---|
| `POST /v1/completions` | 文本补全 |
| `POST /v1/chat/completions` | 对话补全，内部拼接 prompt 后转发 |
| `GET /v1/models` | 当前模型 |
| `GET /health` | 健康检查 |
| `GET /admin/stats` | 批调度与请求统计 |
| `GET/POST /admin/adapters` | 查看与切换 LoRA adapter |

调试时可用 `--stub` 以假引擎验证 HTTP 层，或用 `--dry-run` 只校验配置。

### 离线评估

```bash
python scripts/eval_offline.py
```

报告输出到 `outputs/eval/reward_audit.json`。

## 测试

```bash
pytest                 # 全部测试，缺少 torch 时 smoke 文件自动跳过
pytest -m 'not smoke'  # 纯逻辑测试
pytest -m smoke        # CPU 冒烟
ruff check .           # 代码检查
```

CPU 冒烟使用随机初始化的小模型，验证梯度是否传到 LoRA 参数、掩码与 labels 是否对齐、
解析链路是否走通、损失是否为有限值。它不用于评估训练效果。

## 配置

默认值在 `src/pepdesign/utils/config.py`，配置文件只写与默认值不同的项。
任意字段可以在命令行覆盖：

```bash
python scripts/train_sft.py --config configs/sft_lora.yaml \
    --set sft.learning_rate=1e-4 --set lora.r=16
```

## 说明

- 仓库不包含模型权重和语料；真实训练需要自行准备 LOGOS-3B。
- 完整训练建议使用支持 bf16 的 CUDA 设备。QLoRA 的 4bit 量化在 CPU 上会自动关闭。
- GRPO 的参考模型是 SFT 后的策略，需要先运行阶段一。
