# Tokenforge 产品需求文档（PRD）

> **异构 GPU 推理吞吐与显存估算器**
>
> 输入卡型、模型、并发，锻造出每秒 Token 产能。Forge tokens from silicon.
>
> 版本：v1.0 | 日期：2026-06-03

---

## 1. 产品概述

### 1.1 产品定位

Tokenforge 是一款面向 AI 推理部署工程师的在线估算工具。用户只需输入模型名称（从 ModelScope 自动检索匹配）、GPU 型号与数量、上下文长度及并发要求，系统即可快速估算出该配置下的 **TPS（Tokens Per Second）** 和 **显存占用**，帮助用户在采购硬件或规划部署前做出合理决策。

### 1.2 核心价值

- **告别手工算显存**：自动从 ModelScope 拉取模型参数量、精度、权重大小等元数据
- **异构 GPU 支持**：支持不同型号 GPU 混装场景的估算
- **秒级反馈**：输入即得结果，无需下载任何模型
- **直观对比**：支持多配置方案对比，一目了然

### 1.3 目标用户

| 用户角色 | 核心场景 |
|---------|---------|
| 算法工程师 | 评估新模型部署成本，选型 GPU |
| 运维/SRE | 规划推理集群的硬件配置和并发容量 |
| 技术采购 | 估算不同 GPU 方案的性价比 |
| 创业者/CTO | 快速估算推理服务成本 |

---

## 2. 功能需求

详见根目录 README 与代码实现。核心模块：

1. 模型检索与选择（ModelScope 代理 + 自动补全 + 手动覆盖）
2. GPU 配置（预置卡库 `backend/data/gpus.yaml`，支持异构混装、快速预设）
3. 推理配置（上下文/并发/量化/框架/KV 精度）
4. 计算结果展示（核心指标卡、显存瀑布图、瓶颈分析）
5. 辅助功能（配置分享链接）

---

## 4. 计算模型

### 4.1 显存估算公式

- **模型权重**：`M_weights = P × B_precision × C_quant`
- **KV Cache**：`M_kv = 2 × L × kv_dim × N_ctx × B_kv × concurrency`（GQA 感知，`kv_dim = num_kv_heads × head_dim`）
- **激活值**：`M_act ≈ concurrency × N_ctx × H × B_precision × α`
- **总显存**：`M_total = M_weights + M_kv + M_act + M_overhead`

### 4.2 TPS 估算

- **Prefill（compute bound）**：`T = 2·P·N_in·batch / (FLOPS_total · η_compute)`
- **Decode（bandwidth bound）**：`TPOT = (weights + kv) / (BW_total · η_mem)`，`TPS_decode = batch / TPOT`

### 4.3 效率系数（η）

| 框架 | η_compute | η_mem |
|------|-----------|-------|
| TensorRT-LLM | 0.72 | 0.85 |
| vLLM | 0.60 | 0.80 |
| SGLang | 0.65 | 0.80 |
| llama.cpp | 0.45 | 0.60 |

> 实现见 `backend/app/services/estimator.py`，系数集中于该模块顶部常量，便于按实测校准。

---

## 9. 附录

### 9.1 ModelScope API 参考

- 搜索模型：`PUT https://modelscope.cn/api/v1/models`
- 模型 config：`GET https://modelscope.cn/api/v1/models/{model_id}/repo?FilePath=config.json`

### 9.2 术语表

| 缩写 | 全称 | 说明 |
|------|------|------|
| TPS | Tokens Per Second | 每秒生成 Token 数 |
| TTFT | Time To First Token | 首个 Token 生成延迟 |
| TPOT | Time Per Output Token | 每个输出 Token 的平均延迟 |
| TP | Tensor Parallelism | 张量并行 |
| DP | Data Parallelism | 数据并行 |

*Forge tokens from silicon.*
