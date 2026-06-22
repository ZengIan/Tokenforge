# 🔥 Tokenforge

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![React 18](https://img.shields.io/badge/react-18-61dafb.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-compose-ready-2496ed.svg)](./docker-compose.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

**异构 GPU 大模型推理估算器** — 输入模型、卡型、并行策略，秒级估算 **显存占用**、**吞吐 (TPS)**、**瓶颈** 与 **启动命令**，告别“拍脑袋”配卡。

_`Forge tokens from silicon.`_

> 支持 NVIDIA / AMD / 华为昇腾 / 平头哥 PPU / 海光 DCU / 寒武纪 / 昆仑芯 / 沐曦 等异构 GPU，以及 **vLLM / vllm-ascend / SGLang** 三大推理引擎。

---

## 🎯 为什么需要 Tokenforge？

部署 LLM 时，我们常遇到这些问题：

- 到底需要多少张卡？显存会不会爆？
- TP / PP / DP 怎么配？开不开 FP8 / AWQ / W4A16？
- SGLang 的 DP-Attention 和 vLLM 的 DP 规则不一样，怎么算？
- 昇腾 910C 和平头哥 PPU 该用 vllm-ascend 还是 SGLang？
- 写标书/做方案时，需要一份 Excel 算力说明。

Tokenforge 把这些估算做成了一个 **Web 工具**：下拉选模型、选卡、选引擎，立刻出结果，还能导出 Excel 和启动命令。

---

## ✨ 核心能力

| 能力 | 说明 |
|------|------|
| **异构 GPU 卡库** | 内置 NVIDIA H/H100/A100/RTX、华为 910C/910B、平头哥 PPU PG1_A910E、AMD MI300X、海光 DCU 等，卡库就是 YAML，随时扩展。 |
| **多引擎支持** | vLLM / vllm-ascend / SGLang，量化参数按引擎过滤，命令自动适配。 |
| **显存分解** | 权重、KV Cache、Activation、通信 Buffer 分项列出。 |
| **吞吐估算** | Prefill / Decode TPS、TTFT、TPOT，并判定 Compute / Memory / Bandwidth 瓶颈。 |
| **并行校验** | 自动校验 `TP × PP × DP = 卡数`（vLLM）或 `TP % DP == 0`（SGLang DP-Attention）。 |
| **ModelScope 集成** | 搜索模型、自动拉取架构参数，减少手工输入。 |
| **Excel 导出** | 导出完整估算报告 + 引擎对应的 `launch_server` 启动命令。 |
| **Docker 一键运行** | 单容器同时跑前端 + 后端，开箱即用。 |

---

## 🚀 快速开始

### Docker（推荐，一分钟启动）

```bash
git clone https://github.com/ZengIan/Tokenforge.git
cd Tokenforge
docker compose up --build
# 打开 http://localhost:8000
```

### 本地开发

```bash
# 后端
pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# API 文档 http://localhost:8000/docs

# 前端（另开终端）
cd frontend
npm install
npm run dev
# http://localhost:5173
```

### WSL 一键启动

```bash
bash ./start.sh
```

---

## 📸 界面预览

> 截图建议：运行时截取以下画面，替换下方链接。

| 面板 | 说明 |
|------|------|
| ![模型面板](docs/screenshot-model.png) | 选择/搜索 ModelScope 模型，自动填充参数量、层数、注意力维度。 |
| ![GPU 面板](docs/screenshot-gpu.png) | 下拉选卡、设数量、单机/多机、高速互联方式。 |
| ![推理面板](docs/screenshot-inference.png) | 引擎、量化、并发、上下文长度、TP/PP/DP。 |
| ![结果面板](docs/screenshot-result.png) | 显存饼图、吞吐、瓶颈、启动命令、Excel 导出。 |

---

## 🏗️ 架构

```
Tokenforge/
├── backend/
│   ├── app/services/estimator.py    # 估算引擎（显存 + 吞吐 + 瓶颈）
│   ├── app/services/modelscope.py   # ModelScope 搜索/详情代理
│   ├── app/services/engine.py         # 多引擎（vLLM / vllm-ascend / SGLang）命令生成
│   └── data/gpus.yaml                 # GPU 卡库 ★ 新增卡型只需改这里
├── frontend/
│   ├── src/components/                # 模型 / GPU / 推理 / 结果 面板
│   ├── src/lib/exportExcel.ts         # Excel 报告导出
│   └── src/store.ts                   # Zustand 全局状态
└── docker-compose.yml                 # 单容器启动
```

### 技术栈

- **后端**：FastAPI + Pydantic + YAML 卡库
- **前端**：React 18 + TypeScript + Vite + Tailwind CSS + Zustand
- **估算模型**：理论带宽/算力模型，按 GQA、量化、并行通信损耗修正

---

## 📊 估算模型速览

| 项 | 公式要点 |
|----|----------|
| 权重显存 | `P × bytes(quant) × overhead` |
| KV Cache | `2 × L × kv_dim × ctx × bytes(kv) × concurrency`（GQA 感知） |
| Decode TPOT | `(weights + kv) / (Σ 显存带宽 × η_bw)` |
| Prefill TTFT | `2P × N_in × batch / (Σ 算力 × η_compute)` |

瓶颈自动判定为 **Compute Bound / Memory Bound / Bandwidth Bound**，并给出调参建议。

> ⚠️ YAML 中 `source: estimate` 的国产卡参数为公开估值，生产/投标前请务必用厂商规格书或实测覆盖。工具会在结果区明确提示可信度。

---

## 🛣️ Roadmap

- [x] 多引擎支持（vLLM / vllm-ascend / SGLang）
- [x] 异构 GPU 卡库（国产卡 + NVIDIA 全系列）
- [x] Excel 导出 + 启动命令生成
- [x] 跨机通信损耗建模
- [ ] 实测校准模式（上传 benchmark 自动修正系数）
- [ ] 多模型并发估算
- [ ] 部署方案推荐（最低成本 / 最高吞吐 / 最低延迟）
- [ ] 在线 Demo（Vercel / Hugging Face Spaces）

---

## 🤝 参与贡献

欢迎 PR 和 Issue！

- 发现卡库参数有误？直接改 `backend/data/gpus.yaml` 并提交 PR。
- 想支持新的推理引擎？参考 `backend/app/services/engine.py` 添加分支。
- 有功能建议？开 Issue 描述场景。

---

## 📜 License

MIT © [ZengIan](https://github.com/ZengIan)

---

> 如果 Tokenforge 对你有用，请给个 ⭐ Star，这将帮助我们持续维护国产卡数据与引擎支持！
