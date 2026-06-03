# 🔥 Tokenforge

**异构 GPU 推理吞吐与显存估算器** — 输入卡型、模型、并发，锻造出每秒 Token 产能。
_Forge tokens from silicon._

输入模型（ModelScope 自动检索）、GPU 型号与数量、上下文长度与并发，Tokenforge 秒级估算出该配置下的 **TPS** 与 **显存占用**，并给出瓶颈分析与优化建议，帮助在采购硬件或规划部署前做出决策。

---

## 架构

```
frontend (React + TS + Vite + Tailwind + Zustand)
   │  HTTP/REST  (dev: vite proxy /api → :8000)
backend  (FastAPI)
   ├── ModelScope 代理 + 内存缓存      services/modelscope.py
   ├── 推理估算引擎（纯函数，可测）     services/estimator.py
   └── GPU 卡库 (YAML)                 data/gpus.yaml
```

显卡硬件参数全部内置于 [`backend/data/gpus.yaml`](backend/data/gpus.yaml)，前端通过 `/api/gpus`
拉取后即可下拉选择，自动识别显存 / 带宽 / 算力 / 互联。新增卡型只需编辑该 YAML，无需改代码。

> ⚠ YAML 中 `source: estimate` 的国产卡参数为占位估值，上线/投标前请用厂商 datasheet 或实测覆盖，并改 `source` 为 `datasheet` / `measured@日期`。

---

## 快速开始

### 方式一：Docker Compose（一键）

```bash
docker compose up --build
# 前端 http://localhost:8080   后端 http://localhost:8000/docs
```

### 方式二：本地开发

后端：

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

前端：

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 ，自动代理 /api → :8000
```

---

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/gpus` | GET | 预置 GPU 卡库 |
| `/api/models/search?q=` | GET | 搜索 ModelScope 模型（防抖 + 缓存） |
| `/api/models/{model_id}` | GET | 拉取模型 config.json 架构参数 |
| `/api/estimate` | POST | 提交配置，返回 TPS + 显存估算 |
| `/api/compare` | POST | 批量对比多个方案 |

交互式文档：`http://localhost:8000/docs`

---

## 计算模型（摘要）

| 项 | 公式 |
|----|------|
| 权重 | `P × bytes(quant) × overhead` |
| KV Cache | `2 × L × kv_dim × ctx × bytes(kv) × concurrency`（GQA 感知） |
| Decode TPOT | `(weights + kv) / (Σ带宽 × η_mem)` |
| Prefill | `2P × N_in × batch / (Σ算力 × η_compute)` |

全部系数集中在 [`backend/app/services/estimator.py`](backend/app/services/estimator.py) 顶部，便于按实测校准。
瓶颈判定为 Compute / Memory / Bandwidth Bound 三类并给出对应建议。

---

## 路线图

- [x] **Phase 1 MVP**：GPU 卡库、模型搜索、显存/TPS 估算、结果可视化、瓶颈分析、Docker 部署
- [x] 配置分享链接（URL 序列化）、手动参数覆盖、异构混装
- [ ] **Phase 2**：多方案并排对比 UI、移动端适配优化
- [ ] **Phase 3**：价格估算（$/1M tokens）、PNG/PDF 导出、i18n

---

_估算为理论近似，目标实测误差 < 15%。国产卡参数请以厂商规格书或实测为准。_
