# 🔥 Tokenforge

**异构 GPU 推理吞吐与显存估算器** — 输入模型、卡型、并发，秒级估算 **TPS** 与 **显存占用**，并给出瓶颈分析。
_Forge tokens from silicon._

显卡参数全部内置在 [`backend/data/gpus.yaml`](backend/data/gpus.yaml)，前端下拉即可选卡，自动识别显存 / 带宽 / 算力 / 互联。新增卡型只需编辑该 YAML。

---

## 快速开始

**Docker（推荐）**

```bash
docker compose up --build
# 前端 http://localhost:8080   后端 http://localhost:8000/docs
```

**本地开发**

```bash
# 后端
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 前端（另开终端）
cd frontend && npm install && npm run dev   # http://localhost:5173
```

---

## 技术栈

- **后端**：FastAPI — 估算引擎 + ModelScope 代理 + YAML 卡库
- **前端**：React + TypeScript + Vite + Tailwind + Zustand

```
backend/
  app/services/estimator.py    估算引擎（显存分解 + Prefill/Decode TPS + 瓶颈分析）
  app/services/modelscope.py   ModelScope 搜索/详情代理
  data/gpus.yaml               显卡卡库 ★
frontend/
  src/components/              模型 / GPU / 推理 / 结果 四个面板
```

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/gpus` | GET | 预置 GPU 卡库 |
| `/api/models/search?q=` | GET | 搜索 ModelScope 模型 |
| `/api/models/{model_id}` | GET | 拉取模型架构参数 |
| `/api/estimate` | POST | 返回 TPS + 显存估算 |

## 估算模型

| 项 | 公式 |
|----|------|
| 权重 | `P × bytes(quant) × overhead` |
| KV Cache | `2 × L × kv_dim × ctx × bytes(kv) × concurrency`（GQA 感知） |
| Decode TPOT | `(weights + kv) / (Σ带宽 × η_mem)` |
| Prefill | `2P × N_in × batch / (Σ算力 × η_compute)` |

系数集中在 [`estimator.py`](backend/app/services/estimator.py) 顶部，便于按实测校准。瓶颈分为 Compute / Memory / Bandwidth Bound 三类并给出建议。

> ⚠ YAML 中 `source: estimate` 的国产卡为占位估值，上线前请用厂商规格书或实测覆盖。估算为理论近似，目标实测误差 < 15%。
