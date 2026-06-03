#!/bin/bash

PROJ=/mnt/e/Openclaw/Tokenforge

echo "=== Tokenforge ==="

# 修复 PATH
export PATH="$HOME/.local/bin:$PATH"

# 释放端口
kill $(lsof -ti:8000) 2>/dev/null || true
kill $(lsof -ti:5173) 2>/dev/null || true
sleep 0.5

# 启动后端
echo "启动后端服务..."
cd $PROJ/backend
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# 启动前端
cd $PROJ/frontend
npx vite --host 0.0.0.0 --port 5173 &
FRONTEND_PID=$!

echo ""
echo "后端: http://localhost:8000/docs"
echo "前端: http://localhost:5173"
echo "Ctrl+C 停止"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
