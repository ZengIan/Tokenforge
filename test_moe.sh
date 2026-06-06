#!/bin/bash
cd /mnt/e/Openclaw/Tokenforge
pkill -f 'uvicorn backend.app.main:app' 2>/dev/null
sleep 1
nohup python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 > /tmp/tokenforge.log 2>&1 &
sleep 4
curl -s 'http://localhost:8000/api/models/config?model_id=Qwen/Qwen3.5-397B-A17B' | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('config_found:', d.get('config_found'))
print('config_error:', d.get('config_error'))
print('is_moe:', d.get('is_moe'))
print('active_params_b:', d.get('active_params_b'))
"
