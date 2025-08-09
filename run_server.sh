#!/bin/bash
cd "$(dirname "$0")"
uvicorn ai_core.main:app --reload --host 0.0.0.0 --port 8001
