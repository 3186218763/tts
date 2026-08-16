#!/usr/bin/env bash
# 本地启动/停止对话服务（TTS API + Web）
#
#   bash scripts/run_local.sh          # 启动（后台运行，日志在 logs/）
#   bash scripts/run_local.sh --stop   # 停止
#   bash scripts/run_local.sh --status # 查看状态
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOVITS_ROOT="$(realpath -m "${SOVITS_ROOT:-${PROJECT_ROOT}/../GPT-SoVITS}")"
CONFIG="$PROJECT_ROOT/configs/config.yaml"
LOGS="$PROJECT_ROOT/logs"
mkdir -p "$LOGS"

TTS_PORT="$(sed -nE 's/^  base_url: "http:\/\/127\.0\.0\.1:([0-9]+)"$/\1/p' "$CONFIG" | head -1)"
TTS_PORT="${TTS_PORT:-9880}"
WEB_PORT="${WEB_PORT:-8000}"
SOVITS_BIN="$SOVITS_ROOT/.venv/bin/python"
PROJECT_BIN="$PROJECT_ROOT/.venv/bin/python"

status() {
  printf 'TTS API (:%s): ' "$TTS_PORT"
  curl -sf -o /dev/null "http://127.0.0.1:$TTS_PORT/docs" && echo running || echo stopped
  printf 'Web    (:%s): ' "$WEB_PORT"
  curl -sf -o /dev/null "http://127.0.0.1:$WEB_PORT/healthz" && echo running || echo stopped
}

stop() {
  pkill -f "run_huayin_api.py.*--port $TTS_PORT" 2>/dev/null || true
  pkill -f "frontend.web --host 127.0.0.1 --port $WEB_PORT" 2>/dev/null || true
  sleep 1
  status
}

start() {
  [ -x "$SOVITS_BIN" ] || { echo "缺少 GPT-SoVITS venv：$SOVITS_BIN（先运行 scripts/setup_runtime_env.sh）"; exit 1; }
  [ -x "$PROJECT_BIN" ] || { echo "缺少项目 venv：$PROJECT_BIN（先运行 scripts/setup_runtime_env.sh）"; exit 1; }

  if ! curl -sf -o /dev/null "http://127.0.0.1:$TTS_PORT/docs"; then
    nohup "$SOVITS_BIN" "$PROJECT_ROOT/scripts/run_huayin_api.py" \
      --sovits-root "$SOVITS_ROOT" --python "$SOVITS_BIN" --port "$TTS_PORT" \
      >> "$LOGS/tts_api.log" 2>&1 &
    echo "TTS API 启动中（模型加载约 20~60s，日志: $LOGS/tts_api.log）"
    for _ in $(seq 1 90); do
      curl -sf -o /dev/null "http://127.0.0.1:$TTS_PORT/docs" && break
      sleep 2
    done
    curl -sf -o /dev/null "http://127.0.0.1:$TTS_PORT/docs" || { echo "TTS API 启动失败，见 $LOGS/tts_api.log"; exit 1; }
    echo "TTS API 就绪: http://127.0.0.1:$TTS_PORT"
  else
    echo "TTS API 已在运行: http://127.0.0.1:$TTS_PORT"
  fi

  if ! curl -sf -o /dev/null "http://127.0.0.1:$WEB_PORT/healthz"; then
    nohup "$PROJECT_BIN" -m frontend.web --host 127.0.0.1 --port "$WEB_PORT" \
      >> "$LOGS/web.log" 2>&1 &
    echo "Web 启动中（日志: $LOGS/web.log）"
    for _ in $(seq 1 30); do
      curl -sf -o /dev/null "http://127.0.0.1:$WEB_PORT/healthz" && break
      sleep 1
    done
    curl -sf -o /dev/null "http://127.0.0.1:$WEB_PORT/healthz" || { echo "Web 启动失败，见 $LOGS/web.log"; exit 1; }
    echo "Web 就绪: http://127.0.0.1:$WEB_PORT"
  else
    echo "Web 已在运行: http://127.0.0.1:$WEB_PORT"
  fi

  echo
  status
}

case "${1:-start}" in
  start)   start ;;
  --stop)  stop ;;
  --status|status) status ;;
  *) echo "用法: $0 [start|--stop|--status]"; exit 2 ;;
esac
