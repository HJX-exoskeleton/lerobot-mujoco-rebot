#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "未找到 Python，请先激活 lerobot_rebot 环境或设置 PYTHON_BIN。" >&2
    exit 1
fi

cd "${SCRIPT_DIR}"

"${PYTHON_BIN}" workflow/home_servo.py
"${PYTHON_BIN}" workflow/home_rebot.py
