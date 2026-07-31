#!/bin/bash
# GPT-SoVITS 环境安装脚本（使用官方 install.sh）
set -eE

CONDA_BASE="/home/mtr/miniconda3"
SOVITS_DIR="/home/mtr/tt/GPT-SoVITS"

source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "=== 1/3 创建 conda 环境 GPTSoVits ==="
conda create -n GPTSoVits python=3.10 -y

conda activate GPTSoVits

echo "=== 2/3 运行官方安装脚本（CU126 + ModelScope + UVR5）==="
cd "${SOVITS_DIR}"
bash install.sh --device CU126 --source ModelScope --download-uvr5

echo "=== 3/3 验证安装 ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== GPT-SoVITS 安装完成 ==="
echo "启动 API: conda activate GPTSoVits && cd ${SOVITS_DIR} && python api_v2.py --port 9880"
