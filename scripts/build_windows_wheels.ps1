param (
    [string]$PYTHON_VERSION,
    [string]$TORCH_VERSION,
    [string]$CUDA_VERSION,
    [string]$MAX_JOBS = ""
)

# Conda 环境名称
$ENV_NAME = "build_env_$PYTHON_VERSION"

# 创建 Conda 环境
conda create -y -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

# 安装依赖
conda install -y ninja setuptools wheel pip
pip install --no-cache-dir torch==$TORCH_VERSION numpy --index-url "https://download.pytorch.org/whl/cu$($CUDA_VERSION.Substring(0,2))/"

# 设置环境变量
$env:NUNCHAKU_INSTALL_MODE="ALL"
$env:NUNCHAKU_BUILD_WHEELS="1"
$env:MAX_JOBS=$MAX_JOBS

# 进入当前脚本所在目录并构建 wheels
Set-Location -Path $PSScriptRoot
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }

python -m build --wheel --no-isolation

# 退出 Conda 环境
conda deactivate
Write-Output "Build complete!"
