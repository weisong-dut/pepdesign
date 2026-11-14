#!/usr/bin/env bash
# 云端一键环境搭建 + 验证（AutoDL / 任意 Linux GPU 机器）
#
# 用法（在项目根目录）：
#     bash scripts/cloud_bootstrap.sh              # 装依赖 + 下权重 + 验证
#     bash scripts/cloud_bootstrap.sh --skip-model # 跳过下载权重（只装环境）
#
# 环境约束：
#   1. 装包前必须 unset 代理，否则 pip 无法解析镜像地址
#   2. torch 必须 >= 2.6，否则 transformers 5.x import 时报错
#   3. huggingface-hub 必须 < 2.0，2.x 的 httpx2 zstd 解压器存在问题
#   4. 必须卸载 torchvision，旧 torch 编译的 C 扩展会破坏 LlamaConfig 导入
# 前两条已写进 pyproject.toml；镜像预装的包仍可能干扰，故此处再校验一次。

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PROJ="$(pwd)"
SKIP_MODEL=0
[ "${1:-}" = "--skip-model" ] && SKIP_MODEL=1

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m   ✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m   ! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m   ✗ %s\033[0m\n' "$*"; exit 1; }

# ---------------------------------------------------------------- 0. 快照
step "0/5 机器状态"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version \
        --format=csv,noheader || warn "nvidia-smi 执行失败"
else
    die "找不到 nvidia-smi：这台机器没有 NVIDIA 驱动/显卡"
fi
df -h / "$HOME" 2>/dev/null | sed 's/^/   /'

# 数据盘：AutoDL 的 /root/autodl-tmp，其余平台退回到 $HOME
DATA_ROOT=""
for c in /root/autodl-tmp "$HOME/autodl-tmp" /workspace /data; do
    if [ -d "$c" ]; then DATA_ROOT="$c"; break; fi
done
[ -n "$DATA_ROOT" ] || DATA_ROOT="$HOME"
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf_cache}"
mkdir -p "$HF_HOME"
ok "HF_HOME = $HF_HOME"

# 环境变量不会跨终端保留，给后续终端留一个 source 入口
ENVFILE="$DATA_ROOT/pepdesign_env.sh"
printf 'export HF_HOME=%q\n' "$HF_HOME" > "$ENVFILE"
ok "已写 $ENVFILE（新终端先 source 它）"

# ---------------------------------------------------------------- 1. 装依赖
step "1/5 安装依赖（[train,dev]）"
# 代理会劫持 pip，先清掉所有大小写变体
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
python -m pip install -e '.[train,dev]' || {
    warn "默认源失败，改用清华源重试"
    python -m pip install -e '.[train,dev]' \
        -i https://pypi.tuna.tsinghua.edu.cn/simple
} || die "依赖安装失败"
ok "依赖安装完成"

# ---------------------------------------------------------------- 2. 版本硬约束
step "2/5 校验版本硬约束"
python - "$HF_HOME" <<'PY' || die "版本校验失败"
import sys, re
import torch, transformers, huggingface_hub

def ver(pkg):
    m = re.match(r"(\d+)\.(\d+)", pkg.__version__)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

problems = []
if ver(torch) < (2, 6):
    problems.append(f"torch {torch.__version__} < 2.6（transformers 5.x 需要 torch.accelerator）")
if ver(transformers) < (5, 0):
    problems.append(f"transformers {transformers.__version__} < 5.0")
hfv = ver(huggingface_hub)
if hfv < (1, 31) or hfv >= (2, 0):
    problems.append(f"huggingface-hub {huggingface_hub.__version__} 不在 [1.31, 2.0)")

print(f"   torch          {torch.__version__}")
print(f"   cuda available {torch.cuda.is_available()}")
print(f"   transformers   {transformers.__version__}")
print(f"   huggingface-hub {huggingface_hub.__version__}")
print(f"   HF_HOME        {sys.argv[1]}")

if not torch.cuda.is_available():
    problems.append("torch.cuda.is_available() == False：检查驱动，或在 AutoDL 控制台换带 GPU 的镜像")
for p in problems:
    print("   ✗ " + p)
sys.exit(1 if problems else 0)
PY
ok "版本约束全部满足"

# ---------------------------------------------------------------- 3. 卸载 torchvision
step "3/5 卸载 torchvision"
python -m pip uninstall -y torchvision 2>/dev/null || true
# 真正验证：transformers 在 LlamaConfig 的导入链上会 import torchvision.io
TRANSFORMERS_VERBOSITY=debug python - <<'PY' || die "LlamaConfig 导入失败（看上面的 debug 原始异常）"
from transformers import LlamaConfig
print("   LlamaConfig 导入 OK")
PY
ok "torchvision 已移除，导入链干净"

# ---------------------------------------------------------------- 4. 权重
step "4/5 基座权重 LOGOS-Hub/LOGOS-pretrain-3B"
CACHE_DIR="$HF_HOME/hub/models--LOGOS-Hub--LOGOS-pretrain-3B"
if [ "$SKIP_MODEL" = "1" ]; then
    warn "--skip-model：跳过下载"
elif [ -d "$CACHE_DIR" ] && [ "$(du -sm "$CACHE_DIR" | cut -f1)" -ge 6000 ]; then
    ok "本地已有缓存（$(du -sh "$CACHE_DIR" | cut -f1)），跳过下载"
else
    # 下载需要代理，与 pip 安装相反
    if [ -f /etc/network_turbo ]; then
        # shellcheck disable=SC1091
        source /etc/network_turbo
        ok "已开启 AutoDL 学术加速"
    else
        warn "没有 /etc/network_turbo，直接下载（国外机器可忽略）"
    fi
    python -c "from huggingface_hub import snapshot_download as s; s('LOGOS-Hub/LOGOS-pretrain-3B')" \
        || die "权重下载失败"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    du -sh "$HF_HOME"
    ok "权重下载完成"
fi

# ---------------------------------------------------------------- 5. 验证
step "5/5 逻辑测试 + SFT dry-run"
python -m pytest -q -m 'not smoke' || die "逻辑测试失败"
python scripts/train_sft.py --config configs/sft_lora.yaml --dry-run || die "SFT dry-run 失败"

cat <<EOF

$(printf '\033[1;32m')环境就绪$(printf '\033[0m')

新开终端先执行：
    source $ENVFILE

然后先跑最小验证（约 10 分钟）：
    python scripts/train_sft.py --config configs/sft_lora.yaml \\
        --set sft.max_steps=50 --set sft.save_steps=50
    python scripts/train_grpo.py --config configs/grpo_lora.yaml \\
        --set grpo.max_steps=20 \\
        --set grpo.sft_adapter_path=outputs/sft/checkpoint-50

长训练用 tmux，别让 SSH 断线把进程带走。
EOF
