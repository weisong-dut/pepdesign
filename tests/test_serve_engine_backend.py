"""引擎的后端分派与 vLLM 可用性检查（纯逻辑，不需要 GPU）。

覆盖配置分发、`health()` 的后端上报，以及缺 vllm / 缺 CUDA 时的
可操作错误信息。vLLM 的加载与生成需要 GPU，不在任何测试覆盖范围内。
peft 格式导出的往返测试见 `test_train_cpu_smoke.py`。
"""

from __future__ import annotations

import pytest

from pepdesign.serve.engine import InferenceEngine, _require_vllm
from pepdesign.utils.config import InferConfig


def _vllm_installed() -> bool:
    try:
        import vllm  # noqa: F401
    except ImportError:
        return False
    return True


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


# ---------------------------------------------------------------------------
# 后端分派
# ---------------------------------------------------------------------------
def test_engine_constructs_for_both_backends():
    """两个后端都要能构造出来；构造期抛错会让 `--dry-run` 无法报告环境缺什么。"""
    assert InferenceEngine(InferConfig()).cfg.backend == "transformers"
    assert InferenceEngine(InferConfig(backend="vllm")).cfg.backend == "vllm"


def test_health_reports_the_configured_backend():
    """`/health` 必须如实上报后端；写死会让客户端按错误后端选择超时与并发。"""
    assert InferenceEngine(InferConfig()).health()["backend"] == "transformers"
    assert InferenceEngine(InferConfig(backend="vllm")).health()["backend"] == "vllm"


def test_health_before_load_is_not_ready_but_does_not_raise():
    """加载前查健康状态要返回 `loading` 而不是抛异常，否则 k8s 探针会反复重启容器。"""
    h = InferenceEngine(InferConfig()).health()
    assert h["status"] == "loading"
    assert h["detail"]


def test_generate_before_load_raises_not_ready():
    """没加载就生成要报 ModelNotReadyError（映射成 503），客户端才会重试。"""
    from pepdesign.serve.engine import GenParams, ModelNotReadyError

    engine = InferenceEngine(InferConfig())
    with pytest.raises(ModelNotReadyError, match="尚未加载"):
        engine.generate_batch(["x"], GenParams(max_tokens=4, temperature=0.7, top_p=0.9))


# ---------------------------------------------------------------------------
# 可用性检查：错误信息必须是可操作的
# ---------------------------------------------------------------------------
@pytest.mark.skipif(_vllm_installed(), reason="本机装了 vllm，这条分支测不到")
def test_require_vllm_reports_install_command():
    """没装 vllm 时报错要给出该执行的安装命令，并提示可退回 transformers 后端。"""
    with pytest.raises(ImportError) as exc:
        _require_vllm()
    msg = str(exc.value)
    assert "serve-vllm" in msg, "报错里没有给出可执行的安装命令"
    assert "transformers" in msg, "报错里没有给出退路（改回默认后端）"


@pytest.mark.skipif(
    not _vllm_installed() or _cuda_available(),
    reason="只有'装了 vllm 但没有 CUDA'这一种情况才走这条分支",
)
def test_require_vllm_distinguishes_missing_cuda():
    """装了 vllm 但没有 GPU 时不能报成"请安装 vllm"，否则会让人反复重装。"""
    with pytest.raises(RuntimeError, match="CUDA"):
        _require_vllm()
