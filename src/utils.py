"""
ユーティリティ関数
"""


def detect_device() -> str:
    """
    使用可能なデバイスを自動検出（GPU > MPS > CPU）

    Returns:
        使用するデバイス名 ("cuda", "mps", "cpu")
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def is_fp16_available() -> bool:
    """
    float16が使用可能か判定（CUDA or MPSが利用可能）

    Returns:
        float16が使用可能ならTrue
    """
    try:
        import torch
        return torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        return False
