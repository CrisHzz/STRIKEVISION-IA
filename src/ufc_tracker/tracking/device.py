"""Choose inference hardware without loading detector weights."""

import warnings


def resolve_tracking_device(requested: str = "auto") -> str:
    import torch

    requested = str(requested).strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        if torch.cuda.is_available():
            return "0"
        warnings.warn(
            "Tracking uses CPU: CUDA is unavailable in this Python environment. "
            "A physical NVIDIA GPU alone is insufficient; check the PyTorch build.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    index = requested.removeprefix("cuda:")
    if not index.isdigit():
        raise ValueError("tracking_device must be auto, cpu, cuda:N or a GPU index")
    if not torch.cuda.is_available() or int(index) >= torch.cuda.device_count():
        raise ValueError(f"Requested CUDA device {requested!r} is unavailable")
    return str(int(index))
