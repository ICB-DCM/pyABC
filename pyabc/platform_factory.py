import platform
from .sampler import MulticoreEvalParallelSampler, SingleCoreSampler


_linux = {"sampler": MulticoreEvalParallelSampler}
_macos = {"sampler": MulticoreEvalParallelSampler}
_windows = {"sampler": SingleCoreSampler}

if platform.system() == "Windows":
    _platform_factory = _windows
elif platform.system() == "Darwin":
    _platform_factory = _macos
else:
    _platform_factory = _linux

DefaultSampler = _platform_factory["sampler"]
