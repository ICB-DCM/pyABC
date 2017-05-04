import platform
from .sampler import MulticoreEvalParallelSampler, SingleCoreSampler


_linux = {"sampler": MulticoreEvalParallelSampler}
_windows = {"sampler": SingleCoreSampler}
_platform_factory = _windows if platform.system() == "Windows" else _linux

DefaultSampler = _platform_factory["sampler"]
