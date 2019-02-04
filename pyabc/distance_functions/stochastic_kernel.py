from .comparator import Comparator


RET_SCALE_LIN = "RET_SCALE_LIN"
RET_SCALE_LOG = "RET_SCALE_LOG"
RET_SCALE_LOG10 = "RET_SCALE_LOG10"
RET_SCALES = [RET_SCALE_LIN, RET_SCALE_LOG, RET_SCALE_LOG10]

class StochasticKernel(Comparator):

    def __init__(
            self,
            ret_scale=RET_SCALE_LIN):
        StochasticKernel.check_ret_scale(ret_scale)
        self.ret_scale = ret_scale

    @staticmethod
    def check_ret_scale(ret_scale):
        if ret_scale not in [RET_SCALE_LIN,
                             RET_SCALE_LOG,
                             RET_SCALE_LOG10]:
            raise ValueError(
                f"ret_scale must be in {RET_SCALES}")


class SimpleFUnctionKernel(StochasticKernel):

    def __init__(
            self,
            function,
            ret_scale=RET_SCALE_LIN):
        super().__init__(ret_scale)
        self.function = function

    def __call__(
            x: dict,
            x_0: dict,
            t: int,
            par: dict) -> float
        return self.function(x, x_0)
