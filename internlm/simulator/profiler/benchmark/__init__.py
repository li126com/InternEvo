from internlm.model.registry import Registry, benchmark_initializer
from internlm.simulator.profiler.benchmark import (
    all2all,
    all_gather,
    all_reduce,
    broadcast,
    linear,
    reduce_scatter,
)

# from .all_gather import *
# from .all_reduce import *
# from .broadcast import *
# from .linear import *
# from .multi_head_attn import *
# from .reduce_scatter import *


def register_comm_pref_initializer() -> None:
    benchmark_initializer.register_module(all2all.BENCH_TYPE, all2all.UnitBenchAll2ALL)
    benchmark_initializer.register_module(all_gather.BENCH_TYPE, all_gather.UnitBenchAllGather)
    benchmark_initializer.register_module(all_reduce.BENCH_TYPE, all_reduce.UnitBenchAllReduce)
    benchmark_initializer.register_module(broadcast.BENCH_TYPE, broadcast.UnitBenchBroadcast)
    benchmark_initializer.register_module(reduce_scatter.BENCH_TYPE, reduce_scatter.UnitBenchAllReduceScatter)
    benchmark_initializer.register_module(linear.BENCH_TYPE, linear.UnitBenchLinear)

    # model_initializer.register_module("LLAVA", Llava)
