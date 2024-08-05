import inspect
import os
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from scipy.interpolate import interp1d

# internlm/model/registry.py
# from internlm.model.registry import benchmark_initializer
from internlm.model.registry import benchmark_initializer
from internlm.simulator.common import (
    GLOBAL_BYTE_SIZES_LIST,
    OUT_OF_MEM_LATENCY,
    sync_all,
    sync_local,
)


class Timer:
    def __init__(self, use_event) -> None:
        self.use_event = use_event
        if use_event:
            self.start_t = torch.cuda.Event(enable_timing=True)
            self.end_t = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.use_event:
            self.start_t.record()
        else:
            self.start_t = time.time()

    def end(self, group=None):
        if self.use_event:
            self.end_t.record()
            if group != None:
                dist.barrier(group)
            torch.cuda.synchronize()
            return self.start_t.elapsed_time(self.end_t) / 1000
        else:
            torch.cuda.synchronize()
            return time.time() - self.start_t


def DFS(loop_config: OrderedDict, results: OrderedDict, total_results: List):
    if len(loop_config) == 0:
        total_results.append(deepcopy(results))
        return

    now_key = list(loop_config.keys())[0]
    now_values = loop_config[now_key]
    loop_config.pop(now_key)

    for value in now_values:
        results.update({now_key: value})
        DFS(loop_config, results, total_results)

    loop_config[now_key] = now_values


def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def run_cal_profile(test_type, warmups=2, trials=5):
    BENCH = benchmark_initializer.get_module(module_name=test_type)

    def run_benchmark(test_case):
        # Warmups, establish connections, etc.
        timer = Timer(use_event=True)
        for _ in range(warmups):
            try:
                test_case.run()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return OUT_OF_MEM_LATENCY

        try:
            sync_local()
        except RuntimeError:
            print(
                f"packed_length: {test_case.packed_length}, embed_dim: {test_case.embed_dim}, micro_bsz: {test_case.micro_bsz}, seq_len: {test_case.seq_len}, tp:{test_case.tp_size}",
                flush=True,
            )
            torch.cuda.empty_cache()
            return OUT_OF_MEM_LATENCY

        # time the actual comm op trials times and average it
        duration = 0
        for _ in range(trials):
            timer.start()
            try:
                test_case.run()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return OUT_OF_MEM_LATENCY

            duration += timer.end()

        # maintain and clean performance data
        avg_duration = duration / trials
        return avg_duration

    sync_local()
    # loop over various tensor sizes
    test_args = OrderedDict(BENCH.test_loop)
    total_cases = []

    DFS(test_args, OrderedDict(), total_cases)

    tflop = []
    tflops_list = []
    for _, test_case in enumerate(total_cases):

        try:
            bench = BENCH(**filter_kwargs(BENCH.__init__, test_case))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        except AssertionError:
            torch.cuda.empty_cache()
            break
        else:
            sync_local()
            complexity = bench.complexity()
            if complexity in tflop:
                continue

            avg_lat = run_benchmark(bench)
            tflops = complexity / avg_lat

            tflop.append(complexity)
            tflops_list.append(tflops)

            print(
                f"complexity: {complexity/ 10**12:.3f}, tflops:{tflops/ 10**12:.3f}, avg_duration: {avg_lat*1000:.3f} ms",
                flush=True,
            )

    return tflop, tflops_list


def run_comm_profile(test_type, group, plot_name, warmups=5, trials=20):

    BENCH = benchmark_initializer.get_module(module_name=test_type)

    def run_benchmark(test_case, group):
        # Warmups, establish connections, etc.
        timer = Timer(use_event=True)
        for _ in range(warmups):
            test_case.run()

        sync_all(group)

        # time the actual comm op trials times and average it
        duration = 0
        for _ in range(trials):
            timer.start()
            test_case.run()
            duration += timer.end(group)

        # maintain and clean performance data
        avg_duration = duration / trials
        return avg_duration

    # loop over various tensor sizes
    test_args = OrderedDict(BENCH.test_loop)
    total_cases = []

    DFS(test_args, OrderedDict(), total_cases)

    comm_vols, bws = [], []
    for test_case in total_cases:
        test_case["group"] = group
        bench = BENCH(**filter_kwargs(BENCH.__init__, test_case))

        avg_duration = run_benchmark(bench, group)

        comm_vol = bench.bw_complexity()
        bw = comm_vol / avg_duration

        comm_vols.append(comm_vol)
        bws.append(bw)
        if dist.get_rank() == 0:
            print(
                f"    plot_name: {plot_name}, Buff: {test_case['global_size']/1024**3:.3f} GB, avg bw: {bw/ 1024**3:.3f} GB/s",
                flush=True,
            )

    return comm_vols, bws


def draw_pics(base_path, plot_name, comm_vols, bws):
    x, y = [], []

    spline_model = interp1d(comm_vols, bws, kind="slinear")

    end = GLOBAL_BYTE_SIZES_LIST[-1] // 1024**2
    for i in range(1, end + 1):
        vol = i * 1024**2
        try:
            predice_bw = spline_model(vol)
        except ValueError:
            if vol < GLOBAL_BYTE_SIZES_LIST[0]:
                predice_bw = spline_model(GLOBAL_BYTE_SIZES_LIST[0])
            else:
                predice_bw = spline_model(GLOBAL_BYTE_SIZES_LIST[-1])

        x.append(vol / 1024**2)
        y.append(predice_bw / 1024**3)

    bws = list(map(lambda x: x / 1024**3, bws))
    comm_vols = list(map(lambda x: x / 1024**2, comm_vols))

    pic_path = os.path.join(base_path, plot_name + ".jpg")

    plt.figure(figsize=(12, 6))
    plt.scatter(comm_vols, bws, label="True value")
    plt.plot(x, y, label="Fit value")
    plt.xlabel("Data Transferred (MB)")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title(f"Bandwidth Spline Fit for {plot_name} at different data volume")
    plt.legend()
    plt.grid(True)
    plt.savefig(pic_path)
    plt.show()

    return spline_model


def draw_cal_pics(base_path, plot_name, tflop, tflops):
    # x, y = [], []

    spline_model = interp1d(tflop, tflops, kind="slinear")

    # start = tflop[0]
    # end = tflop[-1]
    # for complexity in range(start, end+1):
    #     try:
    #         predice_tflops = spline_model(complexity)
    #     except ValueError:
    #         if complexity < tflop[0]:
    #             predice_tflops = spline_model(tflop[0])
    #         elif complexity > tflop[-1]:
    #             predice_tflops = spline_model(tflop[-1])

    #     x.append(complexity)
    #     y.append(predice_tflops)

    pic_path = os.path.join(base_path, plot_name + ".jpg")
    tflop = list(map(lambda x: x / 10**12, tflop))
    tflops = list(map(lambda x: x / 10**12, tflops))

    plt.figure(figsize=(12, 6))
    plt.scatter(tflop, tflops, label=f"True value")
    # plt.plot(x, y, label=f"Fit value")
    plt.xlabel("tflop")
    plt.ylabel("Tflops")
    plt.title(f"Tflops Spline Fit for {plot_name} at different tflop")
    plt.legend()
    plt.grid(True)
    plt.savefig(pic_path)
    plt.show()

    return spline_model
