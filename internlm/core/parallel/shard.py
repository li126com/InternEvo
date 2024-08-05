"""
shard strategies for parallel
"""

from typing import Callable, List

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger

logger = get_logger(__file__)


# The head layer in ISP mode is actually a special case,
# and we would prefer a unified segmentation and communication logic.
def get_tensor_split_parallel_mode(is_head: bool = False) -> ParallelMode:
    tp_mode = gpc.config.parallel.tensor.mode

    if tp_mode == "isp" and is_head is False:
        return ParallelMode.WEIGHT
    else:
        return ParallelMode.TENSOR


def get_head_parallel_mode() -> ParallelMode:
    return ParallelMode.TENSOR


def get_parallel_strategies_split_mode(linear_name: str) -> str:
    tp_mode = gpc.config.parallel.tensor.mode

    if linear_name in ("head", "output"):
        return "head"
    elif linear_name in ("wqkv", "wq", "wk", "wv", "wkv", "w1", "w3", "w13"):
        return "column"
    elif linear_name in ("wo", "out_proj", "w2") and tp_mode == "isp":
        return "column"
    elif linear_name in ("wo", "out_proj", "w2"):
        return "row"
    else:
        return "unknown"


def partition_uniform(num_items: int, pipeline_parallel_size: int, num_chunks: int):
    assert (
        num_items % num_chunks == 0
    ), "Layer length should be divided by the number of chunks, otherwise parameter method is recomended"

    parts = [[] for _ in range(pipeline_parallel_size)]
    partition_items = num_items // num_chunks
    for idx in range(num_chunks):
        base_idx = idx * partition_items
        chunk_size = partition_items // pipeline_parallel_size
        left = pipeline_parallel_size - partition_items % pipeline_parallel_size
        if chunk_size == 0:
            raise ValueError("Some nodes in Pipeline have no requests")

        for p in range(pipeline_parallel_size):
            st = base_idx
            base_idx += chunk_size + (p >= left)
            parts[p].append((st, base_idx))

    indexes = []
    for _parts in parts:
        for s, e in _parts:
            indexes.extend(list(range(s, e)))
    assert len(indexes) == len(set(indexes)), indexes  # should have no duplicates
    assert set(indexes) == set(list(range(num_items))), (indexes, num_items)  # should have the same indexes as expected
    return parts


def revise_load_balance_v1(balance_list, min_value):
    for i in range(len(balance_list)):
        # 如果layer数量不够PP切分，先尝试从他后面一个cluster的PP中借一个layer
        count, sentinel = 1, balance_list[i]
        while balance_list[i] < min_value:
            if count == len(balance_list):
                if sentinel == balance_list[i]:
                    raise RuntimeError(f"Unable to continue splitting, balance_list: {balance_list}")
                count, sentinel = 1, balance_list[i]

            next_cluster = (i + count) % len(balance_list)
            if balance_list[next_cluster] - 1 >= min_value:
                balance_list[next_cluster] -= 1
                balance_list[i] += 1
            count += 1


def find_most_max_deviation(distributed, adjust_value):
    """返回和分布最大偏离的idx（注意不一定是绝对值的最大和最小）

    Args:
        distributed (_type_): _description_
        adjust_value (_type_): _description_

    Returns:
        _type_: _description_
    """
    # import pdb; pdb.set_trace();
    sums_act = sum(adjust_value)
    sums_dist = sum(distributed)

    if abs(sums_dist - 1) >= 1e-6:
        distributed = list(map(lambda x: x / sums_dist, distributed))
    if abs(sums_act - 1) >= 1e-6:
        adjust_value = list(map(lambda x: x / sums_act, adjust_value))

    max_positive_diff = 0
    max_negative_diff = 0

    max_pos_idx = -1
    max_neg_idx = -1

    for i, zips in enumerate(zip(distributed, adjust_value)):
        a, b = zips
        diff = abs(a - b)
        if b >= a:
            if diff > max_positive_diff:
                max_positive_diff = diff
                max_pos_idx = i
        else:
            if diff > max_negative_diff:
                max_negative_diff = diff
                max_neg_idx = i

    return max_pos_idx, max_neg_idx


def find_min_max_value(adjust_value):
    min_value, max_value = float("inf"), -float("inf")
    min_idx, max_idx = -1, -1
    for i in range(len(adjust_value)):
        if adjust_value[i] > max_value:
            max_value = adjust_value[i]
            max_idx = i
        if adjust_value[i] < min_value:
            min_value = adjust_value[i]
            min_idx = i
    return min_idx, max_idx


def greedy_filling(min_value, total_diff, balance_list, max_pos_idx, max_neg_idx):
    if total_diff > 0:  # 超出的部分需要减去，比如layer数量
        # 但是每个clusater有自身的下界要求
        if balance_list[max_pos_idx] - 1 >= min_value:
            balance_list[max_pos_idx] -= 1
            total_diff -= 1
            return False
        else:
            _, maxidx = find_min_max_value(balance_list)
            if balance_list[maxidx] - 1 < min_value:
                raise ValueError(f"Unable to continue splitting, balance_list: {balance_list}, min_value: {min_value}")
            balance_list[maxidx] -= 1
            total_diff -= 1
            return False

    if total_diff < 0:  # 不足的部分我们直接补齐，但是一般来说我们没有上界的要求
        balance_list[max_neg_idx] += 1
        total_diff += 1
        return False

    return True


def PP_mem_balance_filling(min_value, total_diff, balance_list):
    # 如果是PP，如果需要
    if total_diff > 0:
        for i in range(len(balance_list)):
            if total_diff > 0:
                if balance_list[i] - 1 < min_value:
                    raise ValueError(
                        f"Unable to continue splitting, balance_list: {balance_list}, min_value: {min_value}"
                    )
                balance_list[i] -= 1
                total_diff -= 1
            else:
                return True

    if total_diff < 0:
        for i in range(len(balance_list) - 1, 0, -1):
            if total_diff < 0:
                balance_list[i] += 1
                total_diff += 1
            else:
                return True

    return False


def revise_load_balance_v2(
    base_value: int, min_value: int, distributed: List[float], relax_boundary: bool = False
) -> List[int]:
    """_summary_

    Args:
        base_value (int): 基准值，各个cluster的具体值根据该值上下浮动
        min_value (int): 每一项取值的下界
        distributed (List[float]): 负载均衡的分布
        total_sums (List[int]): 原始输入的总和
        relax_boundary (bool, optional): 是否可以放松总和的上界. Defaults to False.
            如果对Layer进行负载均衡则必须为False，如果对micro_num则为True

    Raises:
        ValueError: 某些情况下负载均衡是不可行的，比如 micro_num = 1 等情况
            这个时候我们会放弃负载均衡，沿用用户原始的配置

    Returns:
        List[int]: 负载均衡后的结果
    """
    # 检查每一项目
    all_nums = len(distributed) * base_value
    sums_dist = sum(distributed)
    distributed = list(map(lambda x: x / sums_dist, distributed))
    balance_list = list(map(lambda ratio: round(all_nums * ratio), distributed))

    while True:
        total_diff = sum(balance_list) - all_nums

        max_pos_idx, max_neg_idx = find_most_max_deviation(distributed, balance_list)

        # 检查总和是否等于初始值（layer数量和global bsz），尝试进行靠拢
        if not greedy_filling(min_value, total_diff, balance_list, max_pos_idx, max_neg_idx):
            continue

        # 检查每一项是否满足下界，在尽量不改变sum的情况下继续微调
        if balance_list[max_neg_idx] < min_value:
            # 从最大正偏移处借一个值
            if balance_list[max_pos_idx] - 1 >= min_value:
                balance_list[max_pos_idx] -= 1
                balance_list[max_neg_idx] += 1
            else:
                # 如果不能再借
                if not relax_boundary:
                    raise ValueError(f"Unable to continue splitting, balance_list: {balance_list}")
                else:
                    balance_list[max_neg_idx] += 1
                    relax_boundary = False
        else:
            break

    return balance_list


def weighted_sum(weight, value):
    w_sums = sum(weight)
    if abs(w_sums - 1) >= 1e-6:
        weight = list(map(lambda x: x / w_sums, weight))

    sums = 0
    for w, v in zip(weight, value):
        sums += w * v
    return sums


def cluster_load_balance():

    peak_tflops = []
    capacities = []
    gpus_per_cluster = []

    for cluster_info in gpc.clusters:
        peak_tflops.append(cluster_info.peak_tflops)
        capacities.append(cluster_info.capacity)
        gpus = cluster_info.node_num * cluster_info.gpu_per_node
        gpus_per_cluster.append(gpus)

    # capacity_first_cluster = sorted(cluster_list, key=lambda x: x.capacity)
    # tflops_first_cluster = sorted(cluster_list, key=lambda x: x.peak_tflops)

    global_bsz = gpc.config.data.global_bsz
    micro_bsz = gpc.config.data.micro_bsz
    seq_len = gpc.config.data.seq_len
    dp_size = gpc.get_world_size(ParallelMode.DATA)
    cluster_name = gpc.clusters[gpc.get_cluster_local_rank()].name
    rank = gpc.get_global_rank()

    # 根据单卡的峰值tflops来确定micro_num比例
    tflops = []
    for cluster in gpc.clusters:
        tflops.append(cluster.peak_tflops)

    # 负载均衡
    if gpc.get_world_size(ParallelMode.PIPELINE) == 1:
        # import pdb; pdb.set_trace()
        micro_num_all = global_bsz // (micro_bsz * seq_len)
        micro_num = micro_num_all // dp_size

        min_value = 1
        base_value = micro_num
        total_sums = micro_num_all  # TODO: 需不要考虑dp的大小
        relax_boundary = True

    else:
        pp_size = gpc.get_world_size(ParallelMode.PIPELINE)
        assert len(gpc.clusters) % 2 == 0
        layer_per_cluster = gpc.config.model.layer_num // (pp_size // len(gpc.clusters))

        min_value = pp_size // len(gpc.clusters)  # 每个pp stage至少分到一层
        base_value = layer_per_cluster
        total_sums = gpc.config.model.layer_num
        relax_boundary = False

    balance_results = revise_load_balance_v2(
        base_value=base_value,
        min_value=min_value,
        distributed=tflops,
        relax_boundary=relax_boundary,
    )

    new_sum = sum(balance_results)
    old_sum = base_value * len(gpc.clusters)
    if new_sum != old_sum:
        if relax_boundary:
            print(f"Warrning: allow relax constraints, now/old: {new_sum}/{old_sum}")
        else:
            raise ValueError(f"Unexcepted no relax_boundary but new_sum != base_value: {new_sum}/{old_sum}")

    if gpc.get_world_size(ParallelMode.PIPELINE) == 1:
        gpc.config.data.micro_num = balance_results[gpc.get_cluster_local_rank()]
        gpc.micro_num_list = np.array(balance_results)

        print(
            f"Rank: {rank}, cluster_name: {cluster_name}, balance_results: {balance_results}, \
balance micro_num: {gpc.config.data.micro_num}"
        )
    else:
        new_layer_num = balance_results[gpc.get_cluster_local_rank()]

        print(
            f"Rank: {rank}, cluster_name: {cluster_name},balance_results: {balance_results},  \
balance PP layer: {new_layer_num}"
        )

    return balance_results


def pipeline_parallel_sharding_wrapper(
    num_layers: int, num_chunks: int, model_builder: Callable, device: torch.device, **kwargs
):
    """
    build generic model 1d

    Args:
        num_layers (int): The number of layer.
        num_chunks (int): The number of partitions in pipeline parallel.
        device (Optional[Union[str, torch.device]]): The device will be used. torch.device("cuda") by default.

    """
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    all_parts = partition_uniform(num_layers, pipeline_size, num_chunks)
    parts = all_parts[pipeline_rank]

    if gpc.is_rank_for_log():
        logger.info("The layer sharding is %r.", all_parts)

    models = []

    for start, end in parts:
        kwargs["num_layers"] = end - start
        kwargs["first"] = start == 0
        # If there is no content in the final layer, assign the last layer.
        kwargs["last"] = end == num_layers and len(all_parts[-1]) != 0
        kwargs["device"] = device
        kwargs["start_layer_idx"] = start

        chunk = model_builder(**kwargs).to(device)
        setattr(chunk, "first_layer", start)
        setattr(chunk, "last_layer", end)

        models.append(chunk)

    torch.distributed.barrier()

    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    return model
