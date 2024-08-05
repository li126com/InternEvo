import functools
import math
import os
import socket

import torch
import torch.distributed as dist

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.simulator.common import BW, POSITIVE_INFINITY, CostType
from internlm.simulator.profiler.benchmark import register_comm_pref_initializer
from internlm.simulator.profiler.benchmark.multi_head_attn import run_fa_lat_test
from internlm.simulator.profiler.profiler import (
    draw_cal_pics,
    draw_pics,
    run_cal_profile,
    run_comm_profile,
    sync_all,
)
from internlm.utils.common import get_args, get_master_node, parse_args

cost_model = None
scale_ratio = [1.415134488, 1.208864145, 1.1, 1]

fa_cost_cache = {}


def get_group_id(rank, gpus_per_node, intra_size, inter_size):
    intra_x = rank % gpus_per_node
    inter_y = rank // gpus_per_node
    x_idx = intra_x // intra_size
    y_idx = inter_y // inter_size
    # y_stride = gpus_per_node // intra_size
    return x_idx, y_idx


def gen_cal_key(op_type: CostType):
    return f"{op_type}"


def gen_comm_key(op_name, intra_size, inter_size):
    return f"{op_name}_intra_{intra_size}_inter_{inter_size}"


def new_process_group(world_size, gpus_per_node, intra_size, inter_size):
    node_nums = world_size // gpus_per_node
    intra_group_stride = gpus_per_node // intra_size
    inter_group_stride = node_nums // inter_size

    gid_2_group = [[None for _ in range(intra_group_stride)] for _ in range(inter_group_stride)]

    for j_outer in range(inter_group_stride):
        for i_outer in range(intra_group_stride):
            base_idx = i_outer * intra_size + j_outer * inter_size * gpus_per_node
            ranks = []
            for j in range(inter_size):
                idx = base_idx + j * gpus_per_node
                ranks.extend(list(range(idx, idx + intra_size, 1)))
            # if dist.get_rank() == 0:
            #     print(f"base_idx: {base_idx}, intra_size: {intra_size}, inter_size: {inter_size}, key: {key}, ranks: {ranks}", flush=True)
            group = dist.new_group(ranks, backend="nccl")
            gid_2_group[j_outer][i_outer] = (group, ranks)

    return gid_2_group


def gen_perf():
    if "RANK" not in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NPROCS"]
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(int(os.environ["RANK"]) % 8)
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = get_master_node()
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(12345)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    gpus_per_node = 8
    node_num = world_size // gpus_per_node
    config = dict(
        parallel=dict(
            zero1=dict(size=1),
            tensor=dict(size=gpus_per_node, mode="mtp"),
            pipeline=dict(size=world_size // gpus_per_node, interleaved_overlap=True),
            weight=dict(size=1, overlap=True, memory_pool=True),
        ),
        clusters=[
            {
                "name": "nv_cluster",
                "peak_tflops": 320,
                "capacity": 80 * 1024**3,
                "intra_bw": 150,
                "inter_bw": 100,
                "gpu_per_node": 8,
                "node_num": 1,
            },
            {
                "name": "mx_cluster",
                "peak_tflops": 240,
                "capacity": 64 * 1024**3,
                "intra_bw": 150,
                "inter_bw": 100,
                "gpu_per_node": 8,
                "node_num": 1,
            },
        ],
    )

    gpc.load_config(config)

    init_method = f"tcp://[{host}]:{port}"
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend="nccl",
        init_method=init_method,
    )
    group = dist.GroupMember.WORLD

    gpc._register_dist(rank, world_size, group, None, list(range(world_size)), ParallelMode.GLOBAL)
    gpc._global_ranks[ParallelMode.GLOBAL] = rank
    gpc.set_device(local_rank)

    comm_test_list = [
        CostType.ALL2ALL,
        CostType.ALLREDUCE,
        CostType.REDUCESCATTER,
        CostType.ALLGATHER,
        CostType.BROADCAST,
    ]

    register_comm_pref_initializer()

    intra_comm_nums = int(math.log(gpus_per_node)) + 1  # 0,1,2,3
    inter_comm_nums = int(math.log(node_num)) + 1

    data_path = f"./prof_data"
    cal_pic_path = f"{data_path}/pics/cal"
    comm_pic_path = f"{data_path}/pics/comm"

    if dist.get_rank() == 0:
        os.makedirs(comm_pic_path, exist_ok=True)
    if dist.get_rank() == 0:
        os.makedirs(cal_pic_path, exist_ok=True)

    spline_model_dict = {}

    if dist.get_rank() == 0:
        comp_test_list = [CostType.LINEAR]
        for test_op in comp_test_list:
            tflop, tflops = run_cal_profile(test_op)
            spline_model = draw_cal_pics(cal_pic_path, f"{test_op}", tflop, tflops)
            spline_model_dict[gen_cal_key(test_op)] = spline_model

    sync_all()

    for i in range(inter_comm_nums):
        for j in range(intra_comm_nums):
            inter_size, intra_size = 2**i, 2**j
            if inter_size * intra_size != 1:

                x_idx, y_idx = get_group_id(rank, gpus_per_node, intra_size, inter_size)
                groups = new_process_group(world_size, gpus_per_node, intra_size, inter_size)

                for test_type in comm_test_list:
                    key = gen_comm_key(test_op, intra_size, inter_size)
                    if dist.get_rank() == 0:
                        print(
                            f"key: {key}, inter_size: {inter_size}, intra_size: {intra_size}, ranks: {groups[y_idx][x_idx][1]}",
                            flush=True,
                        )
                    pg = groups[y_idx][x_idx][0]
                    assert (
                        pg != -100
                    ), f"key: {key}, x_idx: {x_idx}, y_idx: {y_idx}, rank: {gpc.get_global_rank()}, ranks: {groups[y_idx][x_idx][1]}"
                    comm_vols, bws = run_comm_profile(test_type, pg, key)
                    sync_all()
                    if dist.get_rank() == 0:
                        spline_model_dict[key] = draw_pics(comm_pic_path, key, comm_vols, bws)

    print(f"rank: {gpc.get_global_rank()}, all done!", flush=True)

    if dist.get_rank() == 0:
        pt = os.path.join(data_path, "data.pt")
        with open(pt, "wb") as f:
            torch.save(spline_model_dict, f)


def init_cost_model(cost_model_path):
    global cost_model
    with open(cost_model_path, "rb") as f:
        cost_model = torch.load(f)


def coll_algo_bw(comm_op, size, n):
    if comm_op == CostType.ALL2ALL:
        if n <= 8:
            return size * (n - 1) / n
        else:
            # intra_parts = 8
            one_part = size / n
            return 8 * one_part * (n - 8 / n)
    elif comm_op == CostType.ALLREDUCE:
        return size * 2 * (n - 1) / n
    elif comm_op == CostType.REDUCESCATTER:
        return size * (n - 1) / n
    elif comm_op == CostType.ALLGATHER:
        return size * (n - 1) / n
    elif comm_op == CostType.BROADCAST:
        return size * (n - 1) / n
    elif comm_op == CostType.P2P:
        return size

    raise ValueError(f"unknown comm_op: {comm_op}")


def coll_bus_bw(comm_op, size, n):
    if comm_op == CostType.ALL2ALL:
        return size
    elif comm_op == CostType.ALLREDUCE:
        return size * 2
    elif comm_op == CostType.REDUCESCATTER:
        return size
    elif comm_op == CostType.ALLGATHER:
        return size
    elif comm_op == CostType.BROADCAST:
        return size
    elif comm_op == CostType.P2P:
        return size

    raise ValueError(f"unknown comm_op: {comm_op}")


# 需要判断是否打满带宽
def get_scale_ratio(scale):
    # 通信扩展惩罚系数
    if scale <= 16:
        return 1
    elif 16 < scale <= 32:
        return 1.1
    elif 32 < scale <= 64:
        return 1.2
    elif 64 < scale <= 256:
        return 1.3
    elif 256 < scale <= 512:
        return 1.4
    else:
        return 1.5


comm_matrix_dict = {}


def draw_heatmap(comm_nums: int, comm_volume: int, parallel_mode, use_rail_optim=False):
    """ "Draw a heatmap for communication volume of different parallel mode."

    Args:
        comm_nums (int): "Communication volume."
        comm_volume (int): "Communication volume."
        parallel_mode (_type_):
        use_rail_optim (bool, optional): Whether to consider multi-track optimization.
            Defaults to False.
    """

    global comm_matrix_dict

    if parallel_mode not in comm_matrix_dict:
        comm_matrix_dict[parallel_mode] = [
            [0 for _ in range(gpc.get_world_size(ParallelMode.GLOBAL))]
            for _ in range(gpc.get_world_size(ParallelMode.GLOBAL))
        ]

    comm_mat = comm_matrix_dict[parallel_mode]

    all_ranks = gpc.get_parallel_all_ranks(parallel_mode)

    for sub_id in range(len(all_ranks)):
        ranks = all_ranks[sub_id]
        for i in range(len(ranks)):
            world_size = len(ranks)

            if parallel_mode in [ParallelMode.TENSOR, ParallelMode.WEIGHT]:
                _comm_volume = comm_volume * gpc.config.model["num_layers"]
            elif parallel_mode == ParallelMode.PIPELINE:
                _comm_volume = 8 * 2 * comm_volume * comm_nums
            # elif parallel_mode == ParallelMode.ZERO1:
            #     _comm_volume = comm_volume * world_size
            else:
                _comm_volume = comm_volume

            if _comm_volume < 0:
                _comm_volume = -1 * _comm_volume

            chunk_size = _comm_volume / world_size
            is_intra = gpc.check_pg_is_intra(parallel_mode)

            print(f"sub_id: {sub_id}, parallel_mode: {parallel_mode}, is_intra: {is_intra}", flush=True)
            if is_intra:  # hard code, nvswitch
                for j in range(len(ranks)):
                    if j != i:
                        # print(f"len: {len(ranks)}, i: {i}, j: {j}", flush=True)
                        comm_mat[ranks[i]][ranks[j]] += chunk_size
                        comm_mat[ranks[j]][ranks[i]] += chunk_size
            else:
                if use_rail_optim:
                    pass
                else:
                    if parallel_mode in [ParallelMode.DATA, ParallelMode.ZERO1]:
                        inter_all_ranks = gpc.get_parallel_all_ranks(ParallelMode.INTER_DP_SZIE)
                        intra_all_ranks = gpc.get_parallel_all_ranks(ParallelMode.INTRA_DP_SZIE)

                        if parallel_mode == ParallelMode.DATA:
                            chunk_size /= 2

                        for k in range(len(inter_all_ranks)):
                            t_ranks = inter_all_ranks[k]
                            for p in range(len(t_ranks)):
                                for q in range(len(t_ranks)):
                                    if p != q:
                                        comm_mat[t_ranks[p]][
                                            t_ranks[q]
                                        ] += chunk_size  # / 4 # += (chunk_size // len(t_ranks))
                                        comm_mat[t_ranks[q]][
                                            t_ranks[p]
                                        ] += chunk_size  # / 4 # += (chunk_size // len(t_ranks))

                        for k in range(len(intra_all_ranks)):
                            t_ranks = intra_all_ranks[k]
                            for p in range(len(t_ranks)):
                                for q in range(len(t_ranks)):
                                    if p != q:
                                        comm_mat[t_ranks[p]][
                                            t_ranks[q]
                                        ] += chunk_size  # / 4 # += (chunk_size // len(t_ranks))
                                        comm_mat[t_ranks[q]][
                                            t_ranks[p]
                                        ] += chunk_size  # / 4  # += (chunk_size // len(t_ranks))

                        return
                    elif parallel_mode == ParallelMode.ZERO1:
                        inter_all_ranks = gpc.get_parallel_all_ranks(ParallelMode.INTER_DP_SZIE)
                        for k in range(len(inter_all_ranks)):
                            t_ranks = inter_all_ranks[k]
                            for p in range(len(t_ranks)):
                                for q in range(len(t_ranks)):
                                    if p != q:
                                        comm_mat[t_ranks[p]][t_ranks[q]] += chunk_size // len(t_ranks)
                                        comm_mat[t_ranks[q]][t_ranks[p]] += chunk_size // len(t_ranks)
                        return
                    elif parallel_mode == ParallelMode.PIPELINE:
                        if i < len(ranks) - 1:
                            comm_mat[ranks[i]][ranks[(i + 1) % world_size]] += _comm_volume
                            comm_mat[ranks[(i + 1) % world_size]][ranks[i]] += _comm_volume
                    else:
                        assert False


def get_comm_cost_from_logic(comm_volume: int, parallel_mode: ParallelMode, comm_op: CostType = None, comm_nums=1):
    """根据通信量获得近似的通信延迟,这个函数考虑了跨节点带宽content的情景
    所以为了正确计算延迟，传入的 comm_volume 必须是以单个rank视角下的通信量
    (即代码中实际传入的通信量)

    Args:
        comm_volume (int): 通信量, 单位B
        parallel_mode (ParallelMode): gpc并行模式
        comm_op (CostType, optional): 通信算子

    Returns:
        int: 通信延迟,是乘以10**4后并取整后的数值
    """
    scale = gpc.get_world_size(parallel_mode)

    if scale > 1 and get_args().draw_heatmap:
        draw_heatmap(comm_nums, comm_volume, parallel_mode)

    if parallel_mode == ParallelMode.PIPELINE:
        scale = 2

    if scale <= 1:
        return 0

    is_intra = gpc.check_pg_is_intra(parallel_mode)
    if not is_intra:
        num_partner = gpc.same_group_in_one_node(parallel_mode)
        assert num_partner <= 8, f"num_partner: {num_partner}"
        if parallel_mode == ParallelMode.WEIGHT:
            assert num_partner == 1
        if parallel_mode == ParallelMode.TENSOR:
            assert num_partner == 1
        comm_volume *= num_partner

    bw = BW.A800_NVL if is_intra else (BW.IB / get_scale_ratio(scale))
    return coll_algo_bw(comm_op, comm_volume, scale) / bw  # 转换成ms小数点保留两位


def get_comm_cost_from_cost_data(comm_volume: int, parallel_mode: ParallelMode, comm_op: CostType = None):
    """这里最佳的实现感觉是仿照NCCL的写法，建立起完整的通信代价矩阵，难点是如何确定一次集合通信包含了几个环
    （难道把nccl建图和搜索最优路径的代码用python重写一遍？）

    Args:
        comm_volume (int): _description_
        parallel_mode (ParallelMode): _description_
        comm_op (CostType, optional): _description_. Defaults to None.
    """
    pass


def get_cal_cost(cal_op, flop):
    global cost_model
    assert cost_model is not None
    try:
        flops = cost_model[gen_cal_key(cal_op)](flop)
    except Exception as e:
        print(f"error: {e}", flush=True)
        return POSITIVE_INFINITY
    else:
        return flop / flops  # latency in second.


def get_fa_cost(micro_bsz, seqlen, hidden_size, q_head, kv_head, dtype, is_fwd):
    fa_key = f"{micro_bsz}_{seqlen}_{hidden_size}_{q_head}_{kv_head}"

    if fa_key not in fa_cost_cache:
        print(f"not found FA key : {fa_key}, do profiling...")
        try:
            t_fwd, t_bwd = run_fa_lat_test(micro_bsz, seqlen, hidden_size, q_head, kv_head, dtype=torch.bfloat16)
        except RuntimeError as e:
            print(f"{e}, fa run fail", flush=True)
            t_fwd, t_bwd = float("inf"), float("inf")

        fa_cost_cache[fa_key] = t_fwd, t_bwd

    if is_fwd:
        return fa_cost_cache[fa_key][0]
    else:
        return fa_cost_cache[fa_key][1]


get_comm_cost = get_comm_cost_from_logic

allgather = functools.partial(get_comm_cost, comm_op=CostType.ALLGATHER)
reducescatter = functools.partial(get_comm_cost, comm_op=CostType.REDUCESCATTER)
broadcast = functools.partial(get_comm_cost, comm_op=CostType.BROADCAST)
p2p = functools.partial(get_comm_cost, comm_op=CostType.P2P)
alltoall = functools.partial(get_comm_cost, comm_op=CostType.ALL2ALL)
allreduce = functools.partial(get_comm_cost, comm_op=CostType.ALLREDUCE)
