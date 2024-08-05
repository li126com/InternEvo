#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging
import os
import socket

import torch

# from internlm.core.context.parallel_context import reset_global_context
from internlm.core.context import Config, ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.context.random import reset_seed
from internlm.core.parallel.shard import cluster_load_balance, partition_uniform
from internlm.initialize.launch import args_sanity_check, launch
from internlm.simulator.common import AlgoType, cal_block_p_elem, cal_model_p_elem

# from internlm.simulator.formulas.context import ParallelMode, check_and_modify_parallel_config
# from internlm.simulator.formulas.context import global_context as gpc
from internlm.simulator.formulas.mem import (
    get_backward_mem_peak,
    get_block_output_mm,
    get_block_threshold,
    get_head_input_mm,
    get_head_output_mm,
    get_memory_pool_mm,
    get_norm_output_mm,
    get_p2p_buffer_size,
    get_rotary_emb_sincos_cache_mm,
)
from internlm.simulator.formulas.overlap import TransformerOverlapOneLayer
from internlm.simulator.profiler.perf_comm import (
    allreduce,
    broadcast,
    comm_matrix_dict,
    init_cost_model,
    p2p,
)
from internlm.simulator.tracker.comm_tracker import get_gloabl_comm_tracker
from internlm.simulator.tracker.mem_tracker import get_global_allocator

# from internlm.simulator.elements.tensor import FakeTensor
from internlm.simulator.utils import (
    LinsSolutionNoZ3,
    PPIter,
    SPIter,
    get_bsz_approximate,
    get_bsz_strict,
)
from internlm.utils.common import get_args, parse_args

# global llm logger
logger = logging.getLogger(__file__)


gloab_allocator = get_global_allocator()
global_comm_tracker = get_gloabl_comm_tracker()


def comm_dp_cost(dtype_size, algo, pp_blocks_elem, embedding_elem, zp) -> float:
    """The communication overhead introduced by partitioning OS for parameter synchronization"""
    # The communication overhead introduced by Wdp, where the input is the communication
    # volume from a DP rank perspective.
    # The parameters of MSP and FSP are partitioned by TP and need to be divided by sp_size.
    # The parameters of ISP are partitioned by WP and need to be divided by wp_size.
    if algo in [AlgoType.MSP, AlgoType.FSP]:
        # gradient sync
        wdp_latency = allreduce(dtype_size * (pp_blocks_elem + embedding_elem), ParallelMode.DATA)

        # parameter sync
        # zp_latency = zp * broadcast(dtype_size * (pp_blocks_elem + embedding_elem) / zp,
        # ParallelMode.ZERO1, comm_nums=zp)
        zp_latency = zp * broadcast(dtype_size * pp_blocks_elem / zp, ParallelMode.ZERO1, comm_nums=zp)

    elif algo == AlgoType.ISP:
        # gradient sync
        wdp_block_latency = allreduce(dtype_size * pp_blocks_elem, ParallelMode.WEIGHT_DATA)
        wdp_embedding_latency = allreduce(dtype_size * embedding_elem, ParallelMode.DATA)
        wdp_latency = wdp_block_latency + wdp_embedding_latency

        # parameter sync
        block_zp_latency = zp * broadcast(dtype_size * pp_blocks_elem / zp, ParallelMode.ZERO1, comm_nums=zp)
        embedding_zp_latency = broadcast(dtype_size * embedding_elem, ParallelMode.DATA)
        zp_latency = max(block_zp_latency, embedding_zp_latency)

    return zp_latency, wdp_latency


def pp_comm_overhead(dtype_size, seq_len, hidden_size, pp_size, sp_size, micro_bsz, micro_num):
    """Calculate the latency of P2P communication in PP."""
    if pp_size == 1:
        return 0

    p2p_buffer_size = get_p2p_buffer_size(dtype_size, seq_len, sp_size, micro_bsz, hidden_size)

    warmup_p2p_num = min(pp_size, micro_num)
    one_f_one_b_p2p_num = micro_num - 1
    cooldown_p2p_num = min(pp_size, micro_num)

    p2p_count = warmup_p2p_num + one_f_one_b_p2p_num + cooldown_p2p_num
    p2p_latency = p2p_count * p2p(p2p_buffer_size, ParallelMode.PIPELINE, comm_nums=p2p_count)
    return p2p_latency


def cal_cost(
    pp,
    sp,
    wp,
    zp,
    micro_bsz,
    micro_num,
    algo_type,
    world_size,
    activation_ckpt,
    pp_num_layers=None,
    max_pp_num_layers=None,
    debug=True,
    overlap_wdp=True,
) -> LinsSolutionNoZ3:
    if pp_num_layers is None or max_pp_num_layers is None:
        max_pp_num_layers = 0
        parts = partition_uniform(gpc.config.model.num_layers, pipeline_parallel_size=pp, num_chunks=1)
        for part in parts:
            start, end = part[0]
            num_layer = end - start
            if num_layer > max_pp_num_layers:
                max_pp_num_layers = num_layer
                # max_pp_rank = pp_rank

        pp_num_layers = max_pp_num_layers

        assert max_pp_num_layers > 0

    # Anti-fragmentation penalty
    # if algo_type in [AlgoType.MSP, AlgoType.FSP]:
    #     if sp * zp * wp * pp < (gpc.config.model_size / 1.5):
    #         if debug:
    #             print(f"NO solu: skip sp*zp*wp*pp< 4 solu!\n", flush=True)
    #         return None
    # else:
    #     if zp * wp * pp < (gpc.config.model_size / 1.5):
    #         if debug:
    #             print(f"NO solu: skip zp*wp*pp< 4 solu!\n", flush=True)
    #         return None

    now_global_bsz = micro_bsz * micro_num * gpc.config.data.seq_len * gpc.get_world_size(ParallelMode.DATA)

    dp = gpc.get_world_size(ParallelMode.DATA)
    one_layer_elem = cal_block_p_elem(
        gpc.config.model.hidden_size,
        q_head=gpc.config.model.num_attention_heads,
        kv_head=gpc.config.model.num_kv_attention_heads,
        multiple_of=gpc.config.model.multiple_of,
        mlp_ratio=gpc.config.model.mlp_ratio,
    )

    print(f"pp_num_layers: {pp_num_layers}, one_layer_elem: {one_layer_elem}", flush=True)
    pp_blocks_elem = pp_num_layers * one_layer_elem
    embedding_dp_shared_range = 1 if dp <= 1 else 2
    head_num = 1 if pp > 1 else 2
    embedding_elem = gpc.config.model.vocab_size * gpc.config.model.hidden_size

    if algo_type in [AlgoType.MSP, AlgoType.FSP]:
        embedding_elem_parallel = head_num * embedding_elem / wp / sp
        block_elem_parallel = pp_blocks_elem / wp / sp
        total_p_element = block_elem_parallel + embedding_elem_parallel
        total_os_element = total_p_element / zp
        os_mm_cost = gpc.config.dtype_size * gpc.config.fp32_ratio * 3 * total_os_element  # zp显存消耗
        p_g_mm_cost = 2 * gpc.config.dtype_size * total_p_element  # wp显存消耗
    else:
        embedding_elem_parallel = head_num * embedding_elem / sp
        block_elem_parallel = pp_blocks_elem / wp
        total_p_element = block_elem_parallel + embedding_elem_parallel
        total_os_element = (
            block_elem_parallel / zp + embedding_elem_parallel / embedding_dp_shared_range
        )  # embeding不会被zp切
        os_mm_cost = gpc.config.dtype_size * gpc.config.fp32_ratio * 3 * total_os_element  # zp显存消耗
        p_g_mm_cost = 2 * gpc.config.dtype_size * total_p_element  # wp显存消耗

    zp_comm_cost, wdp_comm_cost = comm_dp_cost(
        dtype_size=gpc.config.dtype_size,
        algo=algo_type,
        pp_blocks_elem=block_elem_parallel,
        embedding_elem=embedding_elem_parallel,
        zp=zp,
    )  # 计算dp相关的通信开销

    # zp_comm_cost=0
    if overlap_wdp:
        wdp_comm_cost = 0

    blocks_activation = get_block_threshold(
        algo=algo_type,
        micro_batch_size=micro_bsz,
        layer_num=gpc.config.model.num_layers,  # 显存阈值根据pp0来计算
        sp_size=sp,
        activation_ckpt=activation_ckpt,
        hidden_dim=gpc.config.model.hidden_size,
        sequence_length=gpc.config.data.seq_len,  # 这里一定要传入没切过的seqlen
        use_fa=gpc.config.model.use_flash_attn,
        head_num=gpc.config.model.num_attention_heads,
        dtype_size=gpc.config.dtype_size // 2,  # dtype_size要除以2，因为激活值计算公式是默认按照fp16类型来的
    )  # isp激活的话，不需要除以wp，因为需要allgather

    if algo_type == AlgoType.ISP:
        isp_mem_pool = get_memory_pool_mm(
            gpc.config.model.mlp_ratio, gpc.config.model.hidden_size, gpc.config.dtype_size
        )
    else:
        isp_mem_pool = 0

    pp_p2p_buffer = (
        get_p2p_buffer_size(gpc.config.dtype_size, gpc.config.data.seq_len, sp, micro_bsz, gpc.config.model.hidden_size)
        if pp > 1
        else 0
    )

    # 下面这些激活的计算不受到重计算的影响
    norm_activation = get_norm_output_mm(
        micro_bsz, gpc.config.data.seq_len, gpc.config.model.hidden_size, sp=sp, dtype_size=gpc.config.dtype_size
    )

    head_input_activation = get_head_input_mm(
        micro_bsz,
        gpc.config.data.seq_len,
        gpc.config.model.hidden_size,
        dtype_size=gpc.config.dtype_size,
        tp_size=sp,
        algo=algo_type,
    )
    head_output_activation = get_head_output_mm(
        micro_bsz, gpc.config.data.seq_len, gpc.config.model.vocab_size, dtype_size=gpc.config.dtype_size
    )
    rotary_emb_sincos_cache_mm = get_rotary_emb_sincos_cache_mm(
        seq_len=gpc.config.data.seq_len,
        pp_size=pp,
        hidden_dim=gpc.config.model.hidden_size,
        head_nums=gpc.config.model.num_attention_heads,
        layer_nums=gpc.config.model.num_layers,
        dtype_size=gpc.config.dtype_size,
    )
    # 对于pp0,占用的激活仍然是 layer_num 份
    block_output_activation = (
        gpc.config.model.num_layers
        * get_block_output_mm(
            micro_bsz, gpc.config.data.seq_len, gpc.config.model.hidden_size, sp=sp, dtype_size=gpc.config.dtype_size
        )
    ) * activation_ckpt  # 只有开启重计算才需要额外加上这部分block激活的输出
    backward_mem_peak = get_backward_mem_peak(
        seq_len=gpc.config.data.seq_len,
        micro_bsz=micro_bsz,
        dtype_size=gpc.config.dtype_size,
        vocab_size=gpc.config.model.vocab_size,
        tp_size=sp,
        hidden_size=gpc.config.model.hidden_size,
    )
    activation = (
        blocks_activation
        + norm_activation
        + head_input_activation
        + head_output_activation
        + block_output_activation
        + backward_mem_peak
    )

    # 总显存开销
    mem_cost1 = (
        p_g_mm_cost + os_mm_cost + activation + isp_mem_pool + rotary_emb_sincos_cache_mm + pp_p2p_buffer
    )  # fwd_bwd显存峰值(需要加上Grad吗？)
    mem_cost2 = p_g_mm_cost + os_mm_cost / 3 * 5  # adamw的显存峰值
    mem_cost = max(mem_cost1, mem_cost2)
    if mem_cost > gpc.config.mem_threshold:
        # A[pp_i][sp_i][wp_i][zp_i] = _100GB
        # C[pp_i][sp_i][wp_i][zp_i] = 0
        if debug:
            print(
                f"NO solu: mem_cost: {mem_cost/1024**3:.2f} GB > mem_threshold: \
{gpc.config.mem_threshold/1024**3:.2f} GB ---- p_g_mm_cost: {p_g_mm_cost/1024**3:.2f} GB, \
os_mm_cost: {os_mm_cost/1024**3:.2f} GB, activation: {activation/1024**3:.2f} GB\n",
                flush=True,
            )
        return None
    # else:
    #     A[pp_i][sp_i][wp_i][zp_i] = mem_cost

    try:
        (wp_comm_cost, sp_comm_cost, comp_wp, comp_attn,) = TransformerOverlapOneLayer(
            micro_bsz=micro_bsz,
            sp_size=sp,
            pp_size=pp,
            world_size=world_size,
            ckpt=activation_ckpt,
            seq_len=gpc.config.data.seq_len,  # 这里需要传原始的seqlen,因为这个类里面还会切sp
            vocab_size=gpc.config.model.vocab_size,
            dtype_size=gpc.config.dtype_size,
            hidden_dim=gpc.config.model.hidden_size,
            num_head=gpc.config.model.num_attention_heads,
            num_kv_head=gpc.config.model.num_kv_attention_heads,
            mlp_ratio=gpc.config.model.mlp_ratio,
            multiple_of=gpc.config.model.multiple_of,
        )._get_overlap(algo_type)
    except KeyError as e:
        print(f"not found FA key: {e}", flush=True)
        return None

    if wp > 1:
        overlap_latency = min(comp_wp, wp_comm_cost) * gpc.config.wp_penalty_coefficient + max(comp_wp, wp_comm_cost)
    else:
        overlap_latency = comp_wp

    def overlaped_fwd_bwd_cost():
        return overlap_latency + sp_comm_cost + comp_attn

    if pp == 1:
        fwd_bwd_cost = gpc.config.model.num_layers * overlaped_fwd_bwd_cost()
        grad_acc = micro_num
        all_fwd_bwd_cost = grad_acc * fwd_bwd_cost  # 算上梯度累积的fwdbwd开销
        pp_comm_cost = 0
    else:
        # 注意这里要使用 max_pp_num_layers 来计算pp的延迟，而不是pp0的 num layer
        fwd_bwd_cost = max_pp_num_layers * overlaped_fwd_bwd_cost()  # 1个pp micro step的fwd_bwd开销
        all_fwd_bwd_cost = micro_num * fwd_bwd_cost  # pp的idea开销(不含bubble)
        pp_p2p_cost = pp_comm_overhead(
            dtype_size=gpc.config.dtype_size,
            seq_len=gpc.config.data.seq_len,
            hidden_size=gpc.config.model.hidden_size,
            pp_size=pp,
            sp_size=sp,
            micro_bsz=micro_bsz,
            micro_num=micro_num,
        )  # pp的p2p延迟
        pp_bubble_cost = (pp - 1) * fwd_bwd_cost  # pp的bubble开销
        pp_comm_cost = pp_p2p_cost + pp_bubble_cost  # pp总的额外开销

    total_latency = all_fwd_bwd_cost + pp_comm_cost + wdp_comm_cost + zp_comm_cost  # fwd_bwd_cost 乘上梯度累加

    # 计算tgs,为了方便取max这里乘了一个-1
    tgs = (-1 * now_global_bsz) / (world_size * total_latency)

    solu = LinsSolutionNoZ3(
        pp=pp,
        sp=sp,
        wp=wp,
        zp=zp,
        seq_len=gpc.config.data.seq_len,
        micro_bsz=micro_bsz,
        micro_num=micro_num,
        algo_type=algo_type,
        pp_comm_cost=pp_comm_cost,
        activation=activation,
        zp_comm_cost=zp_comm_cost,
        wp_comm_cost=wp_comm_cost,
        sp_comm_cost=sp_comm_cost,
        os_mm_cost=os_mm_cost,
        p_g_mm_cost=p_g_mm_cost,
        fwd_bwd_cost=fwd_bwd_cost,
        mem_cost=mem_cost,
        comp_wp=comp_wp,
        comp_attn=comp_attn,
        world_size=world_size,
        activation_ckpt=activation_ckpt,
        tgs=-1 * tgs,
        mem_pool_mm=isp_mem_pool,
        norm_activation=norm_activation,
        head_input_activation=head_input_activation,
        head_output_activation=head_output_activation,
        block_output_activation=block_output_activation,
        wdp_comm_cost=wdp_comm_cost,
        all_fwd_bwd_cost=all_fwd_bwd_cost,
        g_bsz=now_global_bsz,
        pp_p2p_buffer=pp_p2p_buffer,
        rotary_emb_sincos_cache_mm=rotary_emb_sincos_cache_mm,
        modelsize=gpc.config.param_elements / 10**9,
        backward_mem_peak=backward_mem_peak,
        blocks_activation=blocks_activation,
        overlap_latency=overlap_latency,
        total_latency=total_latency,
    )

    gpc.destroy()  # 销毁device mesh
    return solu


def run_loop(
    global_bsz,
    world_size,
    args,
    use_fixed_micro_bsz=False,
    use_strict_bsz=True,
    global_bsz_max=1,
    global_bsz_min=1,
    debug=True,
):
    gpc.load_config(config=Config.from_file(args.config))
    gpc.set_fake_mode(True)

    min_comm_cost, msp_min_cost, fsp_min_cost, isp_min_cost = (
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
    )
    min_cost_solution, msp_min_solu, fsp_min_solu, isp_min_solu = None, None, None, None

    L = gpc.config.model["num_layers"]
    KV_H = gpc.config.model["num_kv_attention_heads"]
    S = gpc.config.data["seq_len"]
    H = gpc.config.model["hidden_size"]
    MICRO_BSZ = gpc.config.data["micro_bsz"]
    MICRO_NUM = gpc.config.data["micro_num"]

    pp_search_range = PPIter(world_size, L)
    sp_search_range = SPIter(world_size, KV_H)
    wp_search_ranges = SPIter(world_size, world_size)
    # zp_search_ranges_max = SPIter(world_size, world_size)
    solutions_list = []
    algo_list = [AlgoType.ISP, AlgoType.MSP, AlgoType.FSP]

    gpc.config["param_elements"] = cal_model_p_elem(
        h=gpc.config.model.hidden_size,
        q_head=gpc.config.model.num_attention_heads,
        kv_head=gpc.config.model.num_kv_attention_heads,
        l=gpc.config.model.num_layers,
        vocab_size=gpc.config.model.vocab_size,
        mlp_ratio=gpc.config.model.mlp_ratio,
        multiple_of=gpc.config.model.multiple_of,
    )
    print(f"param_elements: {gpc.config['param_elements']}", flush=TimeoutError)

    for _, pp in enumerate(pp_search_range):
        for _, sp in enumerate(sp_search_range):
            if not use_fixed_micro_bsz:
                if use_strict_bsz:
                    bs_bns = get_bsz_strict(global_bsz, world_size, pp, sp, S)
                else:
                    bs_bns = get_bsz_approximate(global_bsz_max, global_bsz_min, world_size, pp, sp, S)

                if bs_bns is None or len(bs_bns) == 0:
                    if debug:
                        print(
                            f"NO solu: pp:{pp} , sp:{sp} can't find micro_bsz/micro_num for"
                            f"world_size:{world_size}, seq_len:{S}, \
global bsz range: [{global_bsz_min}-{global_bsz_max}]!",
                            flush=True,
                        )
                    continue
            else:
                bs_bns = [(MICRO_BSZ, MICRO_NUM)]

            for micro_bsz, micro_num in bs_bns:
                for algo_type in algo_list:
                    for activation_ckpt in [0, 1]:
                        for _, wp in enumerate(wp_search_ranges):
                            if algo_type in [AlgoType.MSP, AlgoType.FSP]:
                                if wp > 1:
                                    if debug:
                                        print("NO solu: msp, fsp not support wp>1 !", flush=True)
                                    continue
                                # Zp's search space is constrained by Wp, and it does not change in
                                # multiples of 8; instead, it increments as 1, 2, 3, ..
                                # Here, sp for msp and fsp is tp.
                                zp_search_range = world_size // pp // sp // wp
                            else:
                                # The implementation of zp in InternEvo is different from DeepSpeed.
                                # Zp is further partitioned on the basis of wp
                                zp_search_range = world_size // pp // wp
                            try:
                                assert H % sp == 0, f"embed_dim:{H} must be divisible by sp: {sp}"
                                assert KV_H % sp == 0, f"num_heads: {KV_H} must be divisible by sp: {sp}"
                                assert KV_H >= sp, f"num_heads: {KV_H} must bigger then sp: {sp}"
                                if algo_type != AlgoType.ISP:
                                    assert (
                                        wp * sp * pp * zp_search_range <= world_size
                                    ), f"{algo_type} not support wp and sp share same pg group."
                            except AssertionError as e:
                                if debug:
                                    print(f"NO solu: head assert {e}", flush=True)
                                continue

                            for _, zp in enumerate(range(1, zp_search_range + 1)):
                                # set config
                                print(
                                    f"activation_ckpt: {activation_ckpt}, micro_num: {micro_num}, \
micro_bsz: {micro_bsz}, pp: {pp}, wp: {wp}, zp: {zp}, sp: {sp}, {str(algo_type)}",
                                    flush=True,
                                )

                                # reset_global_context()

                                gpc.destroy()
                                gpc.config.model["checkpoint"] = activation_ckpt
                                gpc.config.parallel["zero1"]["size"] = zp
                                gpc.config.parallel["tensor"]["size"] = sp
                                gpc.config.parallel["tensor"]["mode"] = str(algo_type)
                                gpc.config.parallel["pipeline"]["size"] = pp
                                gpc.config.parallel["weight"]["size"] = wp
                                gpc.config.model_size = 7

                                gpc.config.data["micro_num"] = micro_num
                                gpc.config.data["micro_bsz"] = micro_bsz

                                gpc.config["mem_threshold"] = 80 * 1024**3
                                gpc.config["wp_penalty_coefficient"] = 0.1
                                gpc.config["dtype_size"] = 2
                                gpc.config["fp32_ratio"] = 2

                                reset_seed()

                                try:
                                    launch(
                                        config=gpc.config,
                                        local_rank=0,
                                        rank=0,
                                        world_size=world_size,
                                        host="127.0.0.1",
                                        port=12345,
                                        backend="nccl",
                                        seed=0,
                                        fake_mode=True,
                                    )
                                    args_sanity_check()
                                    assert hasattr(gpc, "config") and gpc.config is not None
                                except AssertionError as e:
                                    if debug:
                                        print(f"NO solu: build gpc failed: {e}\n", flush=True)
                                    continue
                                except ZeroDivisionError as e:
                                    if debug:
                                        print(f"NO solu: build gpc failed: {e}\n", flush=True)
                                    continue

                                solu = cal_cost(
                                    pp=pp,
                                    sp=sp,
                                    wp=wp,
                                    zp=zp,
                                    micro_bsz=micro_bsz,
                                    micro_num=micro_num,
                                    algo_type=algo_type,
                                    world_size=world_size,
                                    activation_ckpt=activation_ckpt,
                                )
                                if solu is None:
                                    continue
                                cost = solu.tgs
                                solutions_list.append(solu)
                                if cost < min_comm_cost:
                                    min_comm_cost = cost
                                    min_cost_solution = solu

                                print(f"solu: {solu}", flush=True)

                                if algo_type == AlgoType.MSP:
                                    if cost < msp_min_cost:
                                        msp_min_cost = cost
                                        msp_min_solu = solu
                                elif algo_type == AlgoType.FSP:
                                    if cost < fsp_min_cost:
                                        fsp_min_cost = cost
                                        fsp_min_solu = solu
                                elif algo_type == AlgoType.ISP:
                                    if cost < isp_min_cost:
                                        isp_min_cost = cost
                                        isp_min_solu = solu

    return solutions_list, min_comm_cost, min_cost_solution, msp_min_solu, fsp_min_solu, isp_min_solu


def run_warrper(global_bsz, world_size, args):
    solutions_list, min_comm_cost, min_cost_solution, msp_min_solu, fsp_min_solu, isp_min_solu = run_loop(
        global_bsz=global_bsz, world_size=world_size, args=args
    )

    if min_cost_solution is not None:
        solutions_list = sorted(solutions_list, key=lambda solu: solu.tgs, reverse=True)
        print("--------------------- END -----------------------", flush=True)
        # print("Max TGS:", min_comm_cost * -1)
        for i, solu in enumerate(solutions_list):
            if i > 5:
                break
            print(f"Top{i} Solution:", solu, flush=True)

        print("--------------------- MSP best solution -----------------------", flush=True)
        if msp_min_solu is not None:
            print(f"self.msp_min_solu : {msp_min_solu}")
        print("--------------------- FSP best solution -----------------------", flush=True)
        if fsp_min_solu is not None:
            print(f"self.fsp_min_solu : {fsp_min_solu}")
        print("--------------------- ISP best solution -----------------------", flush=True)
        if isp_min_solu is not None:
            print(f"self.isp_min_solu : {isp_min_solu}")

        final_res = {
            "algo_type": min_cost_solution.algo_type,
            "seq_len": min_cost_solution.seq_len,
            "micro_num": min_cost_solution.micro_num,
            "micro_bsz": min_cost_solution.micro_bsz,
            "pp_size": min_cost_solution.pp,
            "tp_size": min_cost_solution.sp,
            "wp_size": min_cost_solution.wp_size,
            "zp_size": min_cost_solution.zp_size,
            "activation_ckpt": bool(min_cost_solution.activation_ckpt),
        }
        print(final_res)
    else:
        print("No solution found")


def run_single(global_bsz=4 * 1024 * 1024):
    gpc.load_config(config=Config.from_file(args.config))
    gpc.set_fake_mode(True)
    print(f"gpc.config.parallel: {gpc.config.parallel}")

    gpc.config.data["global_bsz"] = global_bsz
    gpc.config.model_size = args.model_size
    gpc.config["mem_threshold"] = 80 * 1024**3
    gpc.config["wp_penalty_coefficient"] = 0.1
    gpc.config["dtype_size"] = 2
    gpc.config["fp32_ratio"] = 2
    gpc.config["param_elements"] = cal_model_p_elem(
        h=gpc.config.model.hidden_size,
        q_head=gpc.config.model.num_attention_heads,
        kv_head=gpc.config.model.num_kv_attention_heads,
        l=gpc.config.model.num_layers,
        vocab_size=gpc.config.model.vocab_size,
        mlp_ratio=gpc.config.model.mlp_ratio,
        multiple_of=gpc.config.model.multiple_of,
    )

    reset_seed()

    launch(
        config=gpc.config,
        local_rank=0,
        rank=0,
        world_size=world_size,
        host="127.0.0.1",
        port=12345,
        backend="nccl",
        seed=0,
        fake_mode=True,
    )
    args_sanity_check()
    assert hasattr(gpc, "config") and gpc.config is not None

    # cluster_load_balance()

    solu = cal_cost(
        pp=gpc.config.parallel["pipeline"]["size"],
        sp=gpc.config.parallel["tensor"]["size"],
        wp=gpc.config.parallel["weight"]["size"],
        zp=gpc.config.parallel["zero1"]["size"],
        micro_bsz=gpc.config.data["micro_bsz"],
        micro_num=gpc.config.data["micro_num"],
        algo_type=gpc.config.parallel["tensor"]["mode"],
        world_size=world_size,
        activation_ckpt=gpc.config.model["checkpoint"],
    )

    assert solu is not None

    print(f"solu: {solu}")

    # /mnt/inspurfs/wangguoteng.p/comm_matrix
    name = f"internlm2_{gpc.config.model_size}B.pt"
    pt = os.path.join(get_args().draw_heatmap_path, name)

    new_dict = {}
    for name, mat in comm_matrix_dict.items():
        print(f"name: {name}, mat: {mat}", flush=True)
        new_dict[str(name)] = mat

    with open(pt, "wb") as f:
        torch.save(new_dict, f=f)


if __name__ == "__main__":
    args = parse_args()
    hostname = socket.gethostname()
    world_size = args.world_size

    init_cost_model(get_args().pre_profiling_data_path)

    os.environ["fake_mode"] = "1"
    gloab_allocator.init_capcity = 80 * 1024**3
    gloab_allocator.capcity = 80 * 1024**3

    if get_args().run_all_solu:
        run_warrper(4096 * 1024, world_size, args)
    else:
        run_single(get_args().global_batch_size)
