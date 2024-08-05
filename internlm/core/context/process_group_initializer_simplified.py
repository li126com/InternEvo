#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from copy import deepcopy
from enum import Enum

import torch
import torch.distributed as dist

from internlm.utils.timeout import LLM_NCCL_TIMEOUT
from internlm.core.context.process_group_initializer import ParallelMode

class ParallelMeta:
    def __init__(self, parallel_size, mode) -> None:
        self.parallel_size = parallel_size
        self.mode = mode

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self.mode}, {self.parallel_size}"


def determine_intra_inter_size_of_group(one_group_indexs, intra_range=8):
    "Determine the inter size and intra size of a rank group."
    gourp_size = len(one_group_indexs)
    if gourp_size == 1:
        return 1, 1
    else:
        group_stride = one_group_indexs[1] - one_group_indexs[0]
        if group_stride >= intra_range:
            return 1, gourp_size
        else:
            intra_size = intra_range // group_stride
            inter_size = gourp_size // intra_size
            return max(1, intra_size), max(1, inter_size)


class Initializer:
    def __init__(
        self,
        rank: int,
        world_size: int,
        fake_mode: bool = False,
        tensor_mode: str = "fsp",
        parallel_info: dict = None,
    ):
        """Initialize communication groups

        Args:
            rank (int): global rank
            world_size (int): world size
            fake_mode (bool, optional): Whether to create actual NCCL communication
                groups.Defaults to False.
            tensor_mode (str, optional): ISP/FSP/MSP. Defaults to "fsp".
            parallel_info (dict, optional): parallel_info. Defaults to None.
        """
        self.rank = rank
        self.world_size = world_size
        self.fake_mode = fake_mode
        self.tensor_mode = tensor_mode
        self.parallel_info = parallel_info

        # assert sequence_parallel_size == tensor_parallel_size
        super().__init__()

    def init_dist_group(self, use_cpu: bool = False):
        parallel_info, world_size = self.parallel_info, self.world_size

        wp_size = parallel_info["wp"].parallel_size
        # tp_size = parallel_info["tp"].parallel_size
        # pp_size = parallel_info["pp"].parallel_size
        wdp_size = parallel_info["wdp"].parallel_size
        zero1_size = parallel_info["zero1"].parallel_size
        ep_size = parallel_info["ep"].parallel_size
        edp_size = parallel_info["edp"].parallel_size

        re_group_args = {}

        # stride_order means the placement priority of PG groups.
        stride_order = ["tp", "dp", "pp"]
        strides = {}

        def assemble_group(all_ranks, dim_name):
            for ranks in all_ranks:
                if self.fake_mode or len(all_ranks) == 1:
                    group, group_cpu = None, None
                else:
                    group = dist.new_group(ranks, timeout=LLM_NCCL_TIMEOUT)
                    if use_cpu:
                        group_cpu = (
                            dist.new_group(ranks, backend="gloo", timeout=LLM_NCCL_TIMEOUT)
                            if dist.get_backend() != "gloo"
                            else group
                        )
                    else:
                        group_cpu = None

                if self.rank in ranks:
                    local_rank = ranks.tolist().index(self.rank)
                    group_world_size = len(ranks)
                    process_group = group
                    cpu_group = group_cpu
                    ranks_in_group = ranks.tolist()

            new_all_ranks = []
            for ranks in all_ranks:
                new_all_ranks.append(ranks.tolist())

            return (
                local_rank,
                group_world_size,
                process_group,
                cpu_group,
                ranks_in_group,
                new_all_ranks,
                parallel_info[dim_name].mode,
            )

        def split_orthogonal_sub_group(dim_name, indexs, size, stride):
            assert size <= world_size, f"{dim_name} stride: {size} should less then worldsize: {world_size} !"

            indexs = indexs.reshape(-1, stride).T.reshape(-1)
            all_ranks = torch.split(indexs, size)

            return indexs, assemble_group(all_ranks, dim_name)

        def split_horizontal_sub_group(dim_name, indexs, size, stride):
            assert size <= world_size, f"{dim_name} stride: {size} should less then worldsize: {world_size} !"

            indexs = indexs.reshape(stride, -1).reshape(-1)
            all_ranks = torch.split(indexs, size)

            return indexs, assemble_group(all_ranks, dim_name)

        count = 0
        for dim_name in stride_order:
            parallel_size = parallel_info[dim_name].parallel_size
            if parallel_size == 1:
                continue

            if count == 0:
                strides[dim_name] = 1
            else:
                strides[dim_name] = strides[old_dim_name] * parallel_info[old_dim_name].parallel_size

            father_indexs, group_args = split_orthogonal_sub_group(
                dim_name, torch.arange(start=0, end=world_size), size=parallel_size, stride=strides[dim_name]
            )
            re_group_args[dim_name] = group_args

            if dim_name == "dp":
                """
                "EP, EDP, and ZeRO are auxiliary parallel modes within DP."
                """
                if wp_size == 1 and self.tensor_mode != "isp":
                    re_group_args["zero1"] = split_horizontal_sub_group("zero1", father_indexs, zero1_size, zero1_size)[
                        1
                    ]
                    print(f"re_group_args['zero1']: {re_group_args['zero1']}")

                # MoE expert group is subgroup of data parallel group
                if ep_size > 1:
                    ep_indexs, group_ep_args = split_horizontal_sub_group(
                        "ep", father_indexs, size=ep_size, stride=ep_size
                    )
                    re_group_args["ep"] = group_ep_args
                    re_group_args["edp"] = split_orthogonal_sub_group("edp", ep_indexs, edp_size, ep_size)[1]

                one_group_indexs = group_args[4]  # one group ranks
                intra_dp_size, inter_dp_size = determine_intra_inter_size_of_group(one_group_indexs)

                # It will be used in drawing heatmap.
                parallel_info["intra_dp"].parallel_size = intra_dp_size
                parallel_info["inter_dp"].parallel_size = inter_dp_size

                # The only parallel group with a higher priority than DP is TP.
                # see: stride_order = ["tp", "dp", "pp"]
                high_priority_group = parallel_info["tp"].parallel_size

                re_group_args["intra_dp"] = split_horizontal_sub_group(
                    "intra_dp", father_indexs, size=intra_dp_size, stride=high_priority_group
                )[1]

                re_group_args["inter_dp"] = split_orthogonal_sub_group(
                    "inter_dp", father_indexs, size=inter_dp_size, stride=intra_dp_size
                )[1]

            elif dim_name == "tp":
                """
                The situation with isp is somewhat complex. When using isp, the head/embedding is partitioned
                according to the Megatron-TP method and uses the TP communication group, while other modules
                are partitioned according to the WP communication group and reuse the TP communication group
                (but perform DeepSpeed-Ulysses instead of Megatron-TP). Therefore,
                for head/embedding, their Zero1 communication group is orthogonal to the TP group,
                for other modules, their Zero1 communication group is the Wdp communication group
                (orthogonal to the WP/TP communication groups).
                FIXME: Can this be further simplified?
                """
                if self.tensor_mode == "isp":
                    if wp_size == 1:
                        re_group_args["zero1"] = split_horizontal_sub_group(
                            "zero1", father_indexs, zero1_size, zero1_size
                        )[1]
                    else:
                        wp_index, re_group_args["wp"] = split_horizontal_sub_group(
                            "wp", torch.arange(start=0, end=world_size), wp_size, wp_size
                        )
                        re_group_args["wdp"] = split_orthogonal_sub_group("wdp", wp_index, wdp_size, wp_size)[1]
                        re_group_args["zero1"] = split_orthogonal_sub_group(
                            "zero1", father_indexs, zero1_size, wp_size
                        )[1]

            count += 1
            old_dim_name = dim_name

        for name, info in parallel_info.items():
            if info.parallel_size == 1:
                # If the degree of parallelism is 1, for logical consistency,
                # we still need to create a logical communication group
                re_group_args[name] = assemble_group([torch.tensor([self.rank])], name)

        # If two groups are orthogonal to each other and one group has a parallelism degree of 1,
        # then the parallelism degree of the other group is world_size.
        if parallel_info["wp"].parallel_size == 1:
            re_group_args["wdp"] = tuple(list(deepcopy(re_group_args["dp"]))[0:-1] + [parallel_info["wdp"].mode])

        return re_group_args
