#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context

import inspect
import os
import random
import socket
import sys
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.distributed as dist

from internlm.accelerator import get_accelerator
from internlm.core.context.process_group_initializer_simplified import Initializer, ParallelMeta
from internlm.utils.common import SingletonMeta
from internlm.utils.logger import get_logger
from internlm.utils.timeout import LLM_NCCL_TIMEOUT

from . import process_group_initializer as pgroup_initializer
from .process_group_initializer_simplified import ParallelMode
from .random import add_seed, get_seeds, set_mode
from internlm.utils.common import get_args

IS_REPLICA_ZERO_PARALLEL = "is_replica_zero_parallel"
# for isp, with optimizer split in dp group
IS_TENSOR_DATA_PARALLEL = "is_tensor_data_parallel"
# for mtp/msp/fsp, with optimizer split in zero1 group
IS_TENSOR_ZERO_PARALLEL = "is_tensor_zero_parallel"
IS_WEIGHT_ZERO_PARALLEL = "is_weight_zero_parallel"
IS_TENSOR_EXPERT_DATA_PARALLEL = "is_tensor_expert_data_parallel"

logger = get_logger(__file__)
internlm_accelerator = get_accelerator()


class Config(dict):
    """This is a wrapper class for dict objects so that values of which can be
    accessed as attributes.

    Args:
        config (dict): The dict object to be wrapped.
    """

    def __init__(self, config: dict = None):  # pylint: disable=W0231
        if config is not None:
            for k, v in config.items():
                self._add_item(k, v)

    def __missing__(self, key):
        raise KeyError(key)

    def __getattr__(self, key):
        try:
            value = super().__getitem__(key)
            return value
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        super().__setitem__(key, value)

    def _add_item(self, key, value):
        if isinstance(value, dict):
            self.__setattr__(key, Config(value))
        else:
            self.__setattr__(key, value)

    def update(self, config):
        assert isinstance(config, (Config, dict)), "can only update dictionary or Config objects."
        for k, v in config.items():
            self._add_item(k, v)
        return self

    @staticmethod
    def from_file(filename: str):
        """Reads a python file and constructs a corresponding :class:`Config` object.

        Args:
            filename (str): Name of the file to construct the return object.

        Returns:
            :class:`Config`: A :class:`Config` object constructed with information in the file.

        Raises:
            AssertionError: Raises an AssertionError if the file does not exist, or the file is not .py file
        """

        # check config path
        if isinstance(filename, str):
            filepath = Path(filename).absolute()
        elif isinstance(filename, Path):
            filepath = filename.absolute()

        assert filepath.exists(), f"{filename} is not found, please check your configuration path"

        # check extension
        extension = filepath.suffix
        assert extension == ".py", "only .py files are supported"

        # import the config as module
        remove_path = False
        if filepath.parent not in sys.path:
            sys.path.insert(0, (filepath))
            remove_path = True

        module_name = filepath.stem
        source_file = SourceFileLoader(fullname=str(module_name), path=str(filepath))
        module = source_file.load_module()  # pylint: disable=W4902,E1120,W1505

        # load into config
        config = Config()

        for k, v in module.__dict__.items():
            if k.startswith("__") or inspect.ismodule(v) or inspect.isclass(v):
                continue
            else:
                config._add_item(k, v)

        # remove module
        del sys.modules[module_name]
        if remove_path:
            sys.path.pop(0)

        return config


@dataclass
class ClusterInfo:
    # name: str
    name: str
    peak_tflops: float
    capacity: float
    intra_bw: float
    inter_bw: float
    gpu_per_node: int
    node_num: int


class ParallelContext(metaclass=SingletonMeta):
    """This class provides interface functions for users to get the parallel context,
    such as the global rank, the local rank, the world size, etc. of each device.

    """

    def __init__(self):
        # load config from file
        self._config = None
        self._group_map = {}
        # self.clusters = []
        self.clusters = {}
        self.micro_num_list = None
        self._init_attr()

    def _init_attr(self):
        # distributed settings
        self._global_ranks = dict()
        self._local_ranks = dict()
        self._world_sizes = dict()
        self._groups = dict()
        self._cpu_groups = dict()
        self._ranks_in_group = dict()
        self._all_ranks = dict()

        # default parallel args, will be overwritten during process group intialization
        self.world_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 1
        self.tensor_parallel_size = 1
        self.weight_parallel_size = 1
        self.zero1_parallel_size = -1
        self.nettest_parallel_size = 1
        self.expert_parallel_size = -1
        self.num_processes_on_current_node = -1
        self.virtual_pipeline_parallel_size = None
        self.virtual_pipeline_parallel_rank = None
        self._expert_parallel_group_names = []
        self.is_evaluating = False

    @property
    def config(self):
        return self._config

    @property
    def micro_bsz(self):
        return self._config.data.micro_bsz

    @property
    def micro_num(self):
        return self._config.data.micro_num

    @property
    def grad_accum_num(self):
        return self._config.data.gradient_accumulation

    @property
    def expert_parallel_group_names(self):
        return self._expert_parallel_group_names

    def load_config(self, config: Union[dict, str]):
        """Loads the configuration from either a dict or a file.

        Args:
            config (dict or str): Either a dict containing the configuration information or the filename
                of a file containing the configuration information.

        Raises:
            TypeError: Raises a TypeError if `config` is neither a dict nor a str.
        """
        if isinstance(config, str):
            self._config = Config.from_file(config)
        elif isinstance(config, dict):
            self._config = Config(config)
        else:
            raise TypeError("Invalid type for config, only dictionary or string is supported")

    @staticmethod
    def _check_parallel_mode(parallel_mode: ParallelMode):
        assert isinstance(
            parallel_mode, ParallelMode
        ), f"expected the argument parallel_mode to be of enum ParallelMode, but got {type(parallel_mode)}"

    def get_global_rank(self):
        """Returns the global rank of the current device.

        Returns:
            int: The global rank of the current device
        """
        return self._global_ranks[ParallelMode.GLOBAL]

    def get_local_rank(self, parallel_mode: ParallelMode):
        """Returns the local rank of the current device.

        Args:
            parallel_mode: The parallel mode for the rank.

        Returns:
            int: The local rank of the current device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)
        return self._local_ranks.get(parallel_mode, 0)

    def get_next_global_rank(self, parallel_mode: ParallelMode):
        """Returns the global rank of the next device.

        Args:
            parallel_mode: The parallel mode for the rank.

        Returns:
            int: The global rank of the next device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)

        # get rank and world size
        local_rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        ranks_in_group = self.get_ranks_in_group(parallel_mode)

        return ranks_in_group[(local_rank + 1) % world_size]

    def get_prev_global_rank(self, parallel_mode: ParallelMode):
        """Returns the global rank of the previous device.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            int: The global rank of the previous device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)

        # get rank and world size
        local_rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        ranks_in_group = self.get_ranks_in_group(parallel_mode)

        return ranks_in_group[(local_rank - 1) % world_size]

    def is_using_parallel_mode(self, parallel_mode):
        """Returns a boolean value indicating whether the current device is initilized with
        ParallelMode.DATA and its world_size is greater than 1.
        """
        return self.is_initialized(parallel_mode) and self.get_world_size(parallel_mode) > 1

    def is_first_rank(self, parallel_mode: ParallelMode):
        """Returns a boolean value indicating whether the current device is the first one
        among its group for `parallel_mode`.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            bool: a boolean value indicating whether the current device is the first one
            among its group for `parallel_mode`.
        """
        rank = 0
        if self.is_initialized(parallel_mode):
            rank = self.get_local_rank(parallel_mode)
        return rank == 0

    def is_rank_for_log(self):
        """Returns a boolean value indicating whether the current device should print log."""
        is_log_rank = (
            self.is_first_rank(ParallelMode.TENSOR)
            and self.is_first_rank(ParallelMode.WEIGHT)
            and self.is_first_rank(ParallelMode.DATA)
            and self.is_first_rank(ParallelMode.WEIGHT_DATA)
            and self.is_last_rank(ParallelMode.PIPELINE)
        )
        return is_log_rank

    def is_last_rank(self, parallel_mode: ParallelMode):
        """Returns a boolean value indicating whether the current device is the last one
        among its group for `parallel_mode`.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            bool: a boolean value indicating whether the current device is the first one
            among its group for `parallel_mode`.
        """
        rank = 0
        world_size = 1
        if self.is_initialized(parallel_mode):
            rank = self.get_local_rank(parallel_mode)
            world_size = self.get_world_size(parallel_mode)
        return rank == world_size - 1

    def is_pipeline_first_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if self.virtual_pipeline_parallel_size is not None and self.virtual_pipeline_parallel_rank != 0:
                return False
        return self.is_first_rank(ParallelMode.PIPELINE)

    def is_pipeline_last_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if (
                self.virtual_pipeline_parallel_size is not None
                and self.virtual_pipeline_parallel_rank != self.virtual_pipeline_parallel_size - 1
            ):
                return False
        return self.is_last_rank(ParallelMode.PIPELINE)

    def is_no_pp_or_last_stage(self):
        # NOTICE!!!, this will ignore virutal stage
        return not self.is_initialized(ParallelMode.PIPELINE) or self.is_last_rank(ParallelMode.PIPELINE)

    def get_world_size(self, parallel_mode: ParallelMode):
        """Returns the world size for `parallel_mode`.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            int: The world size for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)
        return self._world_sizes.get(parallel_mode, 1)

    def get_group_size(self, process_group):
        if self.fake_mode:
            mode = self._group_map[id(process_group)]
            return self.get_world_size(mode)
        else:
            return dist.get_world_size(process_group)

    def get_group(self, parallel_mode: ParallelMode):
        """Returns the group of the current device for `parallel_mode`.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            torch.distributed.ProcessGroup: The group of the current device for `parallel_mode`.
        """
        self._check_parallel_mode(parallel_mode)
        if parallel_mode not in self._groups:
            return None
        else:
            return self._groups[parallel_mode]

    def get_ranks_in_group(self, parallel_mode: ParallelMode):
        """Returns the rank of the current device for `parallel_mode` in the group.

        Args:
            parallel_mode: The chosen parallel mode.

        Returns:
            int: The rank of the current device for `parallel_mode` in the group.
        """
        self._check_parallel_mode(parallel_mode)
        return self._ranks_in_group[parallel_mode]

    def get_cpu_group(self, parallel_mode: ParallelMode):
        self._check_parallel_mode(parallel_mode)
        return self._cpu_groups[parallel_mode]

    def init_global_dist(
        self,
        rank: int,
        world_size: int,
        backend: str,
        host: str,
        port: int,
        use_cpu: bool = False,
        fake_mode: bool = False,
    ):
        """Initializes the global distributed environment

        Args:
           rank (int): rank for the default process group.
           world_size (int): world size of the default process group.
           backend (str): backend for ``torch.distributed``
           host (str): the master address for distributed training.
           port (str): the master port for distributed training.
           use_cpu (bool): whether to set up cpu process group.
        """

        # find   cluster info
        if "clusters" not in self.config:
            # nv_info = {
            #     "rank_range": [0, 8],
            #     "peak_tflops": 320,
            #     "capacity": 80 * 1024**3,
            #     "intra_bw": 150,
            #     "inter_bw": 100,
            # }
            nv_info = {
                "name": "nv_cluster",
                "peak_tflops": 320,
                "capacity": 80 * 1024**3,
                "intra_bw": 150,
                "inter_bw": 100,
                "gpu_per_node": 8,
                "node_num": 1,
            }
            self.set_cluster_info("nv_cluster", nv_info)
        else:
            for cluster in self.config.clusters:
                self.clusters.append(ClusterInfo(**cluster))

        # initialize the default process group
        if not fake_mode:
            init_method = f"tcp://[{host}]:{port}"
            dist.init_process_group(
                rank=rank,
                world_size=world_size,
                backend=backend,
                init_method=init_method,
                timeout=LLM_NCCL_TIMEOUT,
            )

            # None will give the default global process group for pytorch dist operations
            ranks = list(range(world_size))
            if use_cpu:
                cpu_group = (
                    dist.new_group(ranks, backend="gloo", timeout=LLM_NCCL_TIMEOUT)
                    if dist.get_backend() != "gloo"
                    else None
                )
            else:
                cpu_group = None

            group = dist.GroupMember.WORLD
        else:
            ranks = list(range(world_size))
            group, cpu_group = None, None

        self._register_dist(rank, world_size, group, cpu_group, ranks, [list(range(world_size))], ParallelMode.GLOBAL)
        self._global_ranks[ParallelMode.GLOBAL] = rank

    def _register_dist(self, local_rank, world_size, process_group, cpu_group, ranks_in_group, all_ranks, mode):
        self._check_parallel_mode(mode)
        self._local_ranks[mode] = local_rank
        self._world_sizes[mode] = world_size
        self._groups[mode] = process_group
        self._cpu_groups[mode] = cpu_group
        self._ranks_in_group[mode] = ranks_in_group
        self._group_map[id(process_group)] = mode
        self._all_ranks[mode] = all_ranks

    def check_sanity(self):
        """Checks sanity of the parallel context.

        Raises:
            AssertionError: Raises an AssertionError if the world size does not equal to the product
                of data parallel size, pipeline parallel size and tensor parallel size.
        """
        # for mtp/msp/fsp
        dps = self.data_parallel_size
        pps = self.pipeline_parallel_size
        tps = self.tensor_parallel_size
        ws = self.world_size
        assert ws == dps * pps * tps, (
            f"Expected the world size {ws} to be equal to data"
            f" parallel size ({dps}) * pipeline parallel size "
            f"({pps}) * tensor parallel size ({tps})"
        )

        # for isp
        wps = self.weight_parallel_size
        wdps = self.weight_data_parallel_size
        assert ws == wdps * pps * wps, (
            f"Expected the world size {ws} to be equal to weight data"
            f" parallel size ({wdps}) * pipeline parallel size "
            f"({pps}) * weight parallel size ({wps})"
        )

        assert self.zero1_parallel_size > 0

        # check for fsdp:
        # if zo_size < dp_size, ckpts saving will introduce redundent storage for model weights
        # because pytorch "ShardTensor" need to ensure current global rank equals to saved shard's global rank
        # pytorch vision: 1.13.1+cu117
        if self.data_parallel_size > self.zero1_parallel_size and self.config.parallel.zero1.get("fsdp", False):
            logger.warning(
                f"zo size: {self.zero1_parallel_size} < dp size: {self.data_parallel_size}, "
                "will introduce redundancy when saving fsdp model ckpts, recommend setting them to same value"
            )

        if self.tensor_mode == "isp":
            assert (
                self.zero1_parallel_size <= self.weight_data_parallel_size
            ), f"zero1_size:{self.zero1_parallel_size} should be less than wdp_size:{self.weight_data_parallel_size}"
            assert self.weight_data_parallel_size % self.zero1_parallel_size == 0, (
                f"weight_data_parallel_size:{self.weight_data_parallel_size} % "
                f"zero1_parallel_size: {self.zero1_parallel_size} != 0"
            )
        else:
            assert (
                self.zero1_parallel_size <= self.data_parallel_size
            ), f"zero1_size:{self.zero1_parallel_size} should be less than dp_size:{self.data_parallel_size}"
            assert (
                self.data_parallel_size % self.zero1_parallel_size == 0
            ), f"data_parallel_size:{self.data_parallel_size} % zero1_parallel_size: {self.zero1_parallel_size} != 0"

        assert self.zero1_parallel_size >= 1 and self.zero1_parallel_size <= self.world_size

        assert (
            self.data_parallel_size % self.num_experts == 0 or self.num_experts % self.data_parallel_size == 0
        ), "can not place the experts evenly"

    def _set_parallel_size_from_config(self, config: dict, key: str, attr_name: str):
        if key in config:
            ele = config[key]
            if isinstance(ele, int):
                setattr(self, attr_name, ele)
            elif isinstance(ele, dict):
                setattr(self, attr_name, ele["size"])
            else:
                raise NotImplementedError(
                    f'{"Parallel configuration does not support this kind of argument, please use int or dict"}'
                )

    def init_parallel_groups(self, fake_mode: bool = False):
        """Initializes the parallel groups."""

        # get rank and world size
        self.fake_mode = fake_mode
        rank = self.get_global_rank()
        world_size = self.get_world_size(ParallelMode.GLOBAL)
        self.world_size = world_size

        # set parallel size as attributes for global context
        parallel_config = self.config.get("parallel", None)
        if parallel_config is not None:
            # set default value for parallel size
            if "zero1" not in parallel_config:
                parallel_config._add_item("zero1", dict(size=-1, fsdp=False))
            if "pipeline" not in parallel_config:
                parallel_config._add_item("pipeline", dict(size=1, interleaved_overlap=False))
            if "tensor" not in parallel_config:
                parallel_config._add_item("tensor", dict(size=1, mode="mtp"))
            if "weight" not in parallel_config:
                parallel_config._add_item("weight", dict(size=1, overlap=False, memory_pool=False))

            # get value from config
            self._set_parallel_size_from_config(parallel_config, "weight", "weight_parallel_size")
            self._set_parallel_size_from_config(parallel_config, "tensor", "tensor_parallel_size")
            self._set_parallel_size_from_config(parallel_config, "pipeline", "pipeline_parallel_size")
            self._set_parallel_size_from_config(parallel_config, "zero1", "zero1_parallel_size")
            
        
        if get_args().use_simplified_gp_init:
            self._init_use_simplified_pg(rank, world_size, parallel_config)
        else:
            self._init_pg(rank, world_size, parallel_config)

    def _init_pg(self, rank, world_size, parallel_config):
        # the user should not set the data parallel size manually
        # instead, it should be calculated based on other parallel config
        try:
            self.tensor_mode = parallel_config["tensor"]["mode"]
        except AttributeError:
            self.tensor_mode = "mtp"
        
        self.num_experts = self.config.model.get("num_experts", 1)
        
        self.sequence_parallel_size = self.tensor_parallel_size
        self.data_parallel_size = max(1, self.world_size // self.pipeline_parallel_size // self.sequence_parallel_size)
        self.weight_data_parallel_size = max(
            1, self.world_size // self.pipeline_parallel_size // self.weight_parallel_size
        )

        if (
            isinstance(parallel_config["tensor"], dict)
            and parallel_config["tensor"]["mode"] == "isp"
        ):
            if self.zero1_parallel_size == -1:
                self.zero1_parallel_size = self.weight_data_parallel_size
            self.zero1_parallel_size = max(1, self.zero1_parallel_size)
            assert (
                self.zero1_parallel_size <= self.weight_data_parallel_size
            ), f"zero1_size:{self.zero1_parallel_size} should be less than wdp_size:{self.weight_data_parallel_size}"
            assert self.weight_data_parallel_size % self.zero1_parallel_size == 0, (
                f"weight_data_parallel_size:{self.weight_data_parallel_size} % "
                f"zero1_parallel_size: {self.zero1_parallel_size} != 0"
            )
        else:
            if self.zero1_parallel_size == -1:
                self.zero1_parallel_size = self.data_parallel_size
            self.zero1_parallel_size = max(1, self.zero1_parallel_size)
            assert (
                self.zero1_parallel_size <= self.data_parallel_size
            ), f"zero1_size:{self.zero1_parallel_size} should be less than dp_size:{self.data_parallel_size}"
            assert (
                self.data_parallel_size % self.zero1_parallel_size == 0
            ), f"data_parallel_size:{self.data_parallel_size} % zero1_parallel_size: {self.zero1_parallel_size} != 0"
        assert self.zero1_parallel_size >= 1

        # set sequence parallel value
        if "sequence_parallel" not in parallel_config:
            parallel_config._add_item("sequence_parallel", True)
        if isinstance(parallel_config["tensor"], int) or (
            isinstance(parallel_config["tensor"], dict)
            and parallel_config["tensor"]["mode"] == "mtp"
        ):
            parallel_config["sequence_parallel"] = False

        # the recommended nettest_parallel_size is 32 GPUs
        self.nettest_parallel_size = 32

        assert (
            self.data_parallel_size % self.config.model.get("num_experts", 1) == 0
            or self.config.model.get("num_experts", 1) % self.data_parallel_size == 0
        ), "can not place the experts evenly"

        # by default, expert_parallel_size equals to data_parallel_size, but if the number of experts is smaller
        # than data_parallel_size, set expert_parallel_size to be the number of experts to make sure each device
        # has one expert.
        self.expert_parallel_size = min(self.data_parallel_size, self.config.model.get("num_experts", 1))

        self.check_sanity()

        initializer_args = [
            rank,
            world_size,
            self.weight_parallel_size,
            self.weight_data_parallel_size,
            self.sequence_parallel_size,
            self.data_parallel_size,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
            self.zero1_parallel_size,
            self.nettest_parallel_size,
            self.expert_parallel_size,
        ]

        # run initialization of different process groups
        initializers = []
        if "gqa" in parallel_config and parallel_config["gqa"] is True:
            initializers.append(pgroup_initializer.Initializer_GQA(*initializer_args))
        initializers.append(pgroup_initializer.Initializer_Weight(*initializer_args))
        initializers.append(pgroup_initializer.Initializer_Weight_Data(*initializer_args))
        initializers.append(pgroup_initializer.Initializer_Tensor(*initializer_args))
        initializers.append(pgroup_initializer.Initializer_Data(*initializer_args))
        # initializers.append(pgroup_initializer.Initializer_ISP_Data(*initializer_args))
        # if (
        #     isinstance(parallel_config["tensor"], dict)
        #     and parallel_config["tensor"]["mode"] == TensorParallelMode.isp.name
        # ):
        if (
            isinstance(parallel_config["tensor"], dict)
            and parallel_config["tensor"]["mode"] == "isp"
        ):
            initializers.append(pgroup_initializer.Initializer_Zero1_ISP(*initializer_args))
        else:
            initializers.append(pgroup_initializer.Initializer_Zero1(*initializer_args))
        if isinstance(parallel_config["zero1"], dict) and parallel_config["zero1"].get("fsdp", False):
            initializers.append(pgroup_initializer.Initializer_Zero3_dp(*initializer_args))
        initializers.append(pgroup_initializer.Initializer_Nettest(*initializer_args))
        if self.pipeline_parallel_size > 1:
            initializers.append(pgroup_initializer.Initializer_Pipeline(*initializer_args))
        if self.config.model.get("num_experts", 1) > 1:
            initializers.append(pgroup_initializer.Initializer_Expert_Data(*initializer_args))
        for initializer in initializers:
            parallel_setting = initializer.init_dist_group()
            if isinstance(parallel_setting, list):
                for args in parallel_setting:
                    self._register_dist(*args)
            else:
                self._register_dist(*parallel_setting)
    
    def _init_use_simplified_pg(self, rank, world_size, parallel_config):
        try:
            self.tensor_mode = parallel_config["tensor"]["mode"]
        except AttributeError:
            self.tensor_mode = "mtp"

        self.num_experts = self.config.model.get("num_experts", 1)

        # the user should not set the data parallel size manually
        # instead, it should be calculated based on other parallel config
        self.sequence_parallel_size = self.tensor_parallel_size
        self.data_parallel_size = max(1, self.world_size // self.pipeline_parallel_size // self.sequence_parallel_size)
        self.weight_data_parallel_size = max(
            1, self.world_size // self.pipeline_parallel_size // self.weight_parallel_size
        )

        if self.tensor_mode == "isp":
            if self.zero1_parallel_size == -1:
                self.zero1_parallel_size = self.weight_data_parallel_size
        else:
            if self.zero1_parallel_size == -1:
                self.zero1_parallel_size = self.data_parallel_size

        # set sequence parallel value
        if self.tensor_mode == "mtp":
            parallel_config["sequence_parallel"] = False
        else:
            parallel_config._add_item("sequence_parallel", True)

        # by default, expert_parallel_size equals to data_parallel_size, but if the number of experts is smaller
        # than data_parallel_size, set expert_parallel_size to be the number of experts to make sure each device
        # has one expert.
        self.expert_parallel_size = min(self.data_parallel_size, self.num_experts)

        self.check_sanity()

        parallel_info = {
            "tp": ParallelMeta(self.tensor_parallel_size, ParallelMode.TENSOR),
            "wp": ParallelMeta(self.weight_parallel_size, ParallelMode.WEIGHT),
            "pp": ParallelMeta(self.pipeline_parallel_size, ParallelMode.PIPELINE),
            "dp": ParallelMeta(self.data_parallel_size, ParallelMode.DATA),
            "zero1": ParallelMeta(self.zero1_parallel_size, ParallelMode.ZERO1),
            "wdp": ParallelMeta(self.weight_data_parallel_size, ParallelMode.WEIGHT_DATA),
            "ep": ParallelMeta(self.expert_parallel_size, ParallelMode.EXPERT),
            "edp": ParallelMeta(self.data_parallel_size // self.expert_parallel_size, ParallelMode.EXPERT_DATA),
            "intra_dp": ParallelMeta(-1, ParallelMode.INTRA_DP_SZIE),
            "inter_dp": ParallelMeta(-1, ParallelMode.INTER_DP_SZIE),
        }

        initializer = Initializer(rank, world_size, self.fake_mode, self.tensor_mode, parallel_info)
        parallel_settings = initializer.init_dist_group()

        for name, parallel_setting in parallel_settings.items():
            # print(f"name: {name}, parallel_setting: {parallel_setting}")
            if isinstance(parallel_setting, list):
                for args in parallel_setting:
                    self._register_dist(*args)
            else:
                self._register_dist(*parallel_setting)

    def is_initialized(self, parallel_mode: ParallelMode):
        """Returns a boolean value indicating whether `parallel_mode` is initialized
        in the current system.
        """
        return parallel_mode in self._groups

    def set_fake_mode(self, fake_mode: bool = False):
        self.fake_mode = fake_mode

    def destroy(self):
        """Destroys the current distributed parallel environment."""
        if not self.fake_mode:
            for mode, group in self._groups.items():
                if mode is not ParallelMode.GLOBAL:
                    dist.destroy_process_group(group)
            # destroy global process group
            dist.destroy_process_group()

        self._init_attr()

    def set_device(self, device_ordinal: int = None):
        """Sets distributed processes to be bound to devices.

        Args:
           device_ordinal (int, optional): the device id to be bound to
        """
        global_rank = self.get_global_rank()
        if device_ordinal is None:
            devices_per_node = internlm_accelerator.device_count()
            device_ordinal = global_rank % devices_per_node

        internlm_accelerator.set_device(device_ordinal)
        logger.info(f"process rank {global_rank} is bound to host:{socket.gethostname()} device: {device_ordinal}")

    def set_seed(self, seed: int, dpseed_with_tpoffset: bool = False):
        """Sets seeds for all random libraries.

        Args:
            seed (int): seed for random states
        """
        pipeline_offset = self._local_ranks.get(ParallelMode.PIPELINE, 0)
        global_rank = self.get_global_rank()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # data parallel seed are kept the same in the same pipeline stage
        dp_seed = seed
        if dpseed_with_tpoffset:
            dp_seed = seed + pipeline_offset * 1024
        add_seed(ParallelMode.DATA, dp_seed)
        add_seed(ParallelMode.WEIGHT_DATA, dp_seed)
        add_seed(ParallelMode.DUMMY, dp_seed)

        # model parallel seeds are different across ranks
        if self.is_initialized(ParallelMode.TENSOR):
            tp_rank = self.get_local_rank(ParallelMode.TENSOR)
            tp_seed = seed + tp_rank + pipeline_offset * 1024
            add_seed(ParallelMode.TENSOR, tp_seed)

        if self.is_initialized(ParallelMode.WEIGHT):
            wp_rank = self.get_local_rank(ParallelMode.WEIGHT)
            wp_seed = seed + wp_rank + pipeline_offset * 1024
            add_seed(ParallelMode.WEIGHT, wp_seed)

        # we do not set the random state mode to ParallelMode.DATA until model is built (instead, we use a dummy mode
        # during model construction), this is because the random state will be different in different tensor parallel
        # device of the same data parallel group. The underlying reason is that the device of tp_rank = 0 will perform
        # additional random operations during the RowParallelLinear module building process.
        set_mode(ParallelMode.DUMMY)
        if self.is_using_parallel_mode(ParallelMode.TENSOR):
            set_mode(ParallelMode.TENSOR)
        if self.is_using_parallel_mode(ParallelMode.WEIGHT):
            set_mode(ParallelMode.WEIGHT)

        seeds = get_seeds()
        seed_str = ", ".join([f"{k}: {v}" for k, v in seeds.items()])
        logger.info(
            f"initialized seed on rank {global_rank}, "
            f"numpy: {seed}, python random: {seed}, {seed_str},"
            f"the default parallel seed is {ParallelMode.DATA}."
        )

    def set_virtual_pipeline_parallel_size(self, size):
        self.virtual_pipeline_parallel_size = size

    def set_virtual_pipeline_parallel_rank(self, rank):
        self.virtual_pipeline_parallel_rank = rank

    def get_parallel_all_ranks(self, mode):
        return self._all_ranks[mode]

    def get_cluster_local_rank(self):
        devices_offset = 0
        
        i = 0
        for _, cluster in self.clusters.items():
            devices_offset += cluster.gpu_per_node * cluster.node_num
            if self.get_global_rank() < devices_offset:
                return i
            i += 1
        
        # for i, cluster in enumerate(self.clusters):
        #     devices_offset += cluster.gpu_per_node * cluster.node_num
        #     if self.get_global_rank() < devices_offset:
        #         return i
        raise ValueError

    def get_model_parallel_size(self):
        return self.get_world_size(ParallelMode.PIPELINE) * self.get_world_size(ParallelMode.TENSOR)

    def check_pg_is_intra(self, parallel_mode: ParallelMode):
        pg_group_ranks = self.get_ranks_in_group(parallel_mode)
        if len(pg_group_ranks) > 8:
            return False
        else:
            min_rank = min(pg_group_ranks)
            max_rank = max(pg_group_ranks)
            return (max_rank - min_rank) <= 7

    def same_group_in_one_node(self, parallel_mode: ParallelMode):
        """获得一个节点内有多少个相同类型的PG, 在跨节点通信时会存在带宽竞争
        这里返回的相同PG的数量会乘上每个rank的通信数据量大小

        Args:
            parallel_mode (ParallelMode):

        Returns:
            int: 一个节点内相同类型的PG的数量
        """
        pg_group_ranks = self.get_ranks_in_group(parallel_mode)
        pg_group_ranks = sorted(pg_group_ranks)
        if len(pg_group_ranks) == 1:
            return 1
        else:
            stride = pg_group_ranks[1] - pg_group_ranks[0]
            if stride >= 8:
                return 8
            else:
                return stride

    def set_cluster_info(self, name: str, info: dict):
        self.clusters[name] = ClusterInfo(**info)

    def get_cluster_info(self, name: str):
        return self.clusters[name]

    def get_cluster_name_from_ip(self):
        """
        node_ip_list = [
            'metax-c500-1',
            'metax-c500-2',
            'nvidia-node-1',
            'nvidia-node-2',
        ]
        """
        hostname = socket.gethostname()
        cluster_name = hostname.split("-")[0]
        return cluster_name

    def sort_rank_based_on_ip_and_capacity(self):
        Capacity = []

        def sort_rank(x, y):
            x_name = self.get_cluster_name_from_ip(x)
            y_name = self.get_cluster_name_from_ip(y)
            if x_name == y_name:
                return x_name > y_name
            else:
                x_c = self.clusters[x_name]["capacity"]
                y_c = self.clusters[y_name]["capacity"]
                return x_c > y_c

        for cluster_name, cluster_info in self.clusters.items():
            peak_tflops.append(cluster_info["peak_tflops"])
            # Alpha.append(cluster_info.rank_range[-1] - cluster_info.rank_range[-1] + 1)
            Capacity.append(cluster_info["capacity"])

    def switch_topology_aware_rank_scheduling():
        """
        Switch topology-aware rank scheduling can optimize the performance of small-scale
        collective communications. Currently only supported in Alibaba Cloud.
        """

        local_rank = int(os.environ["LOCAL_RANK"])
        cluster_name = get_cluster_name_from_ip()

        try:
            if cluster_name == "Ali":
                pass
            else:
                rank = int(os.environ["MLP_WORKER_RACK_RANK_INDEX"]) * 8 + local_rank
        except Exception as e:
            logger.error(
                f"The switch topology awareness error is reported, the reason is: {e}",
                "but don’t worry, this error will not affect normal training.",
                "If you train on Alibaba or Volcano Cloud, please contact wangguoteng or lijiaxing",
            )
        else:
            # If there is no any error, hack torch rank.
            os.environ["RANK"] = str(rank)
            if local_rank == 0:
                logger.info("Successfully bound node switch affinity!")


global_context = ParallelContext()
