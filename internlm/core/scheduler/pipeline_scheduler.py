#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/engine

import queue
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.engine import Engine
from internlm.core.naive_amp import NaiveAMPModel
from internlm.core.scheduler import comm
from internlm.utils.common import (
    SchedulerHook,
    check_data_is_packed,
    get_current_device,
    move_to_device,
)
from internlm.utils.logger import get_logger
from internlm.utils.parallel import is_using_isp
from internlm.utils.timeout import llm_timeout

from .base_scheduler import BaseScheduler
from internlm.accelerator import get_accelerator

internlm_accelerator = get_accelerator()
logger = get_logger(__file__)


def get_tensor_shape():
    if hasattr(gpc.config, "TENSOR_SHAPE"):
        return gpc.config.TENSOR_SHAPE

    if not gpc.is_initialized(ParallelMode.PIPELINE):
        return None

    if hasattr(gpc.config, "SEQ_LEN") and hasattr(gpc.config.data, "micro_bsz") and hasattr(gpc.config, "HIDDEN_SIZE"):
        if gpc.config.data.use_packed_dataset and gpc.is_evaluating is False:
            if gpc.config.parallel.sequence_parallel:
                sequence_world_size = gpc.get_world_size(ParallelMode.TENSOR)
                tensor_shape = (
                    1,
                    gpc.config.data["seq_len"] * gpc.config.data["micro_bsz"] // sequence_world_size,
                    gpc.config.model["hidden_size"],
                )
            else:
                tensor_shape = (
                    1,
                    gpc.config.data["seq_len"] * gpc.config.data["micro_bsz"],
                    gpc.config.model["hidden_size"],
                )
        else:
            if gpc.config.parallel.sequence_parallel:
                sequence_world_size = gpc.get_world_size(ParallelMode.TENSOR)
                tensor_shape = (
                    gpc.config.data["micro_bsz"],
                    gpc.config.data["seq_len"] // sequence_world_size,
                    gpc.config.model["hidden_size"],
                )
            else:
                tensor_shape = (
                    gpc.config.data["micro_bsz"],
                    gpc.config.data["seq_len"],
                    gpc.config.model["hidden_size"],
                )
        return torch.Size(tensor_shape)
    else:
        return None


def pack_return_tensors(return_tensors):
    output, label = tuple(zip(*return_tensors))
    if isinstance(output[0], torch.Tensor):
        output = torch.cat(output, dim=0)
    elif isinstance(output[0], (list, tuple)):
        output = tuple(torch.cat(tensors, dim=0) for tensors in zip(*output))
    else:
        raise TypeError("Output of model must be tensor or list/tuple of tensors")
    if isinstance(label[0], torch.Tensor):
        label = torch.cat(label, dim=0)
    elif isinstance(label[0], dict):
        merged_label = {k: [] for k in label[0].keys()}
        for d in label:
            for k, v in d.items():
                merged_label[k].append(v)
        label = {k: torch.cat(v, dim=0) for k, v in merged_label.items()}
    return output, label


@contextmanager
def switch_virtual_pipeline_parallel_rank(rank):
    prev_rank = gpc.virtual_pipeline_parallel_rank
    try:
        gpc.set_virtual_pipeline_parallel_rank(rank)
        yield
    finally:
        gpc.set_virtual_pipeline_parallel_rank(prev_rank)


@contextmanager
def switch_optimizer_grad_sync_skip_mode(optimizer, skip: bool = True):
    prev_mode = optimizer.skip_grad_reduce
    try:
        optimizer.skip_grad_reduce = skip
        yield
    finally:
        optimizer.skip_grad_reduce = prev_mode


class WeightGradStore:
    """
    When using zero bubble pp, WeightGradStore is used to store the args and func for computating weight grad.
    """

    cache = []
    weight_grad_queue = queue.Queue()
    pp_mode = None
    
    @classmethod
    def set_pp_mode(cls, mode):
        cls.pp_mode = mode

    @classmethod
    def size(cls):
        return cls.weight_grad_queue.qsize()

    @classmethod
    def put(cls, weight, bias, input_tensor, grad_output, has_d_bias, grad_compute_func, *args):
        if cls.pp_mode == "ZBH1":
            assert not gpc.is_first_rank(ParallelMode.PIPELINE), "pp rank 0 should not arrive here"
        # Store the weight gradient computation of linear layers.
        cls.cache.append((weight, bias, input_tensor, grad_output, has_d_bias, grad_compute_func, *args))

    @classmethod
    def flush(cls):
        if cls.pp_mode == "ZBH1" and gpc.is_first_rank(ParallelMode.PIPELINE):
            return
        # Collect all stored computations during backward as a W for each micro batch.
        cls.weight_grad_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls):
        if cls.pp_mode == "ZBH1" and gpc.is_first_rank(ParallelMode.PIPELINE):
            return
        assert cls.weight_grad_queue.qsize() > 0
        stored_w_grad_computation = cls.weight_grad_queue.get()
        # Run computation for a single W.
        for weight, bias, input_tensor, grad_output, has_d_bias, grad_compute_func, *args in stored_w_grad_computation:
            grad_weight, grad_bias = grad_compute_func(input_tensor, grad_output, has_d_bias)
            if is_using_isp():
                isp_grad_hook = args[0]
                grad_weight, _ = isp_grad_hook(grad_weight, async_op=False, is_bias=False)
                if grad_bias is not None:
                    grad_bias, _ = isp_grad_hook(grad_bias, async_op=False, is_bias=True)

            # Gradient Accumulation
            weight.grad = weight.grad + grad_weight.data if weight.grad is not None else grad_weight
            if has_d_bias:
                bias.grad = bias.grad + grad_bias.data if bias.grad is not None else grad_bias


class PipelineScheduler(BaseScheduler):
    """
    A helper schedule class for pipeline parallelism running environment.
    It uses non-interleaved 1F1B strategy. Other properties are similar as
    :class:`NonPipelineSchedule`.

    Args:
        num_microbatches (int): The number of microbatches.
        dtype (torch.dtype): Type of data. torch.float by default.
        data_process_func (Callable, optional):
            The post processing function which receives a micro batch of data, and it will be executed
            in `load_micro_batch`.
        tensor_shape (torch.Size, optional): Specified shape in pipeline communication.
        scatter_gather_tensors (bool, optional):
            If set to `True`, communication will be reduced over pipeline when using 1D tensor parallelization.
        scheduler_hooks (Optional[List[SchedulerHook]], optional): List of scheduler hooks.
    """

    def __init__(
        self,
        num_microbatches: int,
        dtype: torch.dtype = torch.float,
        data_process_func: Callable = None,
        tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
        scatter_gather_tensors: bool = False,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
    ):
        assert num_microbatches > 0, f"expected num_microbatches to be larger then 1, but got {num_microbatches}"

        assert not isinstance(
            tensor_shape, int
        ), "tensor_shape type should be one of Union[torch.Size, List[int], Tuple[int]]."

        super().__init__(data_process_func=data_process_func)

        self.num_microbatches = num_microbatches
        self.dtype = dtype
        self._hooks = scheduler_hooks

        self._tensor_shape = (
            tensor_shape if tensor_shape is None or isinstance(tensor_shape, torch.Size) else torch.Size(tensor_shape)
        )

        self.scatter_gather_tensors = scatter_gather_tensors and gpc.is_using_parallel_mode(ParallelMode.TENSOR)

        if gpc.config.parallel.sequence_parallel:
            self.scatter_gather_tensors = False

        # cache for the batch data
        self.batch_data = None

    @property
    def tensor_shape(self) -> torch.Size:
        return self._tensor_shape

    @tensor_shape.setter
    def tensor_shape(self, tensor_shape: torch.Size):
        self._tensor_shape = tensor_shape

    def pre_processing(self, engine):
        self.dtype = gpc.config.model.get("dtype", torch.half)

    @staticmethod
    def _call_engine(engine, data):  # pylint: disable=W0237
        if data is None:
            return None

        if isinstance(data, torch.Tensor):
            return engine(data)
        elif isinstance(data, (list, tuple)):
            return engine(*data)
        elif isinstance(data, dict):
            # print(f"data: {data}, {gpc.get_global_rank()}", flush=True)
            stage_output = data.pop("stage_output", None)
            if stage_output is None:
                return engine(**data)
            elif isinstance(stage_output, torch.Tensor):
                return engine(stage_output, **data)
            elif isinstance(stage_output, (tuple, list)):
                return engine(*stage_output, **data)
            else:
                raise TypeError(
                    f"Expected stage_output to be of type torch.Tensor, list, or tuple, "
                    f"but got {type(stage_output)}"
                )
        else:
            raise TypeError(f"Expected data to be of type torch.Tensor, list, tuple, or dict, but got {type(data)}")

    def load_batch(self, engine, data_iter):
        # Pipeline schedule just puts data in memory,
        batch_data, actual_batch_size = engine.load_batch(data_iter, to_gpu=False)

        # Even if 'use_flash_attn' is False, the data seen when the 'load_batch' is called is still packed,
        # because internlm's current train dataset is packed, even using dummy data.
        # The unpack operation is performed in load_micro_batch().
        if check_data_is_packed(batch_data):
            micro_num = actual_batch_size
        else:
            micro_num = actual_batch_size // gpc.config.data["micro_bsz"]

        self.microbatch_offset = 0
        self.batch_size = actual_batch_size
        self.batch_data, self.batch_label = batch_data
        self.bsz_stride = self.batch_size // micro_num
        # 'num_microbatches' is no longer an initialization parameter,
        # but is determined on the fly by the Scheduler.
        self.num_microbatches = micro_num  # Rampup or variable bsz size.

    def load_micro_batch(self):
        micro_batch_data, micro_batch_label = self._load_micro_batch(
            data=self.batch_data, label=self.batch_label, offset=self.microbatch_offset, bsz_stride=self.bsz_stride
        )

        if self.data_process_func:
            micro_batch_data, micro_batch_label = self.data_process_func(micro_batch_data, micro_batch_label)

        micro_batch_data["label"] = micro_batch_label
        self.microbatch_offset += self.bsz_stride

        return move_to_device(micro_batch_data)

    def _get_data_label_for_current_step(self, stage_output, micro_batch_data):
        if isinstance(micro_batch_data, (tuple, list)):
            assert not self._config.parallel["pipeline"].get("mode", "1F1B") == "ZBV"
            if gpc.is_first_rank(ParallelMode.PIPELINE):
                # for the first stage, we use the data from the
                # dataloader output by default
                data, label = micro_batch_data
            else:
                # for non-first stage, we use the output passed
                # by the previous as the model input
                data = stage_output
                _, label = micro_batch_data
        # normally this way
        elif isinstance(micro_batch_data, dict):
            label = micro_batch_data.pop("label", None)
            data = {"stage_output": stage_output, **micro_batch_data}

        return data, label  # pylint: disable=E0606

    def _call_hooks(self, func_name: str, *args, **kwargs) -> None:
        for hook in self._hooks:
            getattr(hook, func_name)(self, *args, **kwargs)

    def _get_current_microbatch_id(self, step_id: int) -> int:
        """
        Get the current microbatch ID based on the step ID.
        In 1f1b scheduler, the microbatch ID is the same as the step ID,
        but it is important to note that the step ID is calculated separately
        for forward and backward passes.
        """
        return step_id

    def _forward_step(
        self,
        engine,
        input_obj,
        return_tensors,
        return_output_label=True,
        accum_loss=None,
        accum_moe_loss=None,
    ):
        """
        Forward step for passed-in model. If it is the first stage, the input tensor
        is obtained from data_iterator, otherwise the passed-in input_obj is used.
        Returns output tensor. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            input_obj (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Input tensor for this pipeline stage.
            return_tensors (List[:class:`torch.Tensor`]): A list of tensors to return.
            return_output_label (bool, optional): Whether returns output labels.
            accum_loss (optional): Where accumulated loss stores.
            accum_moe_loss (optional): Where accumulated moe loss stores.
        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: output or the loss value of the current
                pipeline stage.
        """
        micro_batch_data = self.load_micro_batch()
        data, label = self._get_data_label_for_current_step(input_obj, micro_batch_data)

        self._call_hooks("before_forward", data)
        if hasattr(gpc.config.model, "num_experts"):
            # moe is used
            output_obj, moe_losses = self._call_engine(engine.model, data)
        else:
            output_obj = self._call_engine(engine.model, data)
        self._call_hooks("after_forward", output_obj)

        if gpc.is_last_rank(ParallelMode.PIPELINE):
            self._call_hooks("post_helper_func", output_obj, label)
            if return_output_label:
                return_tensors.append((output_obj, label))
            if accum_loss is not None:
                self._call_hooks("before_criterion", output_obj, label)
                loss = self._call_engine_criterion(engine, output_obj, label)
                self._call_hooks("after_criterion", loss)

                loss_reduced = loss / self.num_microbatches
                accum_loss.add_(loss_reduced.detach())
                output_obj = loss_reduced

        moe_loss = (
            sum(moe_losses) * gpc.config.loss.moe_loss_coeff  # pylint: disable=E0606
            if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1
            else torch.tensor(0.0, device=get_current_device(), dtype=gpc.config.model.get("dtype"))
        )
        # the moe_loss is computed among the "tensor" group if sequence parallel is enabled, so we need to do allreduce
        if gpc.config.parallel.sequence_parallel or gpc.config.parallel.expert.no_tp:
            dist.all_reduce(moe_loss, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.TENSOR))
            moe_loss.div_(gpc.get_world_size(ParallelMode.TENSOR))
        moe_loss /= self.num_microbatches
        accum_moe_loss.add_(moe_loss.detach())

        return output_obj, moe_loss

    def _backward_step(self, engine, step_id, input_obj, output_obj, output_obj_grad, moe_loss=None):
        """
        Backward step through the passed-in output tensor. If it is the last stage, the
        output_obj_grad is None, otherwise it is the gradients with respect to stage's output tensor.
        Returns the gradients with respect to the input tensor (None if first stage).
        This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            step_id (int): The ID of the current step.
            input_obj (Union[torch.Tensor, List[torch.Tensor]]): Input tensor for this stage.
            output_obj (Union[torch.Tensor, List[torch.Tensor]]): Output tensor for this stage.
            output_obj_grad (Union[torch.Tensor, List[torch.Tensor]]): Gradient of output tensor for this stage.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Gradient of input tensor.
        """

        # Retain the grad on the input_obj.
        if input_obj is not None:
            if isinstance(input_obj, torch.Tensor):
                input_obj.retain_grad()
            else:
                for in_tensor in input_obj:
                    if in_tensor is not None:
                        in_tensor.retain_grad()

        # Backward pass.

        # Only the last microbatch does syncing grad.
        skip_grad_sync = self._get_current_microbatch_id(step_id) != self.num_microbatches - 1

        self._call_hooks("before_backward", output_obj, output_obj_grad)
        with switch_optimizer_grad_sync_skip_mode(engine.optimizer, skip_grad_sync):
            if moe_loss is None or moe_loss.item() == 0.0:
                if output_obj_grad is None:
                    engine.backward(output_obj)
                else:
                    engine.backward_by_grad(output_obj, output_obj_grad)
            else:
                if output_obj_grad is None:
                    engine.backward(output_obj + moe_loss)
                else:
                    # scale the latent loss
                    moe_loss = moe_loss * engine.optimizer.loss_scale
                    # we perform chain rule here by projecting the grad to the direction of
                    # [output_obj_grad, 1], Because moe_loss have no relation with subsequent
                    # layer, we set it to None (will be ragarded as 1).
                    engine.backward_by_grad([output_obj, moe_loss], [output_obj_grad, None])

        # Collect the grad of the input_obj.
        input_obj_grad = None
        if input_obj is not None:
            if isinstance(input_obj, torch.Tensor):
                input_obj_grad = input_obj.grad
            else:
                input_obj_grad = []
                for in_tensor in input_obj:
                    input_obj_grad.append(in_tensor.grad)
        self._call_hooks("after_backward", input_obj_grad)

        return input_obj_grad

    def _forward_only_step(self, engine, return_loss=True, return_output_label=True):
        """
        This function performs forward only computation process. The scheduling of microbatches is similar to the
        warmup phase, where each microbatch first receives the forward input from the previous stage, then performs
        the forward computation, and finally passes the forward computation output to the next stage. There are two
        special cases to note:
        1. The first stage of the pipeline does not need to receive forward input; its input comes from the dataloader.
        2. The last stage of the pipeline does not need to send forward output; its output is returned to the user code
           for processing.

        Args:
            engine (colossalai.engine.Engine): internlm engine for training and inference.
            return_loss (bool, optional): Whether to return the accumulated loss.
            return_output_label (bool, optional): Whether to return outputs and labels.

        Returns:
            Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
                output, label, and accumulated loss.
        """

        # Input, output tensors only need to be saved when doing backward passes
        return_tensors = []
        accum_loss = (
            torch.zeros(1, device=get_current_device())
            if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True)
            else None
        )
        accum_moe_loss = torch.zeros(1, device=get_current_device())

        # Used for tensor meta information communication
        forward_recv_shapes = self.tensor_shape
        need_forward_meta = self.tensor_shape is None

        # Run all forward passes.
        for _ in range(self.num_microbatches):
            # Receive input from the previous stage
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                if forward_recv_shapes is None:
                    forward_recv_shapes = comm.recv_obj_meta()
                input_obj = comm.recv_forward(
                    forward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                input_obj = None

            # Perform forward computation
            output_obj, _ = self._forward_step(
                engine,
                input_obj,
                return_tensors,
                return_output_label=return_output_label,
                accum_loss=accum_loss,
                accum_moe_loss=accum_moe_loss,
            )

            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                if need_forward_meta:
                    comm.send_obj_meta(output_obj)
                    need_forward_meta = False  # send only once.
                # Send the forward computation output to the next stage
                assert output_obj.dtype == self.dtype
                comm.send_forward(output_obj, scatter_gather_tensors=self.scatter_gather_tensors)

        output, label = pack_return_tensors(return_tensors) if len(return_tensors) > 0 else (None, None)

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            dist.all_reduce(accum_moe_loss, group=gpc.get_group(ParallelMode.PIPELINE))

        if accum_loss is not None:
            accum_loss += accum_moe_loss

        return output, label, accum_loss, accum_moe_loss

    def _forward_backward_step(self, engine, return_loss=True, return_output_label=True):
        """
        This function schedules the forward and backward computation of microbatches in the pipeline in a 1F1B manner.
        It consists of three stages: warmup, 1F1B, and cooldown.

        1. Warmup Stage:
        The warmup stage performs num_warmup forward microsteps. The calculation of num_warmup is the pipeline length
        minus the rank of the current pipeline minus 1. For each microstep, it receives data as input from the previous
        stage, performs the forward computation, and then sends the result to the next stage.

        2. 1F1B Stage:
        The 1F1B stage consists of pairs of forward and backward microsteps. It performs num_1f1b_micropairs iterations,
        where num_1f1b_micropairs is calculated as the total number of microbatches minus the number of microbatches in
        the warmup stage. In each iteration, it first performs a forward computation, sends the result to the next
        stage, receives input for the backward computation, performs the backward computation, and finally sends the
        result to the previous stage to receive input for the next forward computation.

        3. Cooldown Stage:
        The cooldown stage performs the same number of iterations as the warmup stage. In each iteration, it receives
        input for the backward computation, performs the backward computation, and finally sends the result to the
        previous stage.

        There are two special cases to consider:
        1. The first stage of the pipeline does not need to receive forward input or send backward output. The last
        stage does not need to send forward output or receive backward input.
        2. Pay attention to the communication between stages and use additional communication to bridge the gap.

        Args:
            engine (Engine): The engine used for computation.
            return_loss (bool, optional): Whether to return the accumulated loss.
            return_output_label (bool, optional): Whether to return outputs and labels.

        Returns:
            Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
            The output, label, and accumulated loss.
        """

        num_warmup_microsteps = (
            gpc.get_world_size(ParallelMode.PIPELINE) - gpc.get_local_rank(ParallelMode.PIPELINE) - 1
        )
        num_warmup_microsteps = min(num_warmup_microsteps, self.num_microbatches)
        num_1f1b_micropairs = self.num_microbatches - num_warmup_microsteps

        # Input, output tensors only need to be saved when doing backward passes
        input_objs = []
        output_objs = []
        moe_losses = []
        return_tensors = []
        accum_loss = (
            torch.zeros(1, device=get_current_device())
            if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True)
            else None
        )
        accum_moe_loss = torch.zeros(1, device=get_current_device())

        # Used for tensor meta information communication
        forward_recv_shapes = self.tensor_shape
        backward_recv_shapes = None
        need_forward_meta = self.tensor_shape is None

        # Run warmup forward passes.
        for i in range(num_warmup_microsteps):
            # Receive the input from the previous stage
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                if forward_recv_shapes is None:
                    forward_recv_shapes = comm.recv_obj_meta()
                input_obj = comm.recv_forward(
                    forward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                input_obj = None

            # Perform forward computation
            output_obj, moe_loss = self._forward_step(
                engine,
                input_obj,
                return_tensors,
                return_output_label=return_output_label,
                accum_loss=accum_loss,
                accum_moe_loss=accum_moe_loss,
            )

            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                if isinstance(output_obj, torch.Tensor):
                    backward_recv_shapes = output_obj.shape
                else:
                    backward_recv_shapes = [out_tensor.shape for out_tensor in output_obj]

                if need_forward_meta:
                    comm.send_obj_meta(output_obj)
                    need_forward_meta = False  # send only once.

            # Send the output of forward computation of this pipeline stage to the next pipeline stage as input for
            # forward computation
            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                assert output_obj.dtype == self.dtype
                comm.send_forward(output_obj, scatter_gather_tensors=self.scatter_gather_tensors)

            input_objs.append(input_obj)
            output_objs.append(output_obj)
            moe_losses.append(moe_loss)
        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_1f1b_micropairs > 0:
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                if forward_recv_shapes is None:
                    forward_recv_shapes = comm.recv_obj_meta()
                input_obj = comm.recv_forward(
                    forward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                input_obj = None

        # Run 1F1B in steady state.
        for i in range(num_1f1b_micropairs):
            # Perform forward computation
            output_obj, moe_loss = self._forward_step(
                engine,
                input_obj,
                return_tensors,
                return_output_label=return_output_label,
                accum_loss=accum_loss,
                accum_moe_loss=accum_moe_loss,
            )

            if gpc.is_last_rank(ParallelMode.PIPELINE):
                output_obj_grad = None
            else:
                assert output_obj.dtype == self.dtype
                output_obj_grad = comm.send_forward_recv_backward(
                    output_obj,
                    backward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )

            # Add input_obj and output_obj to end of list.
            input_objs.append(input_obj)
            output_objs.append(output_obj)
            moe_losses.append(moe_loss)

            # Pop output_obj and output_obj from the start of the list for
            # the backward pass.
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            moe_loss = moe_losses.pop(0)

            input_obj_grad = self._backward_step(engine, i, input_obj, output_obj, output_obj_grad, moe_loss)

            if i == (num_1f1b_micropairs - 1):
                input_obj = None
                if not gpc.is_first_rank(ParallelMode.PIPELINE):
                    comm.send_backward(
                        input_obj_grad,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
            else:
                if gpc.is_first_rank(ParallelMode.PIPELINE):
                    input_obj = None
                else:
                    input_obj = comm.send_backward_recv_forward(
                        input_obj_grad,
                        forward_recv_shapes,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )

        # Run cooldown backward passes.
        for i in range(num_warmup_microsteps):
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            moe_loss = moe_losses.pop(0)

            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                output_obj_grad = comm.recv_backward(
                    backward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                output_obj_grad = None

            input_obj_grad = self._backward_step(
                engine, num_1f1b_micropairs + i, input_obj, output_obj, output_obj_grad, moe_loss
            )

            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                comm.send_backward(input_obj_grad, scatter_gather_tensors=self.scatter_gather_tensors)

        output, label = pack_return_tensors(return_tensors) if len(return_tensors) > 0 else (None, None)

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            dist.all_reduce(accum_moe_loss, group=gpc.get_group(ParallelMode.PIPELINE))

        if accum_loss is not None:
            accum_loss += accum_moe_loss

        return output, label, accum_loss, accum_moe_loss

    @llm_timeout(func_name="nointerleaved_forward_backward_step")
    def forward_backward_step(self, engine, data_iter, forward_only=False, return_loss=True, return_output_label=True):
        """Runs non-interleaved 1F1B schedule, with communication between pipeline stages.
        Returns a tuple with losses if the last stage, an empty tuple otherwise.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                Whether run forward step only. Default is false. If true, no backward will be run.
            return_loss (bool, optional): Whether returns the loss value. Default is true.
            return_output_label (bool, optional): If False, the output and label won't be returned.
        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss, moe_loss), loss and label could be None.
                The loss would be returned only in the last stage. And the moe_loss is accumulated from all stages.
        """

        assert (
            forward_only or return_loss
        ), "The argument 'return_loss' has to be True when 'forward_only' is False, but got False."

        # Load data first
        self.load_batch(engine, data_iter)

        if forward_only:
            output, label, accum_loss, accum_moe_loss = self._forward_only_step(
                engine, return_loss, return_output_label
            )
        else:
            output, label, accum_loss, accum_moe_loss = self._forward_backward_step(
                engine, return_loss, return_output_label
            )

        # Compatible for non-moe
        if hasattr(gpc.config.model, "num_experts"):
            return output, label, accum_loss, accum_moe_loss
        else:
            return output, label, accum_loss


class ZeroBubblePipelineScheduler(PipelineScheduler):
    """
    A helper schedule class for pipeline parallelism running environment.
    It uses non-interleaved 1F1B strategy. Other properties are similar as
    :class:`NonPipelineSchedule`.

    Args:
        num_microbatches (int): The number of microbatches.
        dtype (torch.dtype): Type of data. torch.float by default.
        data_process_func (Callable, optional):
            The post processing function which receives a micro batch of data, and it will be executed
            in `load_micro_batch`.
        tensor_shape (torch.Size, optional): Specified shape in pipeline communication.
        scatter_gather_tensors (bool, optional):
            If set to `True`, communication will be reduced over pipeline when using 1D tensor parallelization.
        scheduler_hooks (Optional[List[SchedulerHook]], optional): List of scheduler hooks.
    """

    def __init__(
        self,
        num_microbatches: int,
        dtype: torch.dtype = torch.float,
        data_process_func: Callable = None,
        tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
        scatter_gather_tensors: bool = False,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
    ):
        super().__init__(
            num_microbatches,
            dtype=dtype,
            data_process_func=data_process_func,
            tensor_shape=tensor_shape,
            scatter_gather_tensors=scatter_gather_tensors,
            scheduler_hooks=scheduler_hooks,
        )
        WeightGradStore.set_pp_mode("ZBH1")

    def _forward_backward_step(self, engine, return_loss=True, return_output_label=True):
        """
        This function schedules the forward and backward computation of microbatches in the pipeline in a 1F1B manner.
        It consists of three stages: warmup, 1F1B, and cooldown.

        1. Warmup Stage:
        The warmup stage performs num_warmup forward microsteps. The calculation of num_warmup is the pipeline length
        minus the rank of the current pipeline minus 1. For each microstep, it receives data as input from the previous
        stage, performs the forward computation, and then sends the result to the next stage.

        2. 1F1B Stage:
        The 1F1B stage consists of pairs of forward and backward microsteps. It performs num_1f1b_micropairs iterations,
        where num_1f1b_micropairs is calculated as the total number of microbatches minus the number of microbatches in
        the warmup stage. In each iteration, it first performs a forward computation, sends the result to the next
        stage, receives input for the backward computation, performs the backward computation, and finally sends the
        result to the previous stage to receive input for the next forward computation.

        3. Cooldown Stage:
        The cooldown stage performs the same number of iterations as the warmup stage. In each iteration, it receives
        input for the backward computation, performs the backward computation, and finally sends the result to the
        previous stage.

        There are two special cases to consider:
        1. The first stage of the pipeline does not need to receive forward input or send backward output. The last
        stage does not need to send forward output or receive backward input.
        2. Pay attention to the communication between stages and use additional communication to bridge the gap.

        Args:
            engine (Engine): The engine used for computation.
            return_loss (bool, optional): Whether to return the accumulated loss.
            return_output_label (bool, optional): Whether to return outputs and labels.

        Returns:
            Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
            The output, label, and accumulated loss.
        """

        num_warmup_microsteps = (
            gpc.get_world_size(ParallelMode.PIPELINE) - gpc.get_local_rank(ParallelMode.PIPELINE) - 1
        )
        num_warmup_microsteps = min(num_warmup_microsteps, self.num_microbatches)
        num_1f1b_micropairs = self.num_microbatches - num_warmup_microsteps

        # Input, output tensors only need to be saved when doing backward passes
        input_objs = []
        output_objs = []
        moe_losses = []
        return_tensors = []
        accum_loss = (
            torch.zeros(1, device=get_current_device())
            if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True)
            else None
        )
        accum_moe_loss = torch.zeros(1, device=get_current_device())

        # Used for tensor meta information communication
        forward_recv_shapes = self.tensor_shape
        backward_recv_shapes = None
        need_forward_meta = self.tensor_shape is None

        f_times = 0
        # Run warmup forward passes.
        for i in range(num_warmup_microsteps):
            # Receive the input from the previous stage
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                if forward_recv_shapes is None:
                    forward_recv_shapes = comm.recv_obj_meta()
                input_obj = comm.recv_forward(
                    forward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                input_obj = None

            # Perform forward computation
            output_obj, moe_loss = self._forward_step(
                engine,
                input_obj,
                return_tensors,
                return_output_label=return_output_label,
                accum_loss=accum_loss,
                accum_moe_loss=accum_moe_loss,
            )
            f_times += 1

            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                if isinstance(output_obj, torch.Tensor):
                    backward_recv_shapes = output_obj.shape
                else:
                    backward_recv_shapes = [out_tensor.shape for out_tensor in output_obj]

                if need_forward_meta:
                    comm.send_obj_meta(output_obj)
                    need_forward_meta = False  # send only once.

            # Send the output of forward computation of this pipeline stage to the next pipeline stage as input for
            # forward computation
            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                assert output_obj.dtype == self.dtype
                comm.send_forward(output_obj, scatter_gather_tensors=self.scatter_gather_tensors)

            input_objs.append(input_obj)
            output_objs.append(output_obj)
            moe_losses.append(moe_loss)
        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_1f1b_micropairs > 0:
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                if forward_recv_shapes is None:
                    forward_recv_shapes = comm.recv_obj_meta()
                input_obj = comm.recv_forward(
                    forward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                input_obj = None

        # Run 1F1B in steady state.
        for i in range(num_1f1b_micropairs):
            # Perform forward computation
            output_obj, moe_loss = self._forward_step(
                engine,
                input_obj,
                return_tensors,
                return_output_label=return_output_label,
                accum_loss=accum_loss,
                accum_moe_loss=accum_moe_loss,
            )
            f_times += 1

            if gpc.is_last_rank(ParallelMode.PIPELINE):
                output_obj_grad = None
            else:
                assert output_obj.dtype == self.dtype
                output_obj_grad = comm.send_forward_recv_backward(
                    output_obj,
                    backward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )

            # Add input_obj and output_obj to end of list.
            input_objs.append(input_obj)
            output_objs.append(output_obj)
            moe_losses.append(moe_loss)

            # Pop output_obj and output_obj from the start of the list for
            # the backward pass.
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            moe_loss = moe_losses.pop(0)

            input_obj_grad = self._backward_step(engine, i, input_obj, output_obj, output_obj_grad, moe_loss)

            if i == (num_1f1b_micropairs - 1):
                input_obj = None
                if not gpc.is_first_rank(ParallelMode.PIPELINE):
                    comm.send_backward(
                        input_obj_grad,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
            else:
                if gpc.is_first_rank(ParallelMode.PIPELINE):
                    input_obj = None
                else:
                    input_obj = comm.send_backward_recv_forward(
                        input_obj_grad,
                        forward_recv_shapes,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )

            WeightGradStore.flush()
            if i >= gpc.get_local_rank(ParallelMode.PIPELINE):
                WeightGradStore.pop()

        # Run cooldown backward passes.
        for i in range(num_warmup_microsteps):
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            moe_loss = moe_losses.pop(0)

            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                output_obj_grad = comm.recv_backward(
                    backward_recv_shapes,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                output_obj_grad = None

            input_obj_grad = self._backward_step(
                engine, num_1f1b_micropairs + i, input_obj, output_obj, output_obj_grad, moe_loss
            )

            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                comm.send_backward(input_obj_grad, scatter_gather_tensors=self.scatter_gather_tensors)

            WeightGradStore.flush()
            WeightGradStore.pop()

        while WeightGradStore.size() > 0:
            WeightGradStore.pop()

        output, label = pack_return_tensors(return_tensors) if len(return_tensors) > 0 else (None, None)

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            dist.all_reduce(accum_moe_loss, group=gpc.get_group(ParallelMode.PIPELINE))

        if accum_loss is not None:
            accum_loss += accum_moe_loss

        return output, label, accum_loss, accum_moe_loss

class InterleavedPipelineScheduler(PipelineScheduler):
    """
    Interleaved Pipeline Scheduler.
    """

    def __init__(
        self,
        num_microbatches: int,
        num_chunks: int,
        dtype: torch.dtype = torch.float,
        data_process_func: Callable = None,
        tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
        scatter_gather_tensors: bool = False,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
        communication_overlap: bool = False,
    ):
        """A helper schedule class for pipeline parallelism running environment.
        It uses interleaved 1F1B strategy. Other properties are similar as
        :class:`NonPipelineSchedule`.

        Args:
            num_microbatches (int): The number of microbatches.
            num_chunks (int): The number of model chunks.
            dtype (torch.dtype, optional): The data type of the tensors. Default is torch.float.
            data_process_func (Callable, optional):
                The preprocessing function which receives a batch of data, and it will be executed in `load_batch`.
            tensor_shape (torch.Size, optional): Specified shape in pipeline communication.
            scatter_gather_tensors (bool, optional):
                If set to `True`, communication will be reduced over pipeline when using 1D tensor parallelization.
            scheduler_hooks (List[SchedulerHook], optional): List of scheduler hooks. Default is None.
            communication_overlap (bool, optional): Whether to enable communication overlap. Default is False.
        """
        assert (
            num_microbatches % gpc.get_world_size(ParallelMode.PIPELINE) == 0
        ), f"num_microbatches: {num_microbatches} must be an integer multiple of pipeline parallel world size"

        assert (
            isinstance(num_chunks, int) and num_chunks > 0
        ), f"expected num_chunks to be an integer and larger than 0, but got {num_chunks}"

        super().__init__(
            num_microbatches,
            dtype=dtype,
            data_process_func=data_process_func,
            tensor_shape=tensor_shape,
            scatter_gather_tensors=scatter_gather_tensors,
            scheduler_hooks=scheduler_hooks,
        )

        gpc.set_virtual_pipeline_parallel_size(num_chunks)
        gpc.set_virtual_pipeline_parallel_rank(0)

        self._num_chunks = num_chunks
        self._communication_overlap = communication_overlap
        # switch 1f1b loop runner function according to communication overlap
        self._run_1f1b_loop = (
            self._run_1f1b_loop_with_overlap if communication_overlap else self._run_1f1b_loop_without_overlap
        )

        # states
        self._pp_size = gpc.get_world_size(ParallelMode.PIPELINE)
        self._pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

        self._accum_loss = None
        self._accum_moe_loss = None
        self._return_tensors = None
        self._input_objs = [[] for _ in range(num_chunks)]
        self._output_objs = [[] for _ in range(num_chunks)]
        self._output_obj_grads = [[] for _ in range(num_chunks)]
        self._moe_losses = [[] for _ in range(num_chunks)]

        self._input_obj_shapes = [self.tensor_shape for _ in range(num_chunks)]
        self._output_obj_shapes = [None for _ in range(num_chunks)]
        self._send_tensor_shape_flags = [self.tensor_shape is None for _ in range(num_chunks)]

    @property
    def tensor_shape(self) -> torch.Size:
        return self._tensor_shape

    @tensor_shape.setter
    def tensor_shape(self, tensor_shape: torch.Size):
        self._tensor_shape = tensor_shape
        self._input_obj_shapes = [self._tensor_shape for _ in range(self._num_chunks)]
        self._send_tensor_shape_flags = [self._tensor_shape is None for _ in range(self._num_chunks)]

    def _clear_state(self) -> None:
        self._accum_loss = None
        self._accum_moe_loss = None
        self._return_tensors = None
        self._input_objs = [[] for _ in range(self._num_chunks)]
        self._output_objs = [[] for _ in range(self._num_chunks)]
        self._output_obj_grads = [[] for _ in range(self._num_chunks)]
        self._moe_losses = [[] for _ in range(self._num_chunks)]

        self._input_obj_shapes = [self.tensor_shape for _ in range(self._num_chunks)]
        self._output_obj_shapes = [None for _ in range(self._num_chunks)]
        self._send_tensor_shape_flags = [self.tensor_shape is None for _ in range(self._num_chunks)]

    def load_batch(self, engine, data_iter):
        super().load_batch(engine, data_iter)
        # overwrite microbatch_offset, since model chunks load the same microbatch, and should tract the offset
        self.microbatch_offset = [0 for _ in range(self._num_chunks)]

    def load_micro_batch(self, model_chunk_id):
        print(f"load_micro_batch_1 {gpc.get_global_rank()}", flush=True)
        micro_batch_data, micro_batch_label = self._load_micro_batch(
            data=self.batch_data,
            label=self.batch_label,
            offset=self.microbatch_offset[model_chunk_id],
            bsz_stride=self.bsz_stride,
        )
        print(f"load_micro_batch_2 {gpc.get_global_rank()}", flush=True)
        if self.data_process_func:
            micro_batch_data, micro_batch_label = self.data_process_func(micro_batch_data, micro_batch_label)
        print(f"load_micro_batch_3 {gpc.get_global_rank()}", flush=True)
        micro_batch_data["label"] = micro_batch_label
        self.microbatch_offset[model_chunk_id] += self.bsz_stride
        print(f"load_micro_batch_4 {gpc.get_global_rank()}", flush=True)
        print(f"micro_batch_data {gpc.get_global_rank()}: {micro_batch_data}, {type(micro_batch_data)}, {len(micro_batch_data.items())}", flush=True)
        # internlm_accelerator.synchronize()
        # for k, v in micro_batch_data.items():
        #     print(f"{gpc.get_global_rank()} {k}: {type(v)}", flush=True)
        #     if isinstance(v, torch.Tensor):
        #         print(f"{gpc.get_global_rank()} {k}: {type(v)}, {v.device.type}, {get_current_device()}", flush=True)
        #         internlm_accelerator.synchronize()
        #         v = v.to(get_current_device())
        #         internlm_accelerator.synchronize()
        #         print(f"{gpc.get_global_rank()} finish move1", flush=True)
        #         v = v.detach()
        #         print(f"{gpc.get_global_rank()} finish move2", flush=True)
        result = move_to_device(micro_batch_data)
        print(f"load_micro_batch_5 {gpc.get_global_rank()}", flush=True)
        return result

    def _forward_step(self, engine, chunk_id):
        """Forward step for passed-in model. If it is the first stage, the input tensor
        is obtained from data_iterator, otherwise the passed-in input_obj is used.
        Returns output tensor. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            chunk_id (int): The id of model chunks.
        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: output or the loss value of the current
                pipeline stage.
        """
        gpc.set_virtual_pipeline_parallel_rank(chunk_id)

        if gpc.is_pipeline_first_stage() and len(self._input_objs[chunk_id]) == len(self._output_objs[chunk_id]):
            self._input_objs[chunk_id].append(None)
        input_obj = self._input_objs[chunk_id][-1]
        if input_obj is not None:
            assert input_obj.requires_grad == True
        
        if not gpc.is_pipeline_first_stage():
            assert input_obj is not None, f"{gpc.get_global_rank()} input is None"
        print(f"before load_micro_batch {gpc.get_global_rank()}", flush=True)
        micro_batch_data = self.load_micro_batch(chunk_id)
        # {'input_ids': tensor([[ 25, 166,   0,  ...,  18,  19,  20]]), 'cu_seqlens': tensor([   0, 2048], dtype=torch.int32), 'indexes': tensor([   0,    1,    2,  ..., 2045, 2046, 2047]), 'max_seqlen': 2048, 'label': torch.tensor([[ 166, -100,    1,  ...,   19,   20, -100]])}
        # micro_batch_data = {
        #     'input_ids': torch.randint(5, 10, (1, 2048), device=get_current_device(), ),
        #     'cu_seqlens': torch.tensor([   0, 2048], dtype=torch.int32, device=get_current_device()),
        #     'indexes': torch.randint(0, 10, (2048,), device=get_current_device()),
        #     'max_seqlen': 2048,
        #     'label': torch.randint(5, 10, (1, 2048), device=get_current_device()),
        # }
        print(f"after load_micro_batch {gpc.get_global_rank()}", flush=True)
        data, label = self._get_data_label_for_current_step(input_obj, micro_batch_data)

        self._call_hooks("before_forward", data)
        if hasattr(gpc.config.model, "num_experts"):
            output_obj, moe_losses = self._call_engine(engine.model[chunk_id], data)
        else:
            output_obj = self._call_engine(engine.model[chunk_id], data)
        # Convert output_obj to fp32 when last model chunk of last stage
        if gpc.is_pipeline_last_stage(ignore_virtual=False) and isinstance(engine.model[chunk_id], NaiveAMPModel):
            output_obj = engine.model[chunk_id].convert_to_fp32(output_obj)
        self._call_hooks("after_forward", output_obj)

        if gpc.is_pipeline_last_stage():
            self._call_hooks("post_helper_func", output_obj, label)

            if self._return_tensors is not None:
                self._return_tensors.append((output_obj, label))
            if self._accum_loss is not None:
                self._call_hooks("before_criterion", output_obj, label)
                loss = self._call_engine_criterion(engine, output_obj, label)
                self._call_hooks("after_criterion", loss)

                loss_reduced = loss / self.num_microbatches
                self._accum_loss.add_(loss_reduced.detach())
                output_obj = loss_reduced

        moe_loss = (
            sum(moe_losses) * gpc.config.loss.moe_loss_coeff  # pylint: disable=E0606
            if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1
            else torch.tensor(0.0, device=get_current_device(), dtype=gpc.config.model.get("dtype"))
        )
        # the moe_loss is computed among the "tensor" group if sequence parallel is enabled, so we need to do allreduce
        if gpc.config.parallel.sequence_parallel or gpc.config.parallel.expert.no_tp:
            dist.all_reduce(moe_loss, op=dist.ReduceOp.AVG, group=gpc.get_group(ParallelMode.TENSOR))
        moe_loss /= self.num_microbatches

        if self._accum_moe_loss is not None:
            self._accum_moe_loss.add_(moe_loss.detach())

        self._output_objs[chunk_id].append(output_obj)
        self._moe_losses[chunk_id].append(moe_loss)
        
        assert output_obj is not None, f"{gpc.get_global_rank()} chunk{chunk_id} output is None"

        return output_obj

    def _backward_step(self, engine, chunk_id, step_id):
        """
        Backward step for passed-in model. If it is the last stage, the input tensor
        is obtained from the previous forward step, otherwise the passed-in input_obj is used.
        Returns input tensor gradient. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            chunk_id (int): The id of model chunks.
            step_id (int): The current step id.

        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: input tensor gradient.
        """
        gpc.set_virtual_pipeline_parallel_rank(chunk_id)

        if gpc.is_pipeline_last_stage() and len(self._output_obj_grads[chunk_id]) == 0:
            self._output_obj_grads[chunk_id].append(None)

        input_obj = self._input_objs[chunk_id].pop(0)
        output_obj = self._output_objs[chunk_id].pop(0)
        output_obj_grad = self._output_obj_grads[chunk_id].pop(0)
        moe_loss = self._moe_losses[chunk_id].pop(0)

        input_obj_grad = super()._backward_step(engine, step_id, input_obj, output_obj, output_obj_grad, moe_loss)

        return input_obj_grad

    def _get_chunk_by_microbatch(self, step_id: int, backward: bool = False) -> int:
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = step_id % (self._pp_size * self._num_chunks)
        chunk_id = microbatch_id_in_group // self._pp_size
        
        if gpc.get_global_rank() == 0:
            print(f"step_id: {step_id}, self._pp_size * self._num_chunks: {self._pp_size * self._num_chunks}, microbatch_id_in_group: {microbatch_id_in_group}, chunk_id: {chunk_id}", flush=True)


        if backward:
            chunk_id = self._num_chunks - chunk_id - 1

        return chunk_id

    def _get_current_microbatch_id(self, step_id: int) -> int:
        # format:
        # microstep_id : 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
        # microbatch_id: 1  2  3  4  1  2  3  4  5  6  7  8  5  6  7  8
        num_microbatch_group = step_id // (self._pp_size * self._num_chunks)
        step_id_in_group = step_id % (self._pp_size * self._num_chunks)
        
        microbatch_id = num_microbatch_group * self._pp_size + step_id_in_group % self._pp_size

        return microbatch_id

    def _run_warmup_loop(
        self,
        engine: Engine,
        num_microsteps: int,
        num_warmup_microsteps: int,
        receive_extra_backward: bool = False,
        forward_only: bool = False,
    ) -> None:
        """
        Run the warm-up loop and prepare data for the 1F1B stage.

        During the warm-up process, for each execution, it first performs a forward computation,
        and then sends the computation result to the next stage.
        It also receives data for the next forward computation.
        Since the input for the first forward computation is not considered initially,
        it needs to receive data once at the beginning.

        After the warm-up is completed, we need to prepare data for the 1F1B stage.
        The data preparation process should be consistent with the communication method of the 1F1B stage.

        Args:
            engine (Engine): The engine to run the warm-up loop.
            num_microsteps (int): The total number of microsteps.
            num_warmup_microsteps (int): The number of warm-up microsteps.
            receive_extra_backward (bool, optional): Whether to receive extra backward input for the 1F1B stage.
                                                     Default is False.
            forward_only (bool, optional): Whether to only perform forward pass. Default is False.
        """
        if not gpc.is_pipeline_first_stage():
            if self._input_obj_shapes[0] is None:
                self._input_obj_shapes[0] = comm.recv_obj_meta()
            self._input_objs[0].append(
                comm.recv_forward(
                    self._input_obj_shapes[0],
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            )
        else:
            self._input_objs[0].append(None)

        for k in range(num_warmup_microsteps):
            chunk_id = self._get_chunk_by_microbatch(k)
            print(f"chunk_id: {chunk_id}", flush=True)

            output_obj = self._forward_step(engine, chunk_id)

            if forward_only:
                # when forward-only, no need to save tensors for a backward pass
                self._input_objs[chunk_id].pop()
                self._output_objs[chunk_id].pop()
                self._moe_losses[chunk_id].pop()

            if not gpc.is_pipeline_last_stage():
                if isinstance(output_obj, torch.Tensor):
                    self._output_obj_shapes[chunk_id] = output_obj.shape
                else:
                    self._output_obj_shapes[chunk_id] = [out_tensor.shape for out_tensor in output_obj]

                if self._send_tensor_shape_flags[chunk_id]:
                    comm.send_obj_meta(output_obj)
                    self._send_tensor_shape_flags[chunk_id] = False  # send only once for each chunk.

            # Determine if tensor should be received from previous stage.
            next_forward_chunk_id = self._get_chunk_by_microbatch(k + 1)

            with switch_virtual_pipeline_parallel_rank(next_forward_chunk_id):
                if not gpc.is_pipeline_first_stage() and self._input_obj_shapes[next_forward_chunk_id] is None:
                    self._input_obj_shapes[next_forward_chunk_id] = comm.recv_obj_meta()
                if k == (num_microsteps - 1) or gpc.is_pipeline_first_stage():
                    input_shape = None
                else:
                    input_shape = self._input_obj_shapes[next_forward_chunk_id]

            # Don't send tensor downstream if on last stage.
            if gpc.is_pipeline_last_stage():
                output_obj = None

            assert output_obj is None or output_obj.dtype == self.dtype

            # Send and receive tensors as appropriate (send tensors computed
            # in this iteration; receive tensors for next iteration).
            if k != (num_warmup_microsteps - 1) or not receive_extra_backward:
                # Normal warm-up communication process, or no need to prepare backward input for the 1F1B stage
                input_obj = comm.send_forward_recv_forward(
                    output_obj,
                    input_shape,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            else:
                # Receive output_obj_grad for next backward, if receive_extra_backward is True.
                if self._communication_overlap:
                    # In this case, we should handle forward and backward communication separately, consistent with the
                    # overlap version of the 1F1B stage
                    input_obj = comm.send_forward_recv_forward(
                        output_obj,
                        input_shape,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                    output_obj_grad = comm.send_backward_recv_backward(
                        None,  # nothing to send
                        self._output_obj_shapes[self._num_chunks - 1],
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                    self._output_obj_grads[self._num_chunks - 1].append(output_obj_grad)
                else:
                    # In this case, we should handle forward and backward communication together, consistent with the
                    # non-overlap version of the 1F1B stage
                    input_obj, output_obj_grad = comm.send_forward_backward_recv_forward_backward(
                        output_obj,
                        None,  # no backward grad to send
                        input_shape,
                        self._output_obj_shapes[self._num_chunks - 1],
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                    self._output_obj_grads[self._num_chunks - 1].append(output_obj_grad)

            self._input_objs[next_forward_chunk_id].append(input_obj)

    def _run_1f1b_loop_with_overlap(
        self,
        engine: Engine,
        num_warmup_microsteps: int,
        num_1f1b_micropairs: int,
        all_warmup_microsteps: bool = False,
    ) -> None:
        """
        Run the 1F1B loop with overlap.

        The 1F1B loop with overlap consists of the following steps:
        1. Perform the forward pass.
        2. Check if the backward input is ready.
        3. Send the forward output and receive the forward input for the next iteration.
        4. Perform the backward pass.
        5. Check if the forward input is ready.
        6. Send the backward output and receive the backward input for the next iteration.

        Args:
            engine (Engine): The engine to run the 1F1B loop.
            num_warmup_microsteps (int): The number of warm-up microsteps.
            num_1f1b_micropairs (int): The number of 1F1B micropairs.
            all_warmup_microsteps (bool, optional): Whether to run all warm-up microsteps. Default is False.
        """

        backward_async_communicator = None

        # Run 1F1B in steady state.
        for k in range(num_1f1b_micropairs):
            forward_microstep_id = k + num_warmup_microsteps
            backward_microstep_id = k
            forward_chunk_id = self._get_chunk_by_microbatch(forward_microstep_id)
            backward_chunk_id = self._get_chunk_by_microbatch(backward_microstep_id, backward=True)

            # 1. Forward pass.
            output_obj = self._forward_step(engine, forward_chunk_id)

            # 2. Check if the backward input is ready.
            if backward_async_communicator is not None:
                output_obj_grad = backward_async_communicator.wait_and_receive()

                if backward_async_communicator.need_receive:
                    self._output_obj_grads[backward_chunk_id].append(output_obj_grad)

            # 3. Send the forward outputs and receive the forward inputs from the previous rank.

            # Check if it is the last model chunk of the last pipeline stage, no need to send forward output.
            gpc.set_virtual_pipeline_parallel_rank(forward_chunk_id)
            if gpc.is_pipeline_last_stage():
                output_obj = None

            # Check if it needs to receive the results from the previous rank.
            next_forward_chunk_id = self._get_chunk_by_microbatch(forward_microstep_id + 1)
            with switch_virtual_pipeline_parallel_rank(next_forward_chunk_id):
                if gpc.is_pipeline_first_stage() or k == num_1f1b_micropairs - 1:
                    input_obj_shape = None
                else:
                    input_obj_shape = self._input_obj_shapes[next_forward_chunk_id]

            assert output_obj is None or output_obj.dtype == self.dtype
            forward_async_communicator = comm.AsynCommunicator(
                output_obj,
                input_obj_shape,
                self.dtype,
                self.scatter_gather_tensors,
                forward=True,
            )
            forward_async_communicator.start()

            # 5. Backward pass.

            input_obj_grad = self._backward_step(engine, backward_chunk_id, backward_microstep_id)

            input_obj = forward_async_communicator.wait_and_receive()
            if forward_async_communicator.need_receive:
                self._input_objs[next_forward_chunk_id].append(input_obj)

            # 6. Send the backward output and receive the backward input for the next iteration.
            gpc.set_virtual_pipeline_parallel_rank(backward_chunk_id)
            if gpc.is_pipeline_first_stage():
                input_obj_grad = None

            next_backward_chunk_id = self._get_chunk_by_microbatch(backward_microstep_id + 1, backward=True)
            with switch_virtual_pipeline_parallel_rank(next_backward_chunk_id):
                if gpc.is_pipeline_last_stage():
                    output_obj_shape = None
                else:
                    output_obj_shape = self._output_obj_shapes[next_backward_chunk_id]

            backward_async_communicator = comm.AsynCommunicator(
                input_obj_grad,
                output_obj_shape,
                self.dtype,
                self.scatter_gather_tensors,
                forward=False,
            )
            backward_async_communicator.start()

        if all_warmup_microsteps:
            if not gpc.is_pipeline_last_stage():
                self._output_obj_grads[self._num_chunks - 1].append(
                    comm.recv_backward(
                        self._output_obj_shapes[self._num_chunks - 1],
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                )
            else:
                self._output_obj_grads[self._num_chunks - 1].append(None)
        else:
            output_obj_grad = backward_async_communicator.wait_and_receive()
            if backward_async_communicator.need_receive:
                backward_chunk_id = self._get_chunk_by_microbatch(num_1f1b_micropairs, backward=True)
                self._output_obj_grads[backward_chunk_id].append(output_obj_grad)

    def _run_1f1b_loop_without_overlap(
        self,
        engine: Engine,
        num_warmup_microsteps: int,
        num_1f1b_micropairs: int,
        all_warmup_microsteps: bool = False,
    ) -> None:
        """
        Run the 1F1B loop without overlap.

        The 1F1B loop without overlap consists of the following steps:
        1. Perform the forward pass.
        2. Perform the backward pass.
        3. Send the forward output of this iteration to the next stage, and send the backward output of this iteration
           to the previous stage, and receive the forward and backward inputs for the next iteration.

        Args:
            engine (Engine): The engine to use for computation.
            num_warmup_microsteps (int): The number of warmup microsteps.
            num_1f1b_micropairs (int): The number of 1F1B micro-pairs.
            all_warmup_microsteps (bool, optional): Whether to run all warmup microsteps. Defaults to False.
        """
        for k in range(num_1f1b_micropairs):
            # Forward pass.
            forward_microstep_id = k + num_warmup_microsteps
            forward_chunk_id = self._get_chunk_by_microbatch(forward_microstep_id)
            output_obj = self._forward_step(engine, forward_chunk_id)

            # Backward pass.
            backward_microstep_id = k
            backward_chunk_id = self._get_chunk_by_microbatch(backward_microstep_id, backward=True)
            input_obj_grad = self._backward_step(engine, backward_chunk_id, backward_microstep_id)

            # Send output_obj and input_obj_grad, receive input_obj
            # and output_obj_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set obj to None.
            gpc.set_virtual_pipeline_parallel_rank(forward_chunk_id)
            if gpc.is_pipeline_last_stage():
                output_obj = None

            gpc.set_virtual_pipeline_parallel_rank(backward_chunk_id)
            if gpc.is_pipeline_first_stage():
                input_obj_grad = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            next_forward_chunk_id = self._get_chunk_by_microbatch(forward_microstep_id + 1)
            with switch_virtual_pipeline_parallel_rank(next_forward_chunk_id):
                if gpc.is_pipeline_first_stage() or k == num_1f1b_micropairs - 1:
                    recv_prev = False
                else:
                    recv_prev = True

            next_backward_chunk_id = self._get_chunk_by_microbatch(backward_microstep_id + 1, backward=True)
            with switch_virtual_pipeline_parallel_rank(next_backward_chunk_id):
                if gpc.is_pipeline_last_stage():
                    recv_next = False
                else:
                    recv_next = True

            input_shape = self._input_obj_shapes[next_forward_chunk_id] if recv_prev else None
            output_shape = self._output_obj_shapes[next_backward_chunk_id] if recv_next else None

            # Communicate objs.
            assert output_obj is None or output_obj.dtype == self.dtype
            input_obj, output_obj_grad = comm.send_forward_backward_recv_forward_backward(
                output_obj,
                input_obj_grad,
                input_shape,
                output_shape,
                dtype=self.dtype,
                scatter_gather_tensors=self.scatter_gather_tensors,
            )

            # Put input_obj and output_obj_grad in data structures in the
            # right location.
            if recv_prev:
                self._input_objs[next_forward_chunk_id].append(input_obj)
            if recv_next:
                self._output_obj_grads[next_backward_chunk_id].append(output_obj_grad)

        # receive necessary data for next cooldown loop
        if all_warmup_microsteps:
            if not gpc.is_pipeline_last_stage():
                self._output_obj_grads[self._num_chunks - 1].append(
                    comm.recv_backward(
                        self._output_obj_shapes[self._num_chunks - 1],
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                )
            else:
                self._output_obj_grads[self._num_chunks - 1].append(None)

    def _run_cooldown_loop(self, engine: Engine, num_microsteps: int, num_1f1b_micropairs: int) -> None:
        """
        Run the cooldown loop.

        The cooldown loop consists of the following steps:
        1. Perform the backward step.
        2. Send the backward output to the next stage and receive inputs for next backward.

        Args:
            engine (Engine): The engine to use for computation.
            num_microsteps (int): The total number of microsteps.
            num_1f1b_micropairs (int): The number of 1F1B micro-pairs.
        """
        for k in range(num_1f1b_micropairs, num_microsteps):
            chunk_id = self._get_chunk_by_microbatch(k, backward=True)

            input_obj_grad = self._backward_step(engine, chunk_id, k)

            next_backward_chunk_id = self._get_chunk_by_microbatch(k + 1, backward=True)

            if k != (num_microsteps - 1) and not (
                gpc.is_pipeline_last_stage(ignore_virtual=True) and next_backward_chunk_id == (self._num_chunks - 1)
            ):
                output_shape = self._output_obj_shapes[next_backward_chunk_id]
            else:
                output_shape = None

            self._output_obj_grads[next_backward_chunk_id].append(
                comm.send_backward_recv_backward(
                    input_obj_grad,
                    output_shape,
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            )

    def _forward_only_step(self, engine: Engine):
        num_microsteps = self.num_microbatches * self._num_chunks
        num_warmup_microsteps = num_microsteps

        self._run_warmup_loop(
            engine,
            num_microsteps,
            num_warmup_microsteps,
            receive_extra_backward=False,
            forward_only=True,
        )

    def _forward_backward_step(self, engine: Engine):
        # Compute number of warmup and remaining microbatches.
        all_warmup_microsteps = False
        num_microsteps = self.num_microbatches * self._num_chunks

        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if self.num_microbatches == self._pp_size:
            num_warmup_steps = num_microsteps
            all_warmup_microsteps = True
        else:
            num_warmup_steps = (self._pp_size - self._pp_rank - 1) * 2
            num_warmup_steps += (self._num_chunks - 1) * self._pp_size
            num_warmup_steps = min(num_warmup_steps, num_microsteps)
        num_1f1b_micropairs = num_microsteps - num_warmup_steps

        # We usually need to prepare an extra backward data for the 1F1B stage when the WarmUp stage ends,
        # because the 1F1B stage typically performs one forward and backward pass together,
        # except in the following cases:
        receive_extra_backward = not (
            all_warmup_microsteps  # Only warmup microsteps
            or gpc.is_pipeline_last_stage(ignore_virtual=True)  # The rank is the last pipeline stage
        )

        # 1. Warmup
        self._run_warmup_loop(
            engine,
            num_microsteps,
            num_warmup_steps,
            receive_extra_backward=receive_extra_backward,
        )

        # 2. 1F1B
        self._run_1f1b_loop(
            engine,
            num_warmup_steps,
            num_1f1b_micropairs=num_1f1b_micropairs,
            all_warmup_microsteps=all_warmup_microsteps,
        )

        # 3. Cooldown
        self._run_cooldown_loop(engine, num_microsteps, num_1f1b_micropairs=num_1f1b_micropairs)

    @llm_timeout(func_name="interleaved_forward_backward_step")
    def forward_backward_step(self, engine, data_iter, forward_only=False, return_loss=True, return_output_label=True):
        """Run interleaved 1F1B schedule (model split into model chunks), with
        communication between pipeline stages as needed.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                Whether run forward step only. Default is false. If true, no backward will be run.
            return_loss (bool, optional): Whether returns the loss value. Default is true.
            return_output_label (bool, optional): If False, the output and label won't be returned.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss, moe_loss), loss and label could be None.
                The loss would be returned only in the last stage. And the moe_loss is accumulated from all stages.
        """
        assert (
            forward_only or return_loss
        ), "The argument 'return_loss' has to be True when 'forward_only' is False, but got False."

        gpc.set_virtual_pipeline_parallel_rank(0)

        self.load_batch(engine, data_iter)

        if return_loss and gpc.is_pipeline_last_stage(ignore_virtual=True):
            self._accum_loss = torch.zeros(1, device=get_current_device())
        self._accum_moe_loss = torch.zeros(1, device=get_current_device())

        if return_output_label:
            self._return_tensors = []

        if forward_only:
            self._forward_only_step(engine)
        else:
            self._forward_backward_step(engine)
        print(f"afrer _forward_backward_step {gpc.get_global_rank()}", flush=True)

        if return_output_label and len(self._return_tensors) > 0:
            output, label = pack_return_tensors(self._return_tensors)
        else:
            output, label = (None, None)
        
        print(f"afrer _forward_backward_step1 {gpc.get_global_rank()}", flush=True)

        if hasattr(gpc.config.model, "num_experts") and gpc.config.model.num_experts > 1:
            dist.all_reduce(self._accum_moe_loss, group=gpc.get_group(ParallelMode.PIPELINE))
        accum_moe_loss = self._accum_moe_loss

        accum_loss = self._accum_loss
        if accum_loss is not None:
            print(f"accum_loss {gpc.get_global_rank()}", flush=True)
            accum_loss += self._accum_moe_loss

        self._clear_state()
        print(f"afrer _forward_backward_step2 {gpc.get_global_rank()}", flush=True)

        # Compatible for non-moe
        if hasattr(gpc.config.model, "num_experts"):
            return output, label, accum_loss, accum_moe_loss
        else:
            return output, label, accum_loss


class ZeroBubblePipelineVShapeScheduler(InterleavedPipelineScheduler):
    """
    ZB-V Scheduler.
    """

    def __init__(
        self,
        num_microbatches: int,
        num_chunks: int,
        dtype: torch.dtype = torch.float,
        data_process_func: Callable = None,
        tensor_shape: Union[torch.Size, List[int], Tuple[int]] = None,
        scatter_gather_tensors: bool = False,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
    ):
        """A helper schedule class for pipeline parallelism running environment.
        It uses ZB-V strategy. Other properties are similar as
        :class:`NonPipelineSchedule`.

        Args:
            num_microbatches (int): The number of microbatches.
            num_chunks (int): The number of model chunks.
            dtype (torch.dtype, optional): The data type of the tensors. Default is torch.float.
            data_process_func (Callable, optional):
                The preprocessing function which receives a batch of data, and it will be executed in `load_batch`.
            tensor_shape (torch.Size, optional): Specified shape in pipeline communication.
            scatter_gather_tensors (bool, optional):
                If set to `True`, communication will be reduced over pipeline when using 1D tensor parallelization.
            scheduler_hooks (List[SchedulerHook], optional): List of scheduler hooks. Default is None.
        """
        
        print("debugg ZeroBubblePipelineVShapeScheduler", flush=True)
        
        assert (
            isinstance(num_chunks, int) and num_chunks == 2
        ), f"expect num_chunks to be an integer and equal to 2 for ZBV, but got {num_chunks}."
        
        assert (
            num_microbatches >= 2 * gpc.get_world_size(ParallelMode.PIPELINE)
        ), f"For ZBV, num_microbatches must be greater than or equal to twice pp size."
        
        super().__init__(
            num_microbatches,
            num_chunks=num_chunks,
            dtype=dtype,
            data_process_func=data_process_func,
            tensor_shape=tensor_shape,
            scatter_gather_tensors=scatter_gather_tensors,
            scheduler_hooks=scheduler_hooks,
        )
                
        del self._run_1f1b_loop
                
        self._special_chunk0_forward = True
        WeightGradStore.set_pp_mode("ZBV")
        gpc.v_shape = True
        
        self._micro_step = [0, 0]
        # self.map_input_output = {}
        
    def _clear_state(self) -> None:
        super()._clear_state()
        self._special_chunk0_forward = True
        
        self._micro_step = [0, 0]
        # self.map_input_output = {}
        
    
    def _backward_step(self, engine, input_obj, output_obj, output_obj_grad, skip_grad_sync=True, moe_loss=None):
        """
        Backward step through the passed-in output tensor. If it is the last stage, the
        output_obj_grad is None, otherwise it is the gradients with respect to stage's output tensor.
        Returns the gradients with respect to the input tensor (None if first stage).
        This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            input_obj (Union[torch.Tensor, List[torch.Tensor]]): Input tensor for this stage.
            output_obj (Union[torch.Tensor, List[torch.Tensor]]): Output tensor for this stage.
            output_obj_grad (Union[torch.Tensor, List[torch.Tensor]]): Gradient of output tensor for this stage.
            skip_grad_sync (bool): Whether skip grad sync or not.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Gradient of input tensor.
        """

        # Retain the grad on the input_obj.
        if input_obj is not None:
            assert input_obj.requires_grad == True, f"{gpc.get_global_rank()}"
            if isinstance(input_obj, torch.Tensor):
                input_obj.retain_grad()
            else:
                for in_tensor in input_obj:
                    if in_tensor is not None:
                        in_tensor.retain_grad()

        # Backward pass.

        # Only the last microbatch does syncing grad.
        self._call_hooks("before_backward", output_obj, output_obj_grad)
        with switch_optimizer_grad_sync_skip_mode(engine.optimizer, skip_grad_sync):
            if moe_loss is None or moe_loss.item() == 0.0:
                if output_obj_grad is None:
                    engine.backward(output_obj)
                else:
                    try:
                        engine.backward_by_grad(output_obj, output_obj_grad)
                    except Exception as e:
                        print("rank:", gpc.get_global_rank(), flush=True)
                        
                        raise e
                    
                        
            else:
                if output_obj_grad is None:
                    engine.backward(output_obj + moe_loss)
                else:
                    # scale the latent loss
                    moe_loss = moe_loss * engine.optimizer.loss_scale
                    # we perform chain rule here by projecting the grad to the direction of
                    # [output_obj_grad, 1], Because moe_loss have no relation with subsequent
                    # layer, we set it to None (will be ragarded as 1).
                    engine.backward_by_grad([output_obj, moe_loss], [output_obj_grad, None])

        # Collect the grad of the input_obj.
        input_obj_grad = None
        if input_obj is not None:
            assert input_obj.grad is not None, f"{gpc.get_global_rank()}"
            if isinstance(input_obj, torch.Tensor):
                input_obj_grad = input_obj.grad
            else:
                input_obj_grad = []
                for in_tensor in input_obj:
                    input_obj_grad.append(in_tensor.grad)
        else:
            assert gpc.is_pipeline_first_stage(), f"{gpc.get_global_rank()}"

        return input_obj_grad
    
    def _schedule_backward(self, engine, chunk_id, skip_grad_sync=True):
        """
        Backward step for passed-in model. If it is the last stage, the input tensor
        is obtained from the previous forward step, otherwise the passed-in input_obj is used.
        Returns input tensor gradient. This is a helper function and can be ignored by users.

        Args:
            engine (colossalai.engine.Engine): Colossalai engine for training and inference.
            chunk_id (int): The id of model chunks.
            step_id (int): The current step id.

        Returns:
            Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]: input tensor gradient.
        """
        gpc.set_virtual_pipeline_parallel_rank(chunk_id)

        if gpc.is_pipeline_last_stage() and len(self._output_obj_grads[chunk_id]) == 0:
            self._output_obj_grads[chunk_id].append(None)
        
        assert len(self._input_objs[chunk_id]) == len(self._output_objs[chunk_id]), f"{gpc.get_global_rank()} {chunk_id} {len(self._input_objs[chunk_id])} {len(self._output_objs[chunk_id])}"

        input_obj = self._input_objs[chunk_id].pop(0)
        output_obj = self._output_objs[chunk_id].pop(0)
        output_obj_grad = self._output_obj_grads[chunk_id].pop(0)
        moe_loss = self._moe_losses[chunk_id].pop(0)
        
        if input_obj is not None:
            # assert self.map_input_output[id(input_obj)] == id(output_obj), f"{gpc.get_global_rank()}"
            assert input_obj.requires_grad == True
        
        if not gpc.is_pipeline_last_stage():
            assert output_obj_grad is not None
        if not gpc.is_pipeline_first_stage():
            assert input_obj is not None
        # import pdb; pdb.set_trace()   
        input_obj_grad = self._backward_step(engine, input_obj, output_obj, output_obj_grad, skip_grad_sync, moe_loss)
        if not gpc.is_pipeline_first_stage():
            assert input_obj_grad is not None, f"{gpc.get_global_rank()}"
        
        WeightGradStore.flush()

        return input_obj_grad
    
    def _schedule_1f1b_F(self, engine, chunk_id):
        self._micro_step[chunk_id] += 1
        print(f"self._micro_step {gpc.get_global_rank()} chunk{chunk_id}: {self._micro_step[chunk_id]}", flush=True)
        output_obj = self._forward_step(engine, chunk_id)
        # if self._input_objs[chunk_id][-1] is not None:
            # self.map_input_output[id(self._input_objs[chunk_id][-1])] = id(output_obj)
                    
        object_send_next = None
        object_send_prev = None   
        recv_next_shape = None
        recv_prev_shape = None
             
        if chunk_id == 1:
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                object_send_prev = output_obj
                recv_prev_shape=self._output_obj_shapes[chunk_id]
        else:
            if gpc.is_last_rank(ParallelMode.PIPELINE):
                # For last rank, chunk0 output does not need to be sent but is directly used for chunk1;
                # input_obj = output_obj.clone().detach()
                # input_obj.requires_grad = True
                input_obj = output_obj.clone().detach()
                input_obj.requires_grad_()
                assert input_obj.is_leaf
                # assert self.dtype == output_obj.dtype
                # input_obj = torch.empty_like(output_obj, requires_grad=True, device=get_current_device(), dtype=self.dtype)
                # input_obj.copy_(output_obj.clone().detach())
                self._input_objs[1].append(input_obj)
            else:
                object_send_next = output_obj
                recv_next_shape = self._output_obj_shapes[chunk_id]
                
        # chunk1 send output prev, recv output_grad prev  
        # chunk0 send output next, recv output_grad next 
        tensor_recv_prev, tensor_recv_next = comm.fused_send_recv_tensor(
            object_send_next=object_send_next,
            object_send_prev=object_send_prev,
            recv_next_shape=recv_next_shape,
            recv_prev_shape=recv_prev_shape,
            dtype=self.dtype,
            scatter_gather_tensors=self.scatter_gather_tensors,
        )
        print(f"after 1F fused_send_recv_tensor {gpc.get_global_rank()} step{self._micro_step[chunk_id]} chunk{chunk_id}", flush=True)
        
        if chunk_id == 0 and gpc.is_last_rank(ParallelMode.PIPELINE):
            assert tensor_recv_prev is None and tensor_recv_next is None
        else:
            output_obj_grad = tensor_recv_prev if tensor_recv_prev is not None else tensor_recv_next
            self._output_obj_grads[chunk_id].append(output_obj_grad)
    
    def _schedule_1f1b_B_W(self, engine, chunk_id, next_unit_chunk_id):
        
        print(f"1f1b_B_W before _schedule_backward {gpc.get_global_rank()} step{self._micro_step[chunk_id]} chunk{chunk_id}", flush=True)
        # import pdb; pdb.set_trace()
        # 1B
        input_obj_grad = self._schedule_backward(engine, chunk_id)
        
        print(f"1f1b_B_W after _schedule_backward {gpc.get_global_rank()} step{self._micro_step[chunk_id]} chunk{chunk_id}", flush=True)
        
        object_send_next = None
        object_send_prev = None
        recv_next_shape = None
        recv_prev_shape = None
        
        if chunk_id == 1:
            if gpc.is_last_rank(ParallelMode.PIPELINE):
                # For last rank, chunk1 input_grad does not need to be sent but is directly used for chunk0.
                # output_grad = input_obj_grad.clone().detach()
                self._output_obj_grads[0].append(input_obj_grad)
            else:
                object_send_next = input_obj_grad
                
            if next_unit_chunk_id == 1:
                if gpc.is_last_rank(ParallelMode.PIPELINE):
                    assert False, "The last pp rank can never have two consecutive unit1 of the same chunk."
                recv_next_shape = self._input_obj_shapes[next_unit_chunk_id]
            else:
                if not (gpc.is_first_rank(ParallelMode.PIPELINE) or self._special_chunk0_forward):
                    recv_prev_shape = self._input_obj_shapes[next_unit_chunk_id]                    
        else:
            assert next_unit_chunk_id != 0, "There will never be two consecutive chunk0 unit1."
            
            if not gpc.is_first_rank(ParallelMode.PIPELINE):
                object_send_prev = input_obj_grad
            
            if not gpc.is_last_rank(ParallelMode.PIPELINE):
                recv_next_shape = self._input_obj_shapes[next_unit_chunk_id]
                
                
        # chunk1 send input_grad next, chunk0 send input_grad prev
        # if next_unit_chunk_id == 1, recv input_obj next
        # if next_unit_chunk_id == 0, recv input_obj prev    
        input_shape = recv_next_shape if recv_next_shape is not None else recv_prev_shape
        tensor_to_send = object_send_next if object_send_next is not None else object_send_prev
        send_next = True if object_send_next is not None else False
        recv_next = True if recv_next_shape is not None else False
        
        async_communicator = comm.AsynCommunicator(
            tensor_to_send=tensor_to_send,
            recv_shape=input_shape,
            dtype=self.dtype,
            scatter_gather_tensors=self.scatter_gather_tensors,
            send_next=send_next,
            recv_next=recv_next,
        )
        async_communicator.start()
        
        print(f"after 1f1b_B_W async_communicator start {gpc.get_global_rank()} step{self._micro_step[chunk_id]} chunk{chunk_id}", flush=True)
        
        # 1W
        WeightGradStore.pop()
        self._call_hooks("after_backward", input_obj_grad)
        
        input_obj = async_communicator.wait_and_receive()
        print(f"after 1f1b_B_W async_communicator wait_and_receive {gpc.get_global_rank()} step{self._micro_step[chunk_id]} chunk{chunk_id}", flush=True)
        assert input_obj is None or isinstance(input_obj, torch.Tensor)
            
        # for the special case, input_obj has already been received and appended at the end of warmup.
        if next_unit_chunk_id == 0 and self._special_chunk0_forward:
            assert input_obj is None
            self._special_chunk0_forward = False
        else:
            if not (next_unit_chunk_id == 1 and gpc.is_last_rank(ParallelMode.PIPELINE)):
                if not (next_unit_chunk_id == 0 and gpc.is_first_rank(ParallelMode.PIPELINE)):
                    assert input_obj is not None, f"{gpc.get_global_rank()} chunk{chunk_id} next_unit_chunk_id{next_unit_chunk_id} receive none input BW"
                else:
                    assert input_obj is None
                self._input_objs[next_unit_chunk_id].append(input_obj)
        
    
    def _1f1b_unit_1(self, engine, chunk_id, next_unit_chunk_id):
        """
        unit1 consists of: 1F + 1B + 1W, all are chunk0 or chunk1
        """
        # 1F
        # import pdb; pdb.set_trace()
        self._schedule_1f1b_F(engine, chunk_id)
        print(f"after _schedule_1f1b_F {gpc.get_global_rank()} step{self._micro_step[chunk_id]} chunk{chunk_id}", flush=True)
        
        # 1B + 1W
        # import pdb; pdb.set_trace()
        self._schedule_1f1b_B_W(engine, chunk_id, next_unit_chunk_id)
        print(f"after _schedule_1f1b_B_W {gpc.get_global_rank()} step{self._micro_step[chunk_id]} chunk{chunk_id}", flush=True)
    
    def _1f1b_unit_2(self, engine, chunk_id):
        """
        unit2 consists of: chunk1 (1F + 1B + 1W) + chunk0 (1B + 1W)
        """
        assert chunk_id == 1
        assert not gpc.is_last_rank(ParallelMode.PIPELINE)
        
        # 1F (chunk1)
        self._schedule_1f1b_F(engine, chunk_id)
        
        
        # 1B + 1W (chunk1)
        input_obj_grad = self._schedule_backward(engine, chunk_id)
            
        # chunk1 send input_grad next, chunk0 recv output_grad next
        async_communicator = comm.AsynCommunicator(
            tensor_to_send=input_obj_grad,
            recv_shape=self._output_obj_shapes[1 - chunk_id],
            dtype=self.dtype,
            scatter_gather_tensors=self.scatter_gather_tensors,
            send_next=True,
            recv_next=True,
        )
        async_communicator.start()
        
        WeightGradStore.pop()
        self._call_hooks("after_backward", input_obj_grad)
        
        output_obj_grad = async_communicator.wait_and_receive()
        assert isinstance(output_obj_grad, torch.Tensor)
        self._output_obj_grads[1 - chunk_id].append(output_obj_grad)
        
        
        # 1B + 1W (chunk0)
        self._schedule_1f1b_B_W(engine, 1 - chunk_id, chunk_id)
        
    
    def _schedule_warmup_F(self, engine, chunk_id, forward_only=False):
        self._micro_step[chunk_id] += 1
        print(f"self._micro_step {gpc.get_global_rank()} chunk{chunk_id}: {self._micro_step[chunk_id]}", flush=True)
        print(f"before _schedule_warmup_F _forward_step {gpc.get_global_rank()}", flush=True)
        # import pdb; pdb.set_trace()
        output_obj = self._forward_step(engine, chunk_id)
        print(f"after _schedule_warmup_F _forward_step {gpc.get_global_rank()}", flush=True)
        # if self._input_objs[chunk_id][-1] is not None:
        #     self.map_input_output[id(self._input_objs[chunk_id][-1])] = id(output_obj)
            
        if forward_only:
            assert False
            # when forward-only, no need to save tensors for a backward pass
            self._input_objs[chunk_id].pop()
            self._output_objs[chunk_id].pop()
            self._moe_losses[chunk_id].pop()
            
        if not gpc.is_pipeline_last_stage():
            if isinstance(output_obj, torch.Tensor):
                self._output_obj_shapes[chunk_id] = output_obj.shape
            else:
                self._output_obj_shapes[chunk_id] = [out_tensor.shape for out_tensor in output_obj]

            if self._send_tensor_shape_flags[chunk_id]:
                assert False
                comm.send_obj_meta(output_obj)
                self._send_tensor_shape_flags[chunk_id] = False  # send only once for each chunk.
        
        if not gpc.is_pipeline_first_stage() and self._input_obj_shapes[chunk_id] is None:
            assert False
            self._input_obj_shapes[chunk_id] = comm.recv_obj_meta()
            
        assert output_obj is None or output_obj.dtype == self.dtype
        
        return output_obj
                
    def _run_warmup_loop(
        self,
        engine: Engine,
        num_warmup_microsteps: int,
        forward_only: bool = False,
    ) -> None:
        """
        Run the warm-up loop and prepare data for the 1F1B stage.

        During the warm-up process, for each execution, it first performs a forward computation,
        and then sends the computation result to the next stage.
        It also receives data for the next forward computation.
        Since the input for the first forward computation is not considered initially,
        it needs to receive data once at the beginning.

        After the warm-up is completed, we need to prepare data for the 1F1B stage.
        The data preparation process should be consistent with the communication method of the 1F1B stage.

        Args:
            engine (Engine): The engine to run the warm-up loop.
            num_microsteps (int): The total number of microsteps.
            num_warmup_microsteps (int): The number of warm-up microsteps.
            receive_extra_backward (bool, optional): Whether to receive extra backward input for the 1F1B stage.
                                                     Default is False.
            forward_only (bool, optional): Whether to only perform forward pass. Default is False.
        """
        
        # For each rank, the warmup stage will be divided into two sub-phases for scheduling.
        num_warmup_microsteps_phase_1 = min(self.num_microbatches, (self._pp_size - self._pp_rank) * 2 - 1)
        num_warmup_microsteps_phase_2 = num_warmup_microsteps - num_warmup_microsteps_phase_1
        
        print(f"global_pp: {gpc.get_global_rank()}, {gpc.get_local_rank(ParallelMode.PIPELINE)}", flush=True)
        print(f"{gpc.get_global_rank()} num_warmup_microsteps_phase_1: {num_warmup_microsteps_phase_1}, num_warmup_microsteps_phase_2: {num_warmup_microsteps_phase_2}", flush=True)
        
        
        if gpc.is_first_rank(ParallelMode.PIPELINE):
            assert num_warmup_microsteps_phase_2 == 0
        if gpc.is_last_rank(ParallelMode.PIPELINE):
            assert num_warmup_microsteps_phase_1 == 1
        
        # get first forward input
        chunk_id = 0
        if not gpc.is_pipeline_first_stage():
            if self._input_obj_shapes[chunk_id] is None:
                self._input_obj_shapes[chunk_id] = comm.recv_obj_meta()
            self._input_objs[chunk_id].append(
                comm.recv_forward(
                    self._input_obj_shapes[chunk_id],
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            )
        else:
            self._input_objs[chunk_id].append(None)
        if not gpc.is_first_rank(ParallelMode.PIPELINE):
            assert self._input_objs[chunk_id][-1] is not None, f"{gpc.get_global_rank()} chunk{chunk_id} receive none input warmup before"
        # Phase1 will only do chunk0 forward
        for micro_step in range(num_warmup_microsteps_phase_1):
            print(f"warmup phase1 micro_step {gpc.get_global_rank()}: {micro_step}", flush=True)
            # forward
            print(f"before _schedule_warmup_F_1 {gpc.get_global_rank()}", flush=True)
            output_obj = self._schedule_warmup_F(engine, chunk_id, forward_only)
            print(f"afrer _schedule_warmup_F_1 {gpc.get_global_rank()}", flush=True)
            if micro_step != num_warmup_microsteps_phase_1 - 1:
                # if micro_step == num_warmup_microsteps_phase_1 - 2:
                #     output_obj = None
                if gpc.is_pipeline_first_stage():
                    input_shape = None
                else:
                    input_shape = self._input_obj_shapes[chunk_id]
                # chunk0 send next, chunk0 recv prev    
                self._input_objs[chunk_id].append(
                    comm.send_forward_recv_forward(
                        output_obj,
                        input_shape,
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                )
                print(f"afrer phase1 send_forward_recv_forward {gpc.get_global_rank()}", flush=True)
                if not gpc.is_pipeline_first_stage():
                    assert self._input_objs[chunk_id][-1] is not None, f"{gpc.get_global_rank()} chunk{chunk_id} receive none input warmup1"
            else:
                if not gpc.is_last_rank(ParallelMode.PIPELINE):  
                    # chunk0 send next, chunk1 recv next
                    _, input_obj = comm.fused_send_recv_tensor(
                        object_send_next=output_obj,
                        recv_next_shape=self._input_obj_shapes[1],
                        dtype=self.dtype,
                        scatter_gather_tensors=self.scatter_gather_tensors,
                    )
                    self._input_objs[1].append(input_obj)
                else:
                    # For last rank, chunk0 output does not need to be sent but is directly used for chunk1
                    # input_obj = output_obj.clone().detach()
                    # input_obj.requires_grad = True
                    input_obj = output_obj.clone().detach()
                    input_obj.requires_grad_()
                    assert input_obj.is_leaf
                    # assert self.dtype == output_obj.dtype
                    # input_obj = torch.empty_like(output_obj, requires_grad=True, device=get_current_device(), dtype=self.dtype)
                    # input_obj.copy_(output_obj.clone().detach())
                    self._input_objs[1].append(input_obj)
                assert self._input_objs[1][-1] is not None, f"{gpc.get_global_rank()} chunk{chunk_id} receive none input warmup2"
                print(f"afrer phase1 fused_send_recv_tensor {gpc.get_global_rank()}", flush=True)

        print(f"afrer phase1 {gpc.get_global_rank()}", flush=True)
        # Phase2 will execute chunk1 and chunk0 forward alternately
        for micro_step in range(num_warmup_microsteps_phase_2):
            
            print(f"warmup phase2 micro_step {gpc.get_global_rank()}: {micro_step}", flush=True)
            chunk_id = 1 - chunk_id
            next_chunk_id = 1 - chunk_id
            
            output_obj = self._schedule_warmup_F(engine, chunk_id, forward_only)
            print(f"afrer _schedule_warmup_F_2 {gpc.get_global_rank()}", flush=True)
            
            
            object_send_next = None
            object_send_prev = None   
            recv_next_shape = None
            recv_prev_shape = None
            
            if chunk_id == 1:
                assert micro_step < num_warmup_microsteps_phase_2 - 1
                # if not (gpc.is_last_rank(ParallelMode.PIPELINE) and micro_step == 0):
                object_send_prev = output_obj
                recv_prev_shape = self._input_obj_shapes[next_chunk_id]
            else:
                if not gpc.is_last_rank(ParallelMode.PIPELINE):  
                    object_send_next = output_obj
                    recv_next_shape = self._input_obj_shapes[next_chunk_id]
            
            # chunk1 send output prev, chunk0 recv input prev 
            # chunk0 send output next, chunk1 recv input next      
            # import pdb; pdb.set_trace()  
            tensor_recv_prev, tensor_recv_next = comm.fused_send_recv_tensor(
                object_send_next=object_send_next,
                object_send_prev=object_send_prev,
                recv_next_shape=recv_next_shape,
                recv_prev_shape=recv_prev_shape,
                dtype=self.dtype,
                scatter_gather_tensors=self.scatter_gather_tensors,
            )
            
            # For last rank, chunk0 output does not need to be sent but is directly used for chunk1
            if chunk_id == 0 and gpc.is_last_rank(ParallelMode.PIPELINE):
                input_obj = output_obj.clone().detach()
                input_obj.requires_grad_()
                assert input_obj.is_leaf
                # assert self.dtype == output_obj.dtype
                # input_obj = torch.empty_like(output_obj, requires_grad=True, device=get_current_device(), dtype=self.dtype)
                # input_obj.copy_(output_obj.clone().detach())
                # input_obj = output_obj.clone().detach()
                # input_obj.requires_grad = True
            else:
                input_obj = tensor_recv_prev if tensor_recv_prev is not None else tensor_recv_next
                
            self._input_objs[next_chunk_id].append(input_obj)
            if not gpc.is_pipeline_first_stage():
                assert self._input_objs[next_chunk_id][-1] is not None, f"{gpc.get_global_rank()} chunk{chunk_id} next_chunk_id{next_chunk_id} receive none input warmup3"
        print(f"afrer phase2 {gpc.get_global_rank()}", flush=True)        
        # The last chunk0 sending needs to be received immediately to avoid conflicts with gradient communication
        if not gpc.is_first_rank(ParallelMode.PIPELINE):
            self._input_objs[0].append(
                comm.recv_forward(
                    self._input_obj_shapes[0],
                    dtype=self.dtype,
                    scatter_gather_tensors=self.scatter_gather_tensors,
                )
            )
            assert self._input_objs[0][-1] is not None, f"{gpc.get_global_rank()}"
        else:
            self._input_objs[0].append(None)
                    
    def _run_1f1b_loop(self,
        engine: Engine,
        num_1f1b_units: int,
    ) -> None:
        """
        1F1B unit schedule: 
        stage1: (pp_size + 1 + pp_rank + 2 * (micro_num - 2 * pp_size)) * unit1
        stage2: (pp_size - 1 - pp_rank) * unit2
        stage3: 1 * special chunk1 unit1

        Args:
            engine (Engine): The engine to use for computation.
            num_1f1b_units (int): The number of 1F1B units.
        """
        # unit schedule
        num_units_stage1 = 2 * self.num_microbatches - 3 * self._pp_size + 1 + self._pp_rank
        num_units_stage2 = self._pp_size - 1 - self._pp_rank
        assert num_units_stage1 + num_units_stage2 + 1 == num_1f1b_units
        
        # chunk schedule: stage1 + stage2 + stage1
        # stage1: chunk1
        # stage2: chunk0 and chunk1 alternately
        stage1_length = self._pp_size - self._pp_rank
        stage2_length = 2 * self._pp_rank + 1 + 2 * (self.num_microbatches - 2 * self._pp_size )
        assert stage1_length * 2 + stage2_length == num_1f1b_units
        stage2_list = [i for i in range(stage1_length, stage1_length + stage2_length)]
        chunk0_units = [stage2_list[i] for i in range(len(stage2_list)) if i % 2 == 0]
        
        # unit stage1
        for unit_step in range(num_units_stage1):
            print(f"unit_step: {unit_step}, {gpc.get_global_rank()}", flush=True)
            if unit_step in chunk0_units:
                chunk_id = 0
            else:
                chunk_id = 1
            
            if unit_step + 1 in chunk0_units:
                next_unit_chunk_id = 0
            else:
                next_unit_chunk_id = 1
            
            # import pdb; pdb.set_trace()    
            self._1f1b_unit_1(engine, chunk_id, next_unit_chunk_id)
        
        print(f"after stage1 {gpc.get_global_rank()}", flush=True)
        # unit stage2
        for unit_step in range(num_units_stage2):
            assert unit_step + num_units_stage1 not in chunk0_units
            self._1f1b_unit_2(engine, 1)
        print(f"after stage2 {gpc.get_global_rank()}", flush=True)
            
        # unit stage3
        assert num_1f1b_units - 1 not in chunk0_units
        self._schedule_1f1b_F(engine, 1)
        input_obj_grad = self._schedule_backward(engine, 1)
        print(f"after stage3 {gpc.get_global_rank()}", flush=True)
        if gpc.is_last_rank(ParallelMode.PIPELINE):
            # For last rank, chunk1 input_grad does not need to be sent but is directly used for chunk0.
            # output_grad = input_obj_grad.clone().detach()
            self._output_obj_grads[0].append(input_obj_grad)
            tensor_to_send = None
            recv_shape = None
        else:
            tensor_to_send = input_obj_grad
            recv_shape = self._output_obj_shapes[0]

        # chunk1 send input_grad next, chunk0 recv output_grad next
        async_communicator = comm.AsynCommunicator(
            tensor_to_send=tensor_to_send,
            recv_shape=recv_shape,
            dtype=self.dtype,
            scatter_gather_tensors=self.scatter_gather_tensors,
            send_next=True,
            recv_next=True,
        )
        async_communicator.start()
        
        print(f"after stage3_1 {gpc.get_global_rank()}", flush=True)
        
        WeightGradStore.pop()
        self._call_hooks("after_backward", input_obj_grad)
        
        print(f"after stage3_2 {gpc.get_global_rank()}", flush=True)
        
        output_obj_grad = async_communicator.wait_and_receive()
        print(f"after stage3_3 {gpc.get_global_rank()}", flush=True)
        assert output_obj_grad is None or isinstance(output_obj_grad, torch.Tensor)
        if not gpc.is_last_rank(ParallelMode.PIPELINE):
            self._output_obj_grads[0].append(output_obj_grad)
        print(f"after stage3_4 {gpc.get_global_rank()}", flush=True)
        
    def _run_cooldown_loop(self, engine):
        """
        Cooldown unit schedule: 
        Unit: 1B + 1W
        Schedule unit chunk0 and unit chunk1 alternatively
        Each pp rank has pp_size chunk0, but only pp_rank chunk1
        """
        chunk0_length = self._pp_size
        chunk1_length = self._pp_rank
        num_cooldown_units = chunk0_length + chunk1_length
        total_list = [i for i in range(chunk1_length * 2)]
        chunk1_units = [total_list[i] for i in range(chunk1_length * 2) if i % 2 != 0]
        
        
        cool_down = [0, 0]
        
        for unit_step in range(num_cooldown_units):
            if unit_step in chunk1_units:
                chunk_id = 1
            else:
                chunk_id = 0
            
            cool_down[chunk_id] += 1
            print(f"cool_down {gpc.get_global_rank()} chunk{chunk_id}: {cool_down[chunk_id]}", flush=True)

            if unit_step + 1 in chunk1_units:
                next_unit_chunk_id = 1
            else:
                next_unit_chunk_id = 0
            
            skip_grad_sync = True if unit_step == num_cooldown_units - 1 else False
            if skip_grad_sync:
                assert chunk_id == 0
            print(f"before _schedule_backward {gpc.get_global_rank()} chunk{chunk_id}", flush=True)
            input_obj_grad = self._schedule_backward(engine, chunk_id, skip_grad_sync)
            print(f"after _schedule_backward {gpc.get_global_rank()} chunk{chunk_id}", flush=True)

            object_send_next = None
            object_send_prev = None
            recv_next_shape = None
            recv_prev_shape = None
            
            if chunk_id == 1:
                assert not gpc.is_first_rank(ParallelMode.PIPELINE)
                if gpc.is_last_rank(ParallelMode.PIPELINE):
                    # For last rank, chunk1 input_grad does not need to be sent but is directly used for chunk0.
                    # output_grad = input_obj_grad.clone().detach()
                    self._output_obj_grads[0].append(input_obj_grad)
                else:
                    object_send_next = input_obj_grad
                    # next unit should be chunk0
                    recv_next_shape = self._output_obj_shapes[0]                    
            else:                
                if not gpc.is_first_rank(ParallelMode.PIPELINE):
                    object_send_prev = input_obj_grad

                if unit_step != num_cooldown_units - 1:
                    if next_unit_chunk_id == 1:
                        assert not gpc.is_first_rank(ParallelMode.PIPELINE)
                        recv_prev_shape = self._output_obj_shapes[next_unit_chunk_id]
                    else:
                        assert not gpc.is_last_rank(ParallelMode.PIPELINE)
                        recv_next_shape = self._output_obj_shapes[next_unit_chunk_id]
                    
                    
            # chunk1 send input_grad next, chunk0 send input_grad prev
            # if next_unit_chunk_id == 1, recv output_grad prev
            # if next_unit_chunk_id == 0, recv output_grad next    
            input_shape = recv_next_shape if recv_next_shape is not None else recv_prev_shape
            tensor_to_send = object_send_next if object_send_next is not None else object_send_prev
            send_next = True if object_send_next is not None else False
            recv_next = True if recv_next_shape is not None else False
            
            async_communicator = comm.AsynCommunicator(
                tensor_to_send=tensor_to_send,
                recv_shape=input_shape,
                dtype=self.dtype,
                scatter_gather_tensors=self.scatter_gather_tensors,
                send_next=send_next,
                recv_next=recv_next,
            )
            async_communicator.start()
            print(f"after async_communicator start {gpc.get_global_rank()} chunk{chunk_id}", flush=True)
            
            # 1W
            WeightGradStore.pop()
            self._call_hooks("after_backward", input_obj_grad)
            
            output_obj_grad = async_communicator.wait_and_receive()
            print(f"after async_communicator wait_and_receive {gpc.get_global_rank()} chunk{chunk_id}", flush=True)
            assert output_obj_grad is None or isinstance(output_obj_grad, torch.Tensor)
            
            # if not(next_unit_chunk_id == 0 and gpc.is_last_rank(ParallelMode.PIPELINE)):
            if output_obj_grad is not None:
                self._output_obj_grads[next_unit_chunk_id].append(output_obj_grad)
            
    def _forward_only_step(self, engine: Engine):
        num_warmup_steps = self.num_microbatches * self._num_chunks

        self._run_warmup_loop(
            engine,
            num_warmup_steps,
            forward_only=True,
        )                                                             
                
    def _forward_backward_step(self, engine: Engine):
        assert self.num_microbatches > self._pp_size
        
        # Compute number of warmup microbatches.
        num_warmup_steps = self._pp_size * 2 - 1
        
        # Compute number of 1F1B unit.
        num_1f1b_units = 2 * self.num_microbatches - num_warmup_steps
        # self._input_objs[1].append(torch.rand((1, 2048, 4096), device=get_current_device(), dtype=torch.bfloat16))
        
        # 1. Warmup
        self._run_warmup_loop(
            engine,
            num_warmup_steps,
        )
        print(f"afrer _run_warmup_loop {gpc.get_global_rank()}", flush=True)
        
        # 2. 1F1B
        self._run_1f1b_loop(
            engine,
            num_1f1b_units,
        )
        print(f"afrer _run_1f1b_loop {gpc.get_global_rank()}", flush=True)
        
        # 3. cooldown
        self._run_cooldown_loop(engine)
        print(f"afrer cooldown {gpc.get_global_rank()}", flush=True)
        # import pdb; pdb.set_trace()
        assert len(self._input_objs[0]) == 0 and len(self._input_objs[1]) == 0, f"{gpc.get_global_rank()}"
        assert len(self._output_objs[0]) == 0 and len(self._output_objs[1]) == 0, f"{gpc.get_global_rank()}"
        assert len(self._output_obj_grads[0]) == 0 and len(self._output_obj_grads[1]) == 0, f"{gpc.get_global_rank()}"
        
        assert WeightGradStore.size() == 0
    