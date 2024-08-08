from typing import List, Union

from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.parallel.shard import pipeline_parallel_sharding_wrapper
from internlm.model.registry import hf_config_initializer, model_initializer
from internlm.model.utils import convert_hf_config
from internlm.utils.common import get_current_device
from internlm.utils.utils import ModelType


def create_model(model_type) -> Union[nn.Module, List[nn.Module]]:

    if model_type == ModelType.HF.name:
        extra_kwargs = {"return_dict": False, "attn_implementation": "flash_attention_2"}
        config = hf_config_initializer.get_module(module_name=model_type)(**extra_kwargs)
        convert_hf_config(config)

    kwargs = dict(gpc.config.model)

    num_layers = kwargs.pop("num_layers")
    num_chunks = kwargs.pop("num_chunks", 1)

    # TODO: fix use_flash_attn parameter config
    kwargs.pop("use_flash_attn", False)
    kwargs.pop("apply_post_layer_norm")
    kwargs.pop("embed_split_hidden", True)

    kwargs["checkpoint"] = float(kwargs.get("checkpoint", False))
    kwargs["device"] = get_current_device()

    if "checkpoint_tp_no_comm" in kwargs:
        kwargs.pop("checkpoint_tp_no_comm")

    model_buidler = model_initializer.get_module(module_name=model_type)

    if not gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
        if model_type == ModelType.HF.name:
            model = model_buidler(config).to(kwargs["device"])
        else:
            kwargs["first"] = kwargs["last"] = True
            kwargs["start_layer_idx"] = 0
            kwargs["num_layers"] = num_layers
            model = model_buidler(**kwargs).to(kwargs["device"])
        setattr(model, "first_layer", 0)
        setattr(model, "last_layer", num_layers)
    else:
        model = pipeline_parallel_sharding_wrapper(num_layers, num_chunks, model_buidler, **kwargs)

    return model
