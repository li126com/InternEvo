from internlm.simulator.common import AlgoType

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc


# 所有公式计算都变成无状态的,计算结果完全由外部传入的参数决定，内部不进行诸如切pp这样的操作
def get_isp_memory_threshold(
    dtype_size: int,
    micro_batch_size: int,
    sequence_length: int,
    hidden_dim: int,
    use_fa: int,
    head_num: int,
    layer_num: int,
    activation_ckpt: int,
    sp_size: int,
):
    """
    Args:
        dtype_size (int): bf16=2, fp32=4
        micro_batch_size (int):
        sequence_length (int):
        hidden_dim (int):
        use_fa (int): 0 or 1
        head_num (int):
        layer_num (int):
        activation_ckpt (int): 0 or 1

    Returns:
        int: activation memory usage.
    """
    # TODO: ht mark pp情况下，rank0的激活值会累积pp个micro_bsz，所以这里是不是还得再乘一个pp_size？
    # TODO: wgt mark 应该不需要，每个pp拿到L/pp个layer，又最多保存pp个micro_num的激活，
    # rank0相当于还是L份layer的激活

    if activation_ckpt:
        layer_num = 0

    """ (0) dropout input: 2bsh
        (1) attention: 2bsh (qkv input) + 3*2bsh(attn input) + 2bsh(attn_out_padded) + 2bsh(out project input)-> 12bsh
        (2) dropout input: 2bsh
        (3) MLP: 2 * (1 + 8/3 + 8/3 + 8/3 +8/3)* bsh =  70 * bsh / 3
                w1_o = self.w1(x)   #  8/3
                w2_o = self.w2(x)   #  8/3
                w3_in = Silu(w1_o, w2_o) #  8/3 + 8/3
                out = self.w3(w3_in) #  8/3
        total: 16bsh + 70 * bsh / 3 = 118 * bsh /3
    """
    activation = (
        dtype_size
        * micro_batch_size
        * sequence_length
        * hidden_dim
        * (118 / 3 + (1 - use_fa) * (5 * head_num * sequence_length / hidden_dim))
        / sp_size
    ) * layer_num
    return activation


def get_msp_memory_threshold(
    dtype_size: int,
    micro_batch_size: int,
    sequence_length: int,
    hidden_dim: int,
    use_fa: int,
    head_num: int,
    layer_num: int,
    activation_ckpt: int,
    sp_size: int,
):
    if activation_ckpt:
        layer_num = 0

    activation = (
        dtype_size
        * micro_batch_size
        * sequence_length
        * hidden_dim
        * (
            12 / 3 + ((118 - 12) / 3) / sp_size + (1 - use_fa) * (5 * head_num * sequence_length / hidden_dim / sp_size)
        )  # TODO: check
    ) * layer_num
    return activation


def get_fsp_memory_threshold(
    dtype_size: int,
    micro_batch_size: int,
    sequence_length: int,
    hidden_dim: int,
    use_fa: int,
    head_num: int,
    layer_num: int,
    activation_ckpt: int,
    sp_size: int,
):
    if activation_ckpt:
        layer_num = 0

    activation = (
        dtype_size
        * micro_batch_size
        * sequence_length
        * hidden_dim
        * (118 / 3 + (1 - use_fa) * (5 * head_num * sequence_length / hidden_dim))
        / sp_size
    ) * layer_num  # 显存阈值根据pp0来计算，需要micro_num >= pp，stage_0需要保存 pp 份才成立
    return activation


# tp=1,sp=1
# seql_len=512, hidden_dim 4096, no tp,sp
# embed shape: torch.Size([1, 4096, 512]) 1
# block shape: torch.Size([4096, 512])
# head shape: torch.Size([4096, 103168])

# tp=4,sp=1
# seql_len=512, hidden_dim 4096
# embed shape: torch.Size([1, 4096, 512])
# block shape: torch.Size([4096, 512])
# head shape: torch.Size([4096, 25792])

# tp=4,sp=4
# embed shape: torch.Size([1, 1024, 512])
# block shape: torch.Size([1024, 512])
# head shape: torch.Size([4096, 25792])

# WP不省激活，因此不受wp影响
# 这里只计算一层的激活，不受pp影响


# embedding output
def get_embedding_output_mm(micro_bsz, seq_len, hidden_dim, sp, algo, dtype_size):
    # [b, hidden_dim, seql_len]
    # sp的world_size是从tp的pg中获得的
    sp_worldsize = gpc.get_world_size(ParallelMode.TENSOR)
    # assert sp == sp_worldsize, f"sp={sp}, sp_world_size:{sp_worldsize}"
    assert sp_worldsize == sp, f"sp={sp}, sp_world_size:{sp_worldsize}, algo: {algo}"
    return dtype_size * micro_bsz * seq_len * hidden_dim // sp


# block output
def get_block_output_mm(micro_bsz, seq_len, hidden_dim, sp, dtype_size):
    # [hidden_dim, packed_length]
    sp_worldsize = gpc.get_world_size(ParallelMode.TENSOR)
    assert sp == sp_worldsize, f"sp={sp}, sp_world_size:{sp_worldsize}"
    return dtype_size * micro_bsz * seq_len * hidden_dim // sp


# norm output
def get_norm_output_mm(micro_bsz, seq_len, hidden_dim, sp, dtype_size):
    # [hidden_dim, packed_length]
    sp_worldsize = gpc.get_world_size(ParallelMode.TENSOR)
    assert sp == sp_worldsize, f"sp={sp}, sp_world_size:{sp_worldsize}"
    return 4 * micro_bsz * seq_len * hidden_dim // sp  # norm的输出是fp32的


# head output
def get_head_output_mm(micro_bsz, seq_len, vocab_size, dtype_size):
    # [seq_len, vocab_size]
    return micro_bsz * dtype_size * seq_len * vocab_size // gpc.get_world_size(ParallelMode.TENSOR)


# head input
def get_head_input_mm(micro_bsz, seq_len, hidden_dim, dtype_size, tp_size, algo):
    # [seq_len, vocab_size]
    if algo in [AlgoType.ISP, AlgoType.FSP]:
        return micro_bsz * dtype_size * seq_len * hidden_dim // tp_size
    else:
        return 0


# rotary embedding sin/cos cache
def get_rotary_emb_sincos_cache_mm(seq_len, pp_size, hidden_dim, head_nums, layer_nums, dtype_size):
    # [sin,cos] * dtype_size * pp切后的layer_nums * 不切的seq_len * head_dim // 2
    return 2 * dtype_size * (layer_nums // pp_size) * seq_len * (hidden_dim // head_nums) // 2


def get_backward_mem_peak(seq_len, micro_bsz, dtype_size, vocab_size, tp_size, hidden_size):
    # 这个函数是峰值位置
    head_input_grad = 2 * dtype_size * seq_len * micro_bsz * hidden_size  # 512 MB (1份激活1份激活的梯度)
    reduce_scatter_grad = head_input_grad / tp_size  # 512 MB / 8
    head_weight_grad = dtype_size * hidden_size * vocab_size / tp_size  #  100.b MB
    return head_input_grad + reduce_scatter_grad + head_weight_grad


def get_memory_pool_mm(mlp_ratio, hidden_size, dtype_size):
    mlp_hidden_size = int(hidden_size * mlp_ratio)
    mlp_hidden_size = 256 * ((mlp_hidden_size + 256 - 1) // 256)
    module_Wqkv = 3 * hidden_size * hidden_size * dtype_size
    module_out_proj = hidden_size * hidden_size * dtype_size
    module_w1 = mlp_hidden_size * hidden_size * dtype_size
    module_w2 = mlp_hidden_size * hidden_size * dtype_size
    module_w3 = mlp_hidden_size * hidden_size * dtype_size
    prefetch_two_layers_weight = 2 * (module_Wqkv + module_out_proj + module_w1 + module_w2 + module_w3)

    return prefetch_two_layers_weight * 2  # all_gather + reduce_scatter approximately


def get_p2p_buffer_size(dtype_size, seq_len, sp_size, micro_bsz, hidden_dim):
    return dtype_size * (seq_len // sp_size) * micro_bsz * hidden_dim


def get_block_threshold(
    algo: AlgoType,
    **kwargs,
):
    """get_block_threshold 获得一层激活的显存占用
    注意:
    (1) seqlen一定是没有被sp切过的
    (2) 公式是基于fp16计算的, 所以传入的 dtype_size 要除以2
    Args:
        dtype_size (int): 数据元素大小, 单位B
        seq_len (int): 没有被切过的seq_len

    Returns:
        float : 一个layer的显存占用, 单位B
    """
    if algo == AlgoType.ISP:
        return get_isp_memory_threshold(**kwargs)
    elif algo == AlgoType.MSP:
        return get_msp_memory_threshold(**kwargs)
    elif algo == AlgoType.FSP:
        return get_fsp_memory_threshold(**kwargs)
    
    raise ValueError(f"unknow algo: {algo}")
