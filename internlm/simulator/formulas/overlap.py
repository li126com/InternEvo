from internlm.simulator.formulas.comm import TransformerCommunication
from internlm.simulator.formulas.comp import TransformerComputation
from internlm.simulator.common import get_model_config


# 1. dtype 加入复杂度
# 2. comm 没有乘以 laynum
# 3. atten 计算还没加
# 4. mmeory check
# 5. 集成simulator
class TransformerOverlapOneLayer:
    def __init__(
        self,
        micro_bsz,
        seq_len,
        vocab_size,
        dtype_size,
        sp_size,
        pp_size,
        world_size,
        ckpt,
        hidden_dim,
        num_head,
        num_kv_head,
        mlp_ratio,
        multiple_of,
    ):
        self.b = micro_bsz  # Batch size
        self.s = seq_len  # Sequence length
        self.vocab_size = vocab_size
        self.sp_scale = sp_size
        self.dtype_size = dtype_size
        self.world_size = world_size
        self.pp_size = pp_size

        self.h, self.a, self.a_kv, self.mlp_ratio, self.multiple_of = hidden_dim, num_head, num_kv_head, mlp_ratio, multiple_of

        self.ckpt = ckpt  # the activation checkpoint

    def _get_overlap(self, algo_type):
        # 一个transformer layer的通信时延 (forward + backward)
        comm_wp, comm_sp = TransformerCommunication(
            self.b,
            self.s,
            self.h,
            self.vocab_size,
            dtype_size=self.dtype_size,
            mlp_ratio=self.mlp_ratio,
            multiple_of=self.multiple_of,
            ckpt=self.ckpt,
        ).communication(algo_type)

        # 一个transformer layer的计算时延 (forward + backward)
        comp_wp, comp_attn = TransformerComputation(
            self.a,
            self.a_kv,
            self.b,
            self.s,
            self.h,
            self.vocab_size,
            dtype_size=self.dtype_size,
            mlp_ratio=self.mlp_ratio,
            multiple_of=self.multiple_of,
            sp_scale=self.sp_scale,
            ckpt=self.ckpt,
        ).total_computation(algo_type)

        return comm_wp, comm_sp, comp_wp, comp_attn
