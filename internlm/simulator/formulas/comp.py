from internlm.simulator.common import AlgoType, CostType
from internlm.simulator.profiler.perf_comm  import get_cal_cost, get_fa_cost


def get_linear_cost(complexity):
    return get_cal_cost(CostType.LINEAR, complexity)  # 转换成ms小数点保留两位


def get_atten_cost_polynomial(complexity):
    return get_cal_cost(CostType.LINEAR, complexity)


def get_atten_cost_predict(micro_bsz, seq_len, hidden_dim, q_head, kv_head, is_fwd):
    """_summary_

    Args:
        micro_bsz (int): b
        seq_len (int): seqlen, 注意这里是完整的seqlen
        hidden_dim (int): 原始的head_dim
        num_heads (int): 原始的num_heads
        sp_tp (int): sp for isp, tp for msp/fsp

    Returns:
        int: latency of fa, unit is second.
    """
    predict = get_fa_cost(
        micro_bsz=micro_bsz,
        seqlen=seq_len,
        hidden_size=hidden_dim,
        q_head=q_head,
        kv_head=kv_head,
        dtype=2,
        is_fwd=is_fwd,
    )
    return predict


class TransformerComputation:
    def __init__(
        self,
        a,
        a_kv,
        b,
        s,
        h,
        vocab_size,
        sp_scale,
        dtype_size,
        mlp_ratio,
        multiple_of,
        use_fa=True,
        cost_data=None,
        ckpt=0,
    ):
        self.a = a
        self.a_kv = a_kv
        self.b = b  # Batch size
        self.s = s  # Sequence length
        self.h = h  # Hidden size
        self.sp_scale = sp_scale
        self.qkv_computation = 0
        self.qkt_computation = 0
        # self.score_v_computation = 0
        # self.post_attention_linear = 0
        # self.first_linear = 0
        # self.second_linear = 0
        # self.logits_computation = 0
        # self.attention_computation = 0
        # self.flash_attention_computation = 0
        # self.mlp_computation = 0
        self.vocab_size = vocab_size
        self.dtype_size = dtype_size
        self.mlp_ratio = mlp_ratio
        self.multiple_of = multiple_of
        self.mlp_hidden_size = self.multiple_of * (
            (int(self.h * self.mlp_ratio) + self.multiple_of - 1) // self.multiple_of
        )
        self.ckpt = ckpt
        self.use_fa = use_fa

    def _compute_embedding(self, scale):
        """
        the head computation is the same as embedding computation.
        msp and fsp share the same computation.

        scale: the scale factor. when the algo is isp, the scale is one; else the scale is self.sp_scale.
        """
        volumn = self.dtype_size * self.b * self.s * self.vocab_size * self.h / scale
        try:
            latency = get_linear_cost(volumn)
        except Exception:
            #import pdb; pdb.set_trace() 
            return get_linear_cost(9895604649984)
        return latency

    def _compute_linears(self):
        """
        compute the latency for linears in one transformer layer, such as wqkv, wo, mlp
        """

        # wqkv
        # ISP: (b, s/sp, h) * (h, 3h)
        # MSP or FSP: (b, s, h) * (h, 3h/sp)
        qkv_volumn = 3 * self.dtype_size * self.b * self.s * self.h * self.h / self.sp_scale
        qkv_latency = get_linear_cost(qkv_volumn)

        # wo
        # ISP: (b, s/sp, h) * (h, h)
        # MSP or FSP: (b, s, h/sp) * (h/sp, h)
        wo_volumn = self.dtype_size * self.b * self.s * self.h * self.h / self.sp_scale
        wo_latency = get_linear_cost(wo_volumn)

        # mlp w1
        # ISP: (b, s/sp, h) * (h, mlp_h)
        # MSP or FSP: (b, s, h) * (h, mlp_h/sp)
        w1_volumn = self.dtype_size * self.b * self.s * self.h * self.mlp_hidden_size / self.sp_scale
        w1_latency = get_linear_cost(w1_volumn)

        # mlp w2
        # ISP: (b, s/sp, h) * (h, mlp_h)
        # MSP or FSP: (b, s, h/sp) * (h/sp, mlp_h)
        w2_volumn = self.dtype_size * self.b * self.s * self.h * self.mlp_hidden_size / self.sp_scale
        w2_latency = get_linear_cost(w2_volumn)

        # mlp w3
        # ISP: (b, s/sp, mlp_h) * (mlp_h, h)
        # MSP or FSP: (b, s, mlp_h/sp) * (mlp_h/sp, h)
        w3_volumn = self.dtype_size * self.b * self.s * self.h * self.mlp_hidden_size / self.sp_scale
        w3_latency = get_linear_cost(w3_volumn)

        total_latency = qkv_latency + wo_latency + w1_latency + w2_latency + w3_latency

        return total_latency

    def _compute_attn(self, is_fwd):
        """
        compute the latency for attention in one transformer layer
        """
        if self.use_fa:
            # 由于我们目前不支持搜索ring attn，所以这里我们sp/tp只切head数量
            a = self.a // self.sp_scale
            a_kv = self.a_kv // self.sp_scale

            total_latency = get_atten_cost_predict(self.b, self.s, self.h, a, a_kv, is_fwd)
        else:
            # QK^T matrix multiplication
            # (b, s, h/sp) * (b, s, h/sp)^T
            qkt_volume = self.dtype_size * self.b * self.s * self.s * self.h / self.sp_scale
            qkt_latency = get_atten_cost_polynomial(qkt_volume)

            # Score dot V
            # (b, s, s) * (b, s, h/sp)
            score_v_volume = self.dtype_size * self.b * self.s * self.s * self.h / self.sp_scale
            score_v_latency = get_atten_cost_polynomial(score_v_volume)

            total_latency = qkt_latency + score_v_latency

        return total_latency

    def _computation(self, embedding_scale):
        # TODO: the following computation exclude norm computation
        """
        ckpt: activation checkpoint {0 or 1}

        the computation latency for each transformer layer

        compu(msp) = compu(forward) + compu(backward)

        compu(backward) = 2 * compu(forward)

        compu(forward) = (compu(linear, (wqkv, wo, mlp)) + compu(attn)) * (ckpt + 1)
        """

        # compute the latency for embedding and head
        embedding_latency = self._compute_embedding(embedding_scale)
        head_latency = embedding_latency

        # compute the latency for linears
        linears_latency = self._compute_linears() * (self.ckpt + 1) + self._compute_linears() * 2

        # compute the latency for attention
        attn_latency = self._compute_attn(is_fwd=True) * (self.ckpt + 1) + self._compute_attn(is_fwd=False)

        # the computation for each transformer layer
        # transformer_latency = linears_latency + attn_latency

        return linears_latency, attn_latency

    def total_computation(self, algo_type):
        if algo_type == AlgoType.ISP:
            # return self.total_computation_isp()
            return self._computation(1.0)
        else:
            return self._computation(self.sp_scale)


# Example usage
# Assuming values for b (batch size), s (sequence length), h (hidden size), num_layers, and vocab_size
# b, s, h, num_layers, vocab_size = 1, 16384, 4096, 32, 10000
# transformer_comp = TransformerComputation(b, s, h,num_layers,vocab_size)
