from typing import List, Tuple
import torch


def byteir〇flash_attn_fwd〡shape(q: List[int], k: List[int], v: List[int], dropout_p: float, softmax_scale: float, casual: bool, return_softmax: bool) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int], List[int], List[int]]:
    batch_size = q[0]
    seqlen_q = q[1]
    num_heads = q[2]
    seqlen_k = k[1]
    softmax_lse = [batch_size, num_heads, seqlen_q]
    softmax_return = [batch_size, num_heads, seqlen_q, seqlen_k]
    rng_shape = [2]
    return q, q, k, v, q, softmax_lse, softmax_return, rng_shape


def byteir〇flash_attn_fwd〡dtype(q_rank_dtype: Tuple[int, int], k_rank_dtype: Tuple[int, int], v_rank_dtype: Tuple[int, int], dropout_p: float, softmax_scale: float, casual: bool, return_softmax: bool) -> Tuple[int, int, int, int, int, int, int, int]:
    q_rank, q_dtype = q_rank_dtype
    return q_dtype, q_dtype, q_dtype, q_dtype, q_dtype, torch.float32, q_dtype, torch.int64


def byteir〇flash_attn_fwd〡has_value_semantics() -> None:
    return


def byteir〇flash_attn_bwd〡shape(dout: List[int], q: List[int], k: List[int], v: List[int], out: List[int], softmax_lse: List[int], dropout_p: float, softmax_scale: float, casual: bool, rng: List[int]) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
    batch_size = q[0]
    seqlen_q = q[1]
    num_heads = q[2]
    seqlen_q_rounded = ((seqlen_q + 127) // 128) * 128
    head_size = q[3]
    head_size_rounded = ((head_size + 31) // 32) * 32
    d_softmax = [batch_size, num_heads, seqlen_q_rounded]
    dq_accum = [batch_size, num_heads, seqlen_q_rounded, head_size_rounded]
    return q, k, v, d_softmax, dq_accum


def byteir〇flash_attn_bwd〡dtype(dout_rank_dtype: Tuple[int, int], q_rank_dtype: Tuple[int, int], k_rank_dtype: Tuple[int, int], v_rank_dtype: Tuple[int, int], out_rank_dtype: Tuple[int, int], softmax_lse_rank_dtype: Tuple[int, int], dropout_p: float, softmax_scale: float, casual: bool, rng_rank_dtype: Tuple[int, int]) -> Tuple[int, int, int, int, int]:
    dq_rank, dq_dtype = q_rank_dtype
    dk_rank, dk_dtype = k_rank_dtype
    dv_rank, dv_dtype = v_rank_dtype
    return dq_dtype, dk_dtype, dv_dtype, torch.float32, torch.float32


def byteir〇flash_attn_bwd〡has_value_semantics() -> None:
    return

byteir_extra_library = {"byteir.flash_attn_fwd": [
                            byteir〇flash_attn_fwd〡shape,
                            byteir〇flash_attn_fwd〡dtype,
                            byteir〇flash_attn_fwd〡has_value_semantics],
                        "byteir.flash_attn_bwd": [
                            byteir〇flash_attn_bwd〡shape,
                            byteir〇flash_attn_bwd〡dtype,
                            byteir〇flash_attn_bwd〡has_value_semantics]
                        }
