# llm2asic/ir/ops.py
"""统一算子清单（OP SET）与形状推导规则。

对应设计文档 top.md §3.3。
"""

from enum import Enum


class Op(str, Enum):
    """项目内部算子集合（OP SET）。"""

    # A. 张量/线性算子（GEMM 族）
    LINEAR = "linear"        # 全连接 + 可选 bias
    MATMUL = "matmul"        # 矩阵乘
    GEMV = "gemv"            # 矩阵-向量乘
    GEMM = "gemm"            # 通用矩阵乘
    BMM = "bmm"              # 批量矩阵乘

    # B. Attention / 序列算子（LLM 专用）
    ROPE = "rope"            # 旋转位置编码
    ATTENTION = "attention"  # 缩放点积注意力（含 causal mask）
    KV_STORE = "kv_store"
    KV_LOAD = "kv_load"
    CONCAT = "concat"
    UNFLATTEN = "unflatten"
    TRANSPOSE = "transpose"
    RESHAPE = "reshape"
    PERMUTE = "permute"

    # C. 归一化 / 激活 / 非线性
    RMSNORM = "rmsnorm"
    LAYERNORM = "layernorm"
    SOFTMAX = "softmax"
    SILU = "silu"
    GELU = "gelu"
    RELU = "relu"
    ADD = "add"              # 残差
    MUL = "mul"
    SUB = "sub"
    DIV = "div"

    # D. 数据搬运 / 嵌入
    EMBEDDING = "embedding"
    CLONE = "clone"
    COPY = "copy"
    CONSTANT = "constant"


# ---------------------------------------------------------------------------
# OP SET 聚合（用于校验 / 文档对应 top.md §3.3）
# ---------------------------------------------------------------------------

OP_SET: dict[Op, str] = {
    Op.LINEAR: "GEMM族: 全连接线性层",
    Op.MATMUL: "GEMM族: 矩阵乘",
    Op.GEMV: "GEMM族: 矩阵-向量乘(decode)",
    Op.GEMM: "GEMM族: 通用矩阵乘",
    Op.BMM: "GEMM族: 批量矩阵乘",
    Op.ROPE: "Attention: 旋转位置编码",
    Op.ATTENTION: "Attention: 缩放点积注意力",
    Op.KV_STORE: "Attention: KV缓存写入",
    Op.KV_LOAD: "Attention: KV缓存读取",
    Op.CONCAT: "Attention: 拼接",
    Op.UNFLATTEN: "Attention: 多头重塑",
    Op.TRANSPOSE: "数值: 转置",
    Op.RESHAPE: "数值: 重塑",
    Op.PERMUTE: "数值: 置换",
    Op.RMSNORM: "归一化: RMSNorm",
    Op.LAYERNORM: "归一化: LayerNorm",
    Op.SOFTMAX: "激活: softmax",
    Op.SILU: "激活: silu",
    Op.GELU: "激活: gelu",
    Op.RELU: "激活: relu",
    Op.ADD: "激活: 残差加法",
    Op.MUL: "激活: 逐元素乘法",
    Op.SUB: "激活: 逐元素减法",
    Op.DIV: "激活: 逐元素除法",
    Op.EMBEDDING: "嵌入: token查表",
    Op.CLONE: "数据: 复制(分支)",
    Op.COPY: "数据: 复制",
    Op.CONSTANT: "数据: 常量",
}


def is_supported_op(name: str) -> bool:
    """判断算子名是否为内部和集中的合法算子。"""
    try:
        Op(name)
        return True
    except ValueError:
        return False
