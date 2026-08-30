## Module 3: 硬件架构生成 —— 详细设计实现分析

Module 3的核心使命是**将Module 2输出的量化后GraphIR，转化为可综合的硬件架构描述**。这是整个编译器中“软件到硬件”的关键转折点——将计算图映射为物理上的数据流硬件流水线。

---

### 整体工作流程概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   输入: 量化后GraphIR + ROM文件元数据                           │
│        包含: 计算图节点、量化权重信息、并行度参数                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: 计算图降维与算子映射 (Graph Lowering & Operator Mapping)            │
│  • 卷积 → im2col + 矩阵乘法 (GEMM) 降维                                     │
│  • 全连接 → 直接映射为矩阵向量乘法                                            │
│  • 激活函数 → 映射为阈值单元(Thresholding Unit)                              │
│  • 逐元素操作 → 映射为流式处理单元                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: 硬件原语选择与参数化 (Hardware Primitive Selection)                 │
│  • 矩阵向量乘单元(MVU): PE并行度 + SIMD宽度参数化                            │
│  • 滑动窗口单元(SWU): 卷积输入展开                                           │
│  • 阈值单元: 激活函数实现                                                    │
│  • FIFO缓冲: 层间数据流缓冲                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: 数据流网络构建 (Dataflow Network Construction)                     │
│  • 层间连接: AXI-Stream通道 + FIFO深度计算                                   │
│  • 流水线平衡: 各阶段吞吐量匹配                                              │
│  • 控制逻辑: 握手信号(valid/ready)生成                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 4: HLS/RTL代码生成 (Code Generation)                                   │
│  • 选择HLS路径或RTL路径                                                     │
│  • 生成各层C++ HLS代码或Verilog模块                                         │
│  • 嵌入权重ROM例化                                                          │
│  • 生成顶层缝合IP (Stitched IP)                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   输出: HLS C++代码 / RTL模块 + 缝合IP                       │
│        传递给Vivado/Vitis进行综合与实现                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 1: 计算图降维与算子映射

FINN的核心理念是将各类神经网络层**统一降维为矩阵向量乘法（GEMM）** ，然后用高度参数化的MVU（Matrix-Vector Unit）统一计算。

#### 1.1 卷积层降维：im2col + GEMM

对于卷积层，FINN采用经典的im2col方法将卷积降维为矩阵乘法：

**数学原理**：
- 输入特征图 `[H, W, C_in]` → im2col展开为 `[H_out * W_out, K_h * K_w * C_in]`
- 卷积核 `[C_out, C_in, K_h, K_w]` → 展开为 `[C_out, K_h * K_w * C_in]`
- 卷积运算 → 矩阵乘法：`[H_out*W_out, C_out] = [H_out*W_out, K_h*K_w*C_in] × [K_h*K_w*C_in, C_out]^T`

```python
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ConvLayerInfo:
    in_channels: int
    out_channels: int
    kernel_h: int
    kernel_w: int
    stride: int
    padding: int
    input_h: int
    input_w: int

def lower_conv_to_gemm(conv_info: ConvLayerInfo) -> dict:
    """
    将卷积层降维为GEMM参数
    
    Returns:
        {
            'matrix_a_rows': H_out * W_out,      # 输出像素数
            'matrix_a_cols': K_h * K_w * C_in,   # 每个位置的输入维度
            'matrix_b_rows': C_out,              # 输出通道数
            'matrix_b_cols': K_h * K_w * C_in,   # 每个卷积核的参数量
            'im2col_params': {...}               # 滑动窗口参数
        }
    """
    # 计算输出尺寸
    h_out = (conv_info.input_h + 2 * conv_info.padding - conv_info.kernel_h) // conv_info.stride + 1
    w_out = (conv_info.input_w + 2 * conv_info.padding - conv_info.kernel_w) // conv_info.stride + 1
    
    return {
        'matrix_a_rows': h_out * w_out,
        'matrix_a_cols': conv_info.kernel_h * conv_info.kernel_w * conv_info.in_channels,
        'matrix_b_rows': conv_info.out_channels,
        'matrix_b_cols': conv_info.kernel_h * conv_info.kernel_w * conv_info.in_channels,
        'im2col_params': {
            'input_h': conv_info.input_h,
            'input_w': conv_info.input_w,
            'kernel_h': conv_info.kernel_h,
            'kernel_w': conv_info.kernel_w,
            'stride': conv_info.stride,
            'padding': conv_info.padding,
            'h_out': h_out,
            'w_out': w_out
        }
    }
```

#### 1.2 算子到硬件原语的映射表

```python
from enum import Enum
from typing import Type

class HWPrimitive(Enum):
    MVU = "matrix_vector_unit"        # 矩阵向量乘法
    SWU = "sliding_window_unit"       # 滑动窗口(im2col)
    THRESHOLD = "threshold_unit"      # 阈值/激活函数
    FIFO = "fifo_buffer"              # FIFO缓冲
    VECTOR_OP = "vector_operation"    # 逐元素操作
    POOL = "pooling_unit"             # 池化
    CONCAT = "concat_unit"            # 拼接
    RESHAPE = "reshape_unit"          # 重塑

# 算子到硬件原语的映射
OP_TO_PRIMITIVE = {
    'convolution': HWPrimitive.SWU,       # SWU做im2col，然后送MVU
    'matmul': HWPrimitive.MVU,
    'linear': HWPrimitive.MVU,            # 全连接 = MVU
    'relu': HWPrimitive.THRESHOLD,
    'leaky_relu': HWPrimitive.THRESHOLD,
    'batch_norm': HWPrimitive.VECTOR_OP,  # 推理时可融合到前层，或独立实现
    'max_pool': HWPrimitive.POOL,
    'avg_pool': HWPrimitive.POOL,
    'concat': HWPrimitive.CONCAT,
    'reshape': HWPrimitive.RESHAPE,
}
```

---

### Step 2: 硬件原语设计

这是Module 3的核心——定义可参数化的硬件计算原语。

#### 2.1 MVU (Matrix-Vector Unit) —— 核心计算单元

MVU是整个加速器的**核心计算引擎**。它的两个关键参数是：

| 参数 | 含义 | 硬件对应 |
|------|------|---------|
| **PE (Processing Elements)** | 输出通道并行度 | 同时计算多少个输出通道 |
| **SIMD (Single-Instruction Multiple-Data)** | 输入通道并行度 | 每个PE内同时累加多少个输入 |

**MVU架构图**：

```
                          ┌─────────────────────────────────────────┐
                          │              MVU (顶层)                 │
                          │                                         │
            ┌─────────────┤  ┌─────────────────────────────────┐   │
            │             │  │         PE Array                │   │
            │  Input      │  │  ┌──────┐ ┌──────┐ ┌──────┐   │   │
            │  Vector     │  │  │ PE 0 │ │ PE 1 │ │ ...  │   │   │
            │  (SIMD      │  │  │SIMD=8│ │SIMD=8│ │SIMD=8│   │   │
            │   stream)   │  │  └──┬───┘ └──┬───┘ └──┬───┘   │   │
            │             │  │     │        │        │       │   │
            │             │  │     └────────┼────────┘       │   │
            │             │  │              ▼                │   │
            │             │  │      Adder Tree (PE级)        │   │
            │             │  │              ▼                │   │
            │             │  │      Threshold Unit           │   │
            │             │  └─────────────────────────────────┘   │
            │             │                                         │
            └─────────────┘  Output Vector (PE个输出通道)          │
                          └─────────────────────────────────────────┘
```

**MVU C++ HLS模板代码**：

```cpp
// finn-hlslib中的MVU模板 (简化版)
template<
    unsigned int SIMD,          // SIMD并行度 (输入通道并行)
    unsigned int PE,            // PE数量 (输出通道并行)
    unsigned int INPW,          // 输入位宽 (bit)
    unsigned int WMW,           // 权重位宽 (bit)
    unsigned int OUTW,          // 输出位宽 (bit)
    unsigned int MATRIX_DIM     // 矩阵维度 (输入特征数量)
>
void mvu(
    hls::stream<ap_uint<INPW>>& in,
    hls::stream<ap_uint<OUTW>>& out,
    // 权重作为常量嵌入 (internal_embedded模式)
    const ap_uint<WMW> weights[PE][MATRIX_DIM / SIMD][SIMD]
) {
    #pragma HLS pipeline II=1
    
    // PE级循环 - 每个PE计算一个输出通道
    for (int pe = 0; pe < PE; pe++) {
        #pragma HLS unroll
        
        ap_int<ACCU_WIDTH> acc = 0;
        
        // SIMD级循环 - 每个周期累加SIMD个输入
        for (int simd = 0; simd < MATRIX_DIM / SIMD; simd++) {
            #pragma HLS pipeline II=1
            
            ap_uint<SIMD * INPW> in_pack = in.read();
            ap_uint<SIMD * WMW> w_pack = weights[pe][simd];
            
            // 拆包并乘加
            for (int s = 0; s < SIMD; s++) {
                #pragma HLS unroll
                ap_int<INPW> in_val = in_pack(s * INPW + INPW - 1, s * INPW);
                ap_int<WMW> w_val = w_pack(s * WMW + WMW - 1, s * WMW);
                acc += in_val * w_val;
            }
        }
        
        // 阈值化 (ReLU等)
        ap_uint<OUTW> out_val = threshold(acc);
        out.write(out_val);
    }
}
```

**PE内部结构**（RTL实现视角）：

```verilog
// 单个PE的RTL实现 (简化)
module pe #(
    parameter SIMD = 8,
    parameter INPW = 4,
    parameter WMW = 4,
    parameter ACCU_WIDTH = 32
)(
    input clk,
    input rst,
    input [SIMD*INPW-1:0] in_pack,
    input [SIMD*WMW-1:0] weight_pack,
    output reg [ACCU_WIDTH-1:0] acc_out
);
    // SIMD个乘法器 + 加法树
    wire [INPW+WMW-1:0] products [0:SIMD-1];
    generate
        for (genvar s = 0; s < SIMD; s++) begin : gen_mul
            assign products[s] = in_pack[s*INPW +: INPW] * 
                                 weight_pack[s*WMW +: WMW];
        end
    endgenerate
    
    // 加法树 (log2(SIMD)级)
    wire [ACCU_WIDTH-1:0] sum;
    adder_tree #(.NUM_INPUTS(SIMD), .INPUT_WIDTH(INPW+WMW)) 
        adder_tree_inst (.inputs(products), .sum(sum));
    
    // 累加器
    always @(posedge clk) begin
        if (rst) acc_out <= 0;
        else acc_out <= acc_out + sum;
    end
endmodule
```

#### 2.2 SWU (Sliding Window Unit) —— 滑动窗口单元

SWU负责将输入特征图通过滑动窗口展开为矩阵向量乘法所需的输入向量。

```cpp
// 滑动窗口单元 (将卷积输入展开)
template<
    unsigned int IFM_CH,        // 输入通道数
    unsigned int KERNEL_DIM,    // 核尺寸 (K_h * K_w)
    unsigned int IFM_DIM,       // 输入特征图尺寸
    unsigned int OFM_DIM,       // 输出特征图尺寸
    unsigned int STRIDE,
    unsigned int INPW
>
void sliding_window_unit(
    hls::stream<ap_uint<INPW>>& in,
    hls::stream<ap_uint<IFM_CH * KERNEL_DIM * INPW>>& out
) {
    // 行缓冲 (Line Buffer) - 存储KERNEL_H行
    ap_uint<INPW> line_buffer[KERNEL_H][IFM_DIM * IFM_CH];
    #pragma HLS array_partition variable=line_buffer complete
    
    // 滑动窗口逻辑
    for (int row = 0; row < OFM_DIM; row++) {
        for (int col = 0; col < OFM_DIM; col++) {
            #pragma HLS pipeline II=1
            
            // 收集KERNEL_H x KERNEL_W窗口内的所有像素
            ap_uint<IFM_CH * KERNEL_DIM * INPW> window_pack = 0;
            int bit_pos = 0;
            for (int kh = 0; kh < KERNEL_H; kh++) {
                for (int kw = 0; kw < KERNEL_W; kw++) {
                    for (int ch = 0; ch < IFM_CH; ch++) {
                        ap_uint<INPW> pixel = line_buffer[kh][(row*STRIDE+kh)*IFM_DIM*IFM_CH + 
                                                                (col*STRIDE+kw)*IFM_CH + ch];
                        window_pack(bit_pos + INPW - 1, bit_pos) = pixel;
                        bit_pos += INPW;
                    }
                }
            }
            out.write(window_pack);
        }
    }
}
```

#### 2.3 Threshold Unit —— 阈值/激活单元

```cpp
// 阈值单元 (ReLU/LeakyReLU等)
template<unsigned int INPW, unsigned int OUTW>
void threshold_unit(
    hls::stream<ap_int<INPW>>& in,
    hls::stream<ap_uint<OUTW>>& out,
    ap_int<INPW> threshold = 0
) {
    #pragma HLS pipeline II=1
    
    ap_int<INPW> in_val = in.read();
    ap_int<OUTW> out_val;
    
    // ReLU: max(0, x)
    if (in_val > threshold) {
        out_val = in_val;  // 可能需要饱和截断到OUTW
    } else {
        out_val = 0;
    }
    
    out.write(out_val);
}
```

#### 2.4 硬件原语选择策略

根据设计目标选择HLS或RTL实现：

| 维度 | HLS路径 | RTL路径 |
|------|---------|---------|
| **开发速度** | 快 (C++模板参数化) | 慢 (手写Verilog) |
| **资源效率** | 大设计时接近RTL | 小设计时更优 |
| **综合时间** | 慢 (10×以上) | 快 |
| **时序性能** | 较慢 | 快45-80% |
| **灵活性** | 高 (模板参数) | 中 (参数化module) |
| **适用场景** | 快速原型、大设计 | 生产部署、小设计 |

**推荐策略**：
- 默认使用**HLS路径** (基于`finn-hlslib`模板库)
- 对于关键路径或资源敏感层，可切换到**RTL路径**
- FINN v0.10+已支持混合HLS/RTL的架构

---

### Step 3: 数据流网络构建

数据流架构的核心是**层间通过FIFO直接连接，数据以流式方式传递，无需外部存储器介入**。

#### 3.1 层间连接设计

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class StreamConnection:
    source_layer: str
    target_layer: str
    data_width: int          # 每周期数据位宽
    fifo_depth: int          # FIFO深度
    throughput: float        # 期望吞吐量 (元素/周期)

class DataflowNetworkBuilder:
    def __init__(self, graph_ir):
        self.graph_ir = graph_ir
        self.connections = []
    
    def build_connections(self) -> List[StreamConnection]:
        """为每对相邻层建立流连接"""
        for i, node in enumerate(self.graph_ir.nodes[:-1]):
            next_node = self.graph_ir.nodes[i + 1]
            
            # 计算连接参数
            data_width = self._calc_data_width(node, next_node)
            fifo_depth = self._calc_fifo_depth(node, next_node)
            throughput = self._calc_throughput(node, next_node)
            
            self.connections.append(StreamConnection(
                source_layer=node.name,
                target_layer=next_node.name,
                data_width=data_width,
                fifo_depth=fifo_depth,
                throughput=throughput
            ))
        
        return self.connections
    
    def _calc_fifo_depth(self, src, dst) -> int:
        """
        计算FIFO深度以匹配上下游吞吐量
        
        原则: FIFO深度需要足够缓冲以应对流水线停顿
        """
        # 简化计算: 2倍最大延迟
        src_latency = self._estimate_latency(src)
        dst_latency = self._estimate_latency(dst)
        return max(16, 2 * (src_latency + dst_latency))
```

#### 3.2 流水线平衡与吞吐量匹配

```python
class PipelineBalancer:
    def __init__(self, graph_ir):
        self.graph_ir = graph_ir
    
    def balance(self) -> dict:
        """
        平衡各层吞吐量，识别瓶颈层
        
        Returns:
            {
                'bottleneck_layers': [...],
                'suggested_parallelism': {...}
            }
        """
        layer_throughputs = {}
        
        for node in self.graph_ir.nodes:
            # 计算该层在当前并行度下的吞吐量 (元素/周期)
            if node.op_type in ['convolution', 'matmul', 'linear']:
                # MVU吞吐量 = PE * SIMD / (矩阵维度 / SIMD) 
                # 简化: 1 element/cycle per PE
                pe = node.attributes.get('pe_parallel', 16)
                throughput = pe  # 每周期输出PE个元素
            else:
                throughput = 1  # 标量操作
            
            layer_throughputs[node.name] = throughput
        
        # 找到最小吞吐量 (瓶颈)
        min_throughput = min(layer_throughputs.values())
        bottleneck_layers = [name for name, tp in layer_throughputs.items() 
                             if tp == min_throughput]
        
        # 建议提升瓶颈层的并行度
        suggestions = {}
        for name in bottleneck_layers:
            current_pe = self._get_pe(name)
            suggestions[name] = {
                'current_pe': current_pe,
                'suggested_pe': current_pe * 2,
                'reason': 'bottleneck_layer'
            }
        
        return {
            'bottleneck_layers': bottleneck_layers,
            'min_throughput': min_throughput,
            'suggestions': suggestions
        }
```

#### 3.3 AXI-Stream接口

所有硬件模块通过**AXI-Stream协议**连接，使用标准的`valid`/`ready`握手信号：

```verilog
// AXI-Stream接口定义
interface axi_stream_if #(
    parameter DATA_WIDTH = 8
) (
    input clk,
    input rst
);
    logic [DATA_WIDTH-1:0] tdata;
    logic                  tvalid;
    logic                  tready;
    logic                  tlast;   // 可选: 帧结束标志
    
    modport source (
        output tdata, tvalid, tlast,
        input  tready
    );
    modport sink (
        input  tdata, tvalid, tlast,
        output tready
    );
endinterface
```

---

### Step 4: 完整硬件生成流程

#### 4.1 整体生成流程

FINN的编译流程包括多个分析和变换pass：

```
QONNX模型 (来自Module 1/2)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Pass 1: 图优化 (Graph Optimization)                       │
│  • 折叠BatchNorm到卷积                                      │
│  • 消除冗余reshape操作                                      │
│  • 算子融合                                                │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Pass 2: 降维 (Lowering)                                   │
│  • 卷积 → SWU + MVU                                        │
│  • 确定各层硬件原语                                         │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Pass 3: 数据流生成 (Dataflow Generation)                  │
│  • 为每层分配计算资源 (PE/SIMD)                             │
│  • 插入FIFO缓冲                                            │
│  • 生成流水线连接                                           │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Pass 4: HLS代码生成 (HLS Code Generation)                 │
│  • 实例化finn-hlslib模板                                    │
│  • 嵌入权重ROM (internal_embedded模式)                     │
│  • 生成顶层缝合代码                                         │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Pass 5: IP打包与集成 (IP Packaging & Integration)         │
│  • 生成Vivado IP (stitched_ip)                  │
│  • 生成测试激励 (testbench)                    │
│  • 生成Python驱动                                  │
└───────────────────────────────────────────────────────────┘
        │
        ▼
    输出: HLS代码 / RTL + IP + 测试平台
```

#### 4.2 代码生成器实现

```python
class HLSGenerator:
    """HLS代码生成器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.template_env = self._setup_jinja_env()
    
    def generate_layer_hls(self, node, weights_info) -> str:
        """为单层生成HLS代码"""
        if node.op_type == 'convolution':
            return self._generate_conv_hls(node, weights_info)
        elif node.op_type == 'linear':
            return self._generate_fc_hls(node, weights_info)
        elif node.op_type == 'relu':
            return self._generate_threshold_hls(node)
        # ...
    
    def _generate_conv_hls(self, node, weights_info) -> str:
        """生成卷积层的HLS代码 (SWU + MVU)"""
        template = """
        #include <ap_int.h>
        #include <hls_stream.h>
        
        void {{ layer_name }}(
            hls::stream<ap_uint<{{ inpw }}>>& in,
            hls::stream<ap_uint<{{ outpw }}>>& out
        ) {
            #pragma HLS dataflow
            
            // 滑动窗口单元
            hls::stream<ap_uint<{{ swu_out_width }}>> swu_out;
            #pragma HLS stream variable=swu_out depth={{ fifo_depth }}
            
            sliding_window_unit<
                {{ ifm_ch }},
                {{ kernel_dim }},
                {{ ifm_dim }},
                {{ ofm_dim }},
                {{ stride }},
                {{ inpw }}
            >(in, swu_out);
            
            // 矩阵向量乘单元 (权重内嵌)
            const ap_uint<{{ wmw }}> weights[{{ pe }}][{{ matrix_dim / simd }}][{{ simd }}] = {
                {{ weight_initializer }}
            };
            
            mvu<
                {{ simd }},
                {{ pe }},
                {{ inpw }},
                {{ wmw }},
                {{ outpw }},
                {{ matrix_dim }}
            >(swu_out, out, weights);
        }
        """
        return self.template_env.from_string(template).render(
            layer_name=node.name,
            inpw=node.attributes.get('input_bit_width', 4),
            outpw=node.attributes.get('output_bit_width', 4),
            wmw=node.attributes.get('weight_bit_width', 4),
            pe=node.attributes.get('pe_parallel', 16),
            simd=node.attributes.get('simd_parallel', 8),
            ifm_ch=node.attributes.get('in_channels'),
            # ... 更多参数
            weight_initializer=self._format_weights(weights_info)
        )
    
    def generate_top(self, layers: List) -> str:
        """生成顶层模块"""
        template = """
        void top(
            hls::stream<ap_uint<{{ inpw }}>>& input,
            hls::stream<ap_uint<{{ outpw }}>>& output
        ) {
            #pragma HLS dataflow
            
            // 层间流
            {% for layer in layers %}
            hls::stream<ap_uint<{{ layer.stream_width }}>> stream_{{ layer.idx }};
            #pragma HLS stream variable=stream_{{ layer.idx }} depth={{ layer.fifo_depth }}
            {% endfor %}
            
            // 实例化各层
            {% for layer in layers %}
            {{ layer.func_name }}(
                {% if loop.first %}input{% else %}stream_{{ layer.idx - 1 }}{% endif %},
                {% if loop.last %}output{% else %}stream_{{ layer.idx }}{% endif %}
            );
            {% endfor %}
        }
        """
        return self.template_env.from_string(template).render(
            inpw=self.graph_ir.input_bit_width,
            outpw=self.graph_ir.output_bit_width,
            layers=self._prepare_layer_info()
        )
```

#### 4.3 RTL路径生成 (可选)

对于追求极致性能的场景，可生成手写RTL替代HLS：

```python
class RTLGenerator:
    """RTL代码生成器 (Verilog/VHDL)"""
    
    def generate_mvu_rtl(self, params: dict) -> str:
        """生成MVU的Verilog RTL"""
        template = """
        module mvu #(
            parameter PE = {{ pe }},
            parameter SIMD = {{ simd }},
            parameter INPW = {{ inpw }},
            parameter WMW = {{ wmw }},
            parameter OUTW = {{ outpw }},
            parameter MATRIX_DIM = {{ matrix_dim }}
        )(
            input clk,
            input rst,
            input [SIMD*INPW-1:0] in_data,
            input in_valid,
            output in_ready,
            output [PE*OUTW-1:0] out_data,
            output out_valid,
            input out_ready
        );
            // PE阵列实例化
            generate
                for (genvar pe_idx = 0; pe_idx < PE; pe_idx++) begin : gen_pe
                    pe #(
                        .SIMD(SIMD),
                        .INPW(INPW),
                        .WMW(WMW)
                    ) pe_inst (
                        .clk(clk),
                        .rst(rst),
                        .in_pack(in_data),
                        .weight_pack(weights[pe_idx][simd_idx]),
                        .acc_out(acc_outs[pe_idx])
                    );
                end
            endgenerate
            
            // 输出打包与阈值化
            // ...
        endmodule
        """
        return self.template_env.from_string(template).render(**params)
```

---

### Step 5: 设计空间探索 (DSE)

FINN支持自动或手动的设计空间探索，在不同资源与性能间权衡。

```python
class DesignSpaceExplorer:
    """设计空间探索器"""
    
    def __init__(self, graph_ir, target_device: str):
        self.graph_ir = graph_ir
        self.target_device = target_device
        self.resource_limits = self._get_device_resources(target_device)
    
    def explore(self) -> List[dict]:
        """探索不同并行度配置"""
        candidates = []
        
        # 并行度候选值
        pe_options = [4, 8, 16, 32, 64]
        simd_options = [2, 4, 8, 16]
        
        for pe in pe_options:
            for simd in simd_options:
                # 估算资源占用
                resources = self._estimate_resources(pe, simd)
                
                # 检查是否在目标器件限制内
                if self._within_limits(resources):
                    # 估算性能
                    throughput = self._estimate_throughput(pe, simd)
                    latency = self._estimate_latency(pe, simd)
                    
                    candidates.append({
                        'pe': pe,
                        'simd': simd,
                        'resources': resources,
                        'throughput': throughput,
                        'latency': latency,
                        'score': self._score(throughput, resources)
                    })
        
        # 按得分排序
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates
    
    def _estimate_resources(self, pe: int, simd: int) -> dict:
        """估算FPGA资源占用"""
        # 简化模型: LUT ≈ PE * SIMD * 常数
        lut = pe * simd * 50 + 1000
        dsp = pe * simd * 1  # 每个乘法器一个DSP
        bram = pe * 2  # 权重缓存
        return {'LUT': lut, 'DSP': dsp, 'BRAM': bram}
```

---

### 输出规范

Module 3的输出应包含：

**1. HLS C++源代码目录**：
```
hls_src/
├── top.cpp               # 顶层模块
├── top.h
├── layer_conv1.cpp       # 各层实现
├── layer_conv1.h
├── layer_fc.cpp
├── weights/              # 权重ROM文件 (从Module 2复制)
│   ├── layer_conv1_weights.mem
│   └── ...
└── build_dataflow.py     # FINN风格的数据流构建脚本
```

**2. 缝合IP (Stitched IP)**：
- Vivado IP格式的加速器核心
- AXI-Stream输入/输出接口
- 可直接集成到Vivado Block Design中

**3. 测试与部署文件**：
- `testbench.cpp` / `testbench.v`：仿真测试激励
- `driver.py`：Python驱动代码
- `build.tcl`：Vivado综合脚本
- `constraints.xdc`：时序约束文件

**4. metadata.json**：
```json
{
  "top_module": "top",
  "layers": [
    {
      "name": "conv1",
      "implementation": "hls",
      "source_file": "layer_conv1.cpp",
      "pe": 16,
      "simd": 8,
      "estimated_lut": 2450,
      "estimated_dsp": 128,
      "estimated_bram": 4,
      "throughput": "16 elem/cycle",
      "latency": "125 cycles"
    }
  ],
  "total_estimated_resources": {
    "LUT": 12450,
    "DSP": 384,
    "BRAM": 16
  },
  "target_device": "xczu7ev-ffvc1156-2-e"
}
```

---

### 关键设计决策总结

| 决策点 | 推荐方案 | 依据 |
|--------|---------|------|
| **计算范式** | 统一降维为GEMM + MVU | FINN成熟实践 |
| **实现路径** | 默认HLS，关键层切RTL | 开发速度 vs 资源效率权衡 |
| **权重存储** | internal_embedded (ROM内嵌) | 消除权重加载延迟 |
| **层间通信** | AXI-Stream + FIFO | 标准流式协议 |
| **流水线** | 全数据流，层间FIFO解耦 | 最大化并行度 |
| **并行度配置** | PE × SIMD = 目标吞吐量 | 设计空间探索确定 |
| **代码生成** | Jinja2模板引擎 | 灵活可扩展 |

这个Module 3的设计充分借鉴了AMD FINN框架的成熟架构，将量化神经网络计算图高效映射为定制化的数据流硬件，为Module 4的RTL综合与最终FPGA部署奠定了坚实基础。
