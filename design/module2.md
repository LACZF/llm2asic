## Module 2: 量化与权重预处理 —— 详细设计实现分析

Module 2的核心使命是**将Module 1输出的浮点权重张量，转换为适合硬件直接使用的定点格式，并进行针对硬件数据流架构的预处理优化**。这是连接"算法模型"与"硬件电路"的关键桥梁。

---

### 整体工作流程概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   输入: GraphIR (来自Module 1)                              │
│        包含: 计算图 + FP32权重张量 + 量化参数(如有)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: 图优化与算子融合 (Graph Optimization & Operator Fusion)             │
│  • BatchNorm折叠到卷积层/全连接层                                            │
│  • 量化缩放因子(scale)与零点(zero point)折叠到权重                          │
│  • 激活函数与相邻算子融合                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: 权重量化 (Weight Quantization)                                     │
│  • 策略选择: PTQ (快速) 或 QAT (高精度)                                     │
│  • 量化方案: 对称量化 / 非对称量化                                           │
│  • 精度选择: 按层/按通道混合精度                                            │
│  • 校准数据集收集与统计量计算                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: 权重预处理与重排 (Weight Preprocessing & Reordering)               │
│  • 权重重排: 适配硬件PE阵列的数据流布局                                      │
│  • 权重分块: 按输出通道/输入通道分块                                        │
│  • 稀疏权重处理: 剪枝后的权重压缩编码                                        │
│  • 偏置处理: 量化后偏置的合并与存储                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 4: ROM初始化文件生成 (ROM Initialization File Generation)             │
│  • 格式选择: .mem (XPM) / .coe (Vivado IP) / .hex (通用)                   │
│  • 按层组织: 每层权重独立ROM文件                                            │
│  • 元数据生成: 权重地址映射、位宽、深度信息                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   输出: 量化后的GraphIR + ROM初始化文件                      │
│        传递给Module 3进行硬件架构生成                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 1: 图优化与算子融合

在量化之前，首先需要对计算图进行优化，将某些算子融合到相邻层中，以减少量化误差和硬件开销。

#### 1.1 BatchNorm折叠

BatchNorm层在推理时可以折叠到前一层的权重和偏置中。

**数学原理**：

对于卷积层 `y = W * x + b`，后接BatchNorm：
```
μ = batch mean, σ² = batch variance, γ = scale, β = shift
y_bn = γ * (y - μ) / sqrt(σ² + ε) + β
    = (γ / sqrt(σ² + ε)) * (W * x + b) + (β - γ * μ / sqrt(σ² + ε))
    = W' * x + b'
```
其中：
```
W' = W * γ / sqrt(σ² + ε)
b' = (b - μ) * γ / sqrt(σ² + ε) + β
```

**实现代码**：

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class BatchNormParams:
    gamma: np.ndarray      # 缩放因子
    beta: np.ndarray       # 偏移量
    moving_mean: np.ndarray
    moving_var: np.ndarray
    eps: float = 1e-5

def fold_batch_norm(weight: np.ndarray, bias: Optional[np.ndarray], 
                    bn: BatchNormParams) -> tuple:
    """将BatchNorm参数折叠到卷积/全连接层的权重和偏置中"""
    # 计算缩放因子
    scale = bn.gamma / np.sqrt(bn.moving_var + bn.eps)
    
    # 折叠到权重
    # 对于卷积: weight shape = [out_ch, in_ch, kh, kw]
    # 需要对输出通道维度进行广播
    if weight.ndim == 4:  # 卷积
        weight_folded = weight * scale.reshape(-1, 1, 1, 1)
    else:  # 全连接
        weight_folded = weight * scale.reshape(-1, 1)
    
    # 折叠到偏置
    if bias is not None:
        bias_folded = bias * scale + (bn.beta - bn.moving_mean * scale)
    else:
        bias_folded = bn.beta - bn.moving_mean * scale
    
    return weight_folded, bias_folded
```

#### 1.2 量化缩放因子折叠

当使用非对称量化时，量化缩放因子和零点也可以折叠到权重中。FINN的streamlining transformations会自动将量化器scale值折叠到threshold操作中。

**原理**：对于量化后的卷积，其计算可表示为：
```
y_quant = quantize(Σ (W_quant * x_quant) * scale_w * scale_x / scale_y)
```
其中 `scale_w * scale_x / scale_y` 可以预先计算并合并到权重或偏置中。

---

### Step 2: 权重量化

#### 2.1 量化策略选择

编译器需要支持两种量化策略：

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **PTQ (Post-Training Quantization)** | 无需重训练，速度快（几分钟） | 精度可能有损失 | 快速原型、精度要求不高的场景 |
| **QAT (Quantization-Aware Training)** | 精度更高 | 需要完整训练集，耗时长 | 生产部署、精度敏感场景 |

**实现建议**：先尝试PTQ，若精度损失可接受则直接使用；否则fallback到QAT。

#### 2.2 量化方案

**对称量化 (Symmetric Quantization)**：
```
x_q = round(clip(x / scale, -2^(b-1), 2^(b-1)-1))
```
- 零点固定为0
- 适合权重分布关于0对称的场景
- 硬件实现更简单

**非对称量化 (Asymmetric Quantization)**：
```
x_q = round(clip(x / scale + zero_point, 0, 2^b - 1))
```
- 有零点偏移
- 适合激活值分布不均匀的场景（如ReLU后的输出）

**实现代码**：

```python
from enum import Enum
from typing import Tuple
import numpy as np

class QuantScheme(Enum):
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"

class QuantStrategy(Enum):
    PTQ = "ptq"
    QAT = "qat"

def compute_quant_params(data: np.ndarray, bit_width: int, 
                         scheme: QuantScheme) -> Tuple[float, int]:
    """计算量化参数: scale和zero_point"""
    min_val, max_val = np.min(data), np.max(data)
    
    if scheme == QuantScheme.SYMMETRIC:
        # 对称量化: zero_point = 0
        abs_max = max(abs(min_val), abs(max_val))
        scale = abs_max / (2 ** (bit_width - 1) - 1)
        zero_point = 0
    else:
        # 非对称量化
        qmin, qmax = 0, 2 ** bit_width - 1
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = int(round(qmin - min_val / scale))
        zero_point = max(qmin, min(qmax, zero_point))  # 截断到有效范围
    
    return scale, zero_point

def quantize_tensor(data: np.ndarray, bit_width: int, 
                    scheme: QuantScheme) -> Tuple[np.ndarray, float, int]:
    """量化张量"""
    scale, zero_point = compute_quant_params(data, bit_width, scheme)
    
    if scheme == QuantScheme.SYMMETRIC:
        qmin, qmax = -2**(bit_width-1), 2**(bit_width-1) - 1
        data_q = np.round(np.clip(data / scale, qmin, qmax)).astype(np.int32)
    else:
        qmin, qmax = 0, 2**bit_width - 1
        data_q = np.round(np.clip(data / scale + zero_point, qmin, qmax)).astype(np.int32)
    
    return data_q, scale, zero_point
```

#### 2.3 校准数据集收集 (PTQ)

PTQ需要使用校准数据集来分析权重和激活的分布：

```python
class Calibrator:
    def __init__(self, model, calibration_loader, num_samples=1000):
        self.model = model
        self.calibration_loader = calibration_loader
        self.num_samples = num_samples
        self.activations = {}  # 存储各层激活值统计
    
    def collect_stats(self):
        """收集各层激活值的统计信息"""
        hooks = []
        
        def hook_fn(name):
            def fn(module, input, output):
                if name not in self.activations:
                    self.activations[name] = []
                self.activations[name].append(output.detach().cpu().numpy())
            return fn
        
        # 为每个需要量化的层注册hook
        for name, module in self.model.named_modules():
            if self._need_quantize(module):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # 运行校准
        for i, (inputs, _) in enumerate(self.calibration_loader):
            if i >= self.num_samples:
                break
            self.model(inputs)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        # 计算每个层的量化参数
        quant_params = {}
        for name, acts in self.activations.items():
            all_acts = np.concatenate(acts, axis=0)
            scale, zero_point = compute_quant_params(all_acts, 8, QuantScheme.ASYMMETRIC)
            quant_params[name] = {'scale': scale, 'zero_point': zero_point}
        
        return quant_params
```

#### 2.4 混合精度量化

不同层对量化精度的敏感度不同，可以采用混合精度策略：

```python
class MixedPrecisionQuantizer:
    def __init__(self, sensitivity_profile: dict):
        """
        sensitivity_profile: 层名 -> 精度位宽 (如 {'conv1': 8, 'conv2': 4, 'fc': 6})
        """
        self.sensitivity_profile = sensitivity_profile
    
    def quantize_model(self, graph_ir: GraphIR) -> GraphIR:
        for node in graph_ir.nodes:
            if node.name in self.sensitivity_profile:
                bit_width = self.sensitivity_profile[node.name]
            else:
                bit_width = 8  # 默认8-bit
            
            # 获取该节点关联的权重
            for weight_name in node.weight_names:
                weight = graph_ir.weights[weight_name]
                weight.data_q, weight.scale, weight.zero_point = quantize_tensor(
                    weight.data, bit_width, QuantScheme.SYMMETRIC
                )
                weight.quant_bit_width = bit_width
        return graph_ir
```

**精度分配策略**：
- 首层和最后一层对精度更敏感，使用较高精度（如8-bit）
- 中间层可以使用较低精度（如4-bit、2-bit甚至1-bit）
- Vitis AI默认使用INT8

---

### Step 3: 权重预处理与重排

量化完成后，需要对权重数据进行针对硬件架构的预处理和重排。

#### 3.1 内存模式选择

FINN定义了三种内存模式，控制权重如何存储和访问：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **internal_embedded** | 权重作为常量"烘焙"到硬件模块中 | 权重固定、追求最小资源占用 |
| **internal_decoupled** | 权重存储在独立内存中，通过权重流读取 | 需要运行时更新权重 |
| **external** | 权重从外部源流入 | 权重太大无法片上存储 |

对于"权重转RTL"的场景，**internal_embedded**模式最为关键——权重被直接嵌入到RTL中作为常量，消除运行时权重加载延迟。

**注意**：RTL MVAU目前不支持 `internal_embedded` 模式，仅支持 `internal_decoupled` 和 `external`。若需要RTL级别的权重嵌入，可能需要自定义实现或使用HLS MVAU。

#### 3.2 权重重排 (Weight Reordering)

为了适配硬件PE阵列的数据流，需要对权重进行重排。

**卷积权重重排**：

对于卷积层，权重 shape 为 `[out_ch, in_ch, kh, kw]`。硬件PE阵列通常按输出通道并行，因此需要将权重按输出通道分块：

```python
def reorder_conv_weights(weight: np.ndarray, pe_parallel: int, 
                         simd_parallel: int) -> np.ndarray:
    """
    重排卷积权重以适配硬件数据流
    
    Args:
        weight: shape [out_ch, in_ch, kh, kw]
        pe_parallel: PE并行度 (输出通道并行)
        simd_parallel: SIMD并行度 (输入通道并行)
    
    Returns:
        重排后的权重数组
    """
    out_ch, in_ch, kh, kw = weight.shape
    
    # Step 1: 将卷积核展开为 [out_ch, in_ch * kh * kw]
    weight_flat = weight.reshape(out_ch, -1)
    
    # Step 2: 按输出通道分块 (PE维度)
    out_ch_padded = ((out_ch + pe_parallel - 1) // pe_parallel) * pe_parallel
    weight_pad = np.pad(weight_flat, 
                        ((0, out_ch_padded - out_ch), (0, 0)), 
                        mode='constant', constant_values=0)
    
    # Step 3: 按输入通道分块 (SIMD维度)
    in_dim = in_ch * kh * kw
    in_dim_padded = ((in_dim + simd_parallel - 1) // simd_parallel) * simd_parallel
    weight_pad = np.pad(weight_pad, 
                        ((0, 0), (0, in_dim_padded - in_dim)),
                        mode='constant', constant_values=0)
    
    # Step 4: 重排为 [pe_blocks, simd_blocks, pe_parallel, simd_parallel]
    pe_blocks = out_ch_padded // pe_parallel
    simd_blocks = in_dim_padded // simd_parallel
    weight_reordered = weight_pad.reshape(pe_blocks, pe_parallel, 
                                          simd_blocks, simd_parallel)
    weight_reordered = weight_reordered.transpose(0, 2, 1, 3)
    
    return weight_reordered
```

**全连接权重重排**：

全连接层可以视为 `[out_features, in_features]` 的矩阵乘法，重排方式类似：

```python
def reorder_fc_weights(weight: np.ndarray, pe_parallel: int, 
                       simd_parallel: int) -> np.ndarray:
    """重排全连接层权重"""
    out_features, in_features = weight.shape
    
    # 按输出维度分块 (PE)
    out_padded = ((out_features + pe_parallel - 1) // pe_parallel) * pe_parallel
    weight_pad = np.pad(weight, ((0, out_padded - out_features), (0, 0)), 
                        mode='constant', constant_values=0)
    
    # 按输入维度分块 (SIMD)
    in_padded = ((in_features + simd_parallel - 1) // simd_parallel) * simd_parallel
    weight_pad = np.pad(weight_pad, ((0, 0), (0, in_padded - in_features)),
                        mode='constant', constant_values=0)
    
    # 重排
    pe_blocks = out_padded // pe_parallel
    simd_blocks = in_padded // simd_parallel
    weight_reordered = weight_pad.reshape(pe_blocks, pe_parallel,
                                          simd_blocks, simd_parallel)
    weight_reordered = weight_reordered.transpose(0, 2, 1, 3)
    
    return weight_reordered
```

#### 3.3 稀疏权重处理

对于剪枝后的稀疏模型，需要特殊编码以利用稀疏性：

```python
class SparseWeightEncoder:
    def __init__(self, sparsity_threshold: float = 1e-6):
        self.threshold = sparsity_threshold
    
    def encode_sparse(self, weight: np.ndarray) -> dict:
        """编码稀疏权重为 (索引, 值) 对"""
        # 找到非零权重
        mask = np.abs(weight) > self.threshold
        indices = np.where(mask)
        values = weight[mask]
        
        # 计算稀疏率
        sparsity = 1.0 - (np.sum(mask) / weight.size)
        
        return {
            'indices': indices,
            'values': values,
            'sparsity': sparsity,
            'original_shape': weight.shape
        }
    
    def generate_sparse_rom_data(self, sparse_data: dict, 
                                  bit_width: int) -> np.ndarray:
        """生成稀疏权重的ROM数据"""
        indices = sparse_data['indices']
        values = sparse_data['values']
        
        # 将索引和值打包为固定格式
        # 格式: [index, value] 或使用位压缩
        num_entries = len(values)
        rom_data = np.zeros((num_entries, 2), dtype=np.int32)
        rom_data[:, 0] = self._flatten_index(indices, sparse_data['original_shape'])
        rom_data[:, 1] = values.astype(np.int32)
        
        return rom_data
```

#### 3.4 偏置处理

量化后的偏置需要与权重一起存储：

```python
def quantize_bias(bias: np.ndarray, weight_scale: float, 
                  input_scale: float) -> np.ndarray:
    """
    量化偏置
    
    对于卷积: y = Σ(W * x) + b
    量化后: y_q = Σ(W_q * x_q) * (scale_w * scale_x) + b_q * scale_b
    
    为了简化，将偏置缩放到与累加器相同的尺度
    """
    accumulator_scale = weight_scale * input_scale
    bias_q = np.round(bias / accumulator_scale).astype(np.int32)
    return bias_q
```

---

### Step 4: ROM初始化文件生成

量化并重排后的权重需要生成硬件可读的ROM初始化文件。

#### 4.1 文件格式选择

AMD FPGA生态支持多种ROM初始化格式：

| 格式 | 说明 | 适用工具 |
|------|------|---------|
| **.mem** | ASCII十六进制，每行一个地址 | XPM (AMD参数化宏) |
| **.coe** | Vivado IP核格式，包含`memory_initialization_radix`和`memory_initialization_vector` | Vivado Block Memory Generator |
| **.hex** | 通用十六进制格式，Verilog `$readmemh`可读 | 通用RTL仿真 |

**推荐**：生成`.mem`格式，因为XPM是AMD官方推荐的存储器生成方式，同时也可转换为`.coe`格式兼容Vivado IP。

#### 4.2 ROM数据生成

```python
class ROMGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_mem_file(self, data: np.ndarray, bit_width: int,
                          layer_name: str) -> Path:
        """
        生成.mem格式的ROM初始化文件
        
        Args:
            data: 量化后的权重数据 (已重排)
            bit_width: 每个权重的位宽
            layer_name: 层名称，用于文件名
        """
        # 将数据打包为指定位宽的字
        if bit_width < 8:
            # 打包多个权重到一个字节
            packed = self._pack_bits(data, bit_width)
        else:
            packed = data.astype(np.uint8)
        
        # 生成.mem文件
        mem_path = self.output_dir / f"{layer_name}_weights.mem"
        with open(mem_path, 'w') as f:
            for val in packed.flatten():
                f.write(f"{val:02X}\n")  # 每行一个十六进制字节
        
        return mem_path
    
    def _pack_bits(self, data: np.ndarray, bit_width: int) -> np.ndarray:
        """将低位宽数据打包到字节中"""
        values_per_byte = 8 // bit_width
        data_flat = data.flatten()
        
        # 填充到values_per_byte的整数倍
        pad_len = (values_per_byte - len(data_flat) % values_per_byte) % values_per_byte
        data_padded = np.pad(data_flat, (0, pad_len), mode='constant', constant_values=0)
        
        # 打包
        packed = np.zeros(len(data_padded) // values_per_byte, dtype=np.uint8)
        for i in range(len(packed)):
            byte = 0
            for j in range(values_per_byte):
                idx = i * values_per_byte + j
                byte |= (int(data_padded[idx]) & ((1 << bit_width) - 1)) << (j * bit_width)
            packed[i] = byte
        
        return packed
    
    def generate_coe_file(self, data: np.ndarray, bit_width: int,
                          layer_name: str, radix: int = 16) -> Path:
        """生成.coe格式的ROM初始化文件 (Vivado IP)"""
        coe_path = self.output_dir / f"{layer_name}_weights.coe"
        
        # 确定数据格式
        if bit_width <= 8:
            data_str = [f"{int(v):02X}" for v in data.flatten()]
        elif bit_width <= 16:
            data_str = [f"{int(v):04X}" for v in data.flatten()]
        else:
            data_str = [f"{int(v):08X}" for v in data.flatten()]
        
        with open(coe_path, 'w') as f:
            f.write(f"memory_initialization_radix={radix};\n")
            f.write("memory_initialization_vector=\n")
            f.write(",\n".join(data_str))
            f.write(";\n")
        
        return coe_path
    
    def generate_metadata(self, layer_name: str, data: np.ndarray,
                          bit_width: int, address: int) -> dict:
        """生成权重元数据，供Module 3使用"""
        return {
            'layer_name': layer_name,
            'rom_file': f"{layer_name}_weights.mem",
            'data_width': bit_width,
            'rom_depth': data.size,
            'base_address': address,
            'original_shape': list(data.shape)
        }
```

#### 4.3 XPM ROM例化模板

生成的.mem文件可通过XPM在RTL中例化：

```verilog
// XPM单端口ROM例化模板
xpm_memory_sprom #(
    .MEMORY_SIZE(8192),           // 总bits数 = data_width * depth
    .MEMORY_PRIMITIVE("auto"),    // "auto", "distributed", "block"
    .MEMORY_INIT_FILE("layer1_weights.mem"),  // ROM初始化文件
    .MEMORY_INIT_PARAM(""),       // 或直接使用参数初始化
    .USE_MEM_INIT(1),             // 1: 使用初始化文件
    .WAKEUP_TIME("disable_sleep"),
    .MESSAGE_CONTROL(0),
    .ECC_MODE("no_ecc"),
    .AUTO_SLEEP_TIME(0),
    .READ_DATA_WIDTH(8),
    .ADDR_WIDTH_A(10)             // log2(depth)
) xpm_memory_sprom_inst (
    .douta(rom_data_out),         // 输出数据
    .addra(read_address),         // 读地址
    .clka(clock),                 // 时钟
    .rsta(reset),                 // 复位
    .sleep(1'b0),
    .injectdbiterra(1'b0),
    .injectsbiterra(1'b0)
);
```

---

### 输出规范

Module 2的输出应包含：

**1. 量化后的GraphIR**：
- 每个权重张量添加 `data_q` (量化后整数数据)、`scale`、`zero_point`、`quant_bit_width` 字段
- 每个节点添加 `mem_mode` 字段 (internal_embedded/internal_decoupled/external)
- 每个节点添加 `pe_parallel`、`simd_parallel` 等硬件并行度参数

**2. ROM文件目录**：
```
weights_rom/
├── layer_conv1_weights.mem
├── layer_conv1_weights.coe
├── layer_conv2_weights.mem
├── layer_fc_weights.mem
├── metadata.json          # 所有层的元数据汇总
└── weights_info.yaml      # 人类可读的权重信息
```

**3. metadata.json 示例**：
```json
{
  "layers": [
    {
      "name": "conv1",
      "op_type": "convolution",
      "rom_file": "layer_conv1_weights.mem",
      "data_width": 4,
      "rom_depth": 16384,
      "base_address": 0,
      "shape": [64, 3, 3, 3],
      "mem_mode": "internal_embedded",
      "pe_parallel": 16,
      "simd_parallel": 8
    }
  ],
  "total_weight_bits": 65536,
  "total_rom_size_bytes": 8192
}
```

---

### 关键设计决策总结

| 决策点 | 推荐方案 | 依据 |
|--------|---------|------|
| 量化策略 | 先PTQ，精度不足则QAT | Vitis AI实践 |
| 量化方案 | 权重用对称量化，激活用非对称量化 | 硬件实现简单 |
| 精度配置 | 混合精度：首尾层8-bit，中间层4-bit | 精度-面积权衡 |
| 内存模式 | internal_embedded (权重烘焙到RTL) | 极致性能 |
| ROM格式 | .mem (XPM) + .coe (Vivado兼容) | AMD官方推荐 |
| 权重打包 | 低位宽打包到字节 | 减少ROM位宽 |

这个Module 2的设计充分考虑了与AMD FINN/Vitis AI工具链的兼容性，同时也为自定义RTL生成提供了灵活的接口。
