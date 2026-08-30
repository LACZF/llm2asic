# llm2asic/architect/archdesc.py
"""ArchDesc 数据结构与构建。对应设计文档 architect.md §1,§3,§7。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class EngineSpec:
    engine_type: str
    params: dict = field(default_factory=dict)
    weight_rom: str | None = None
    scale_rom: str | None = None
    attrs: dict = field(default_factory=dict)


@dataclass
class DataflowLink:
    src: str
    dst: str
    fifo_depth: int = 64
    data_width: int = 32
    protocol: str = "axis"


@dataclass
class ArchDesc:
    top_module: str = "llm_accel"
    input_width: int = 24
    output_width: int = 24
    engines: list = field(default_factory=list)
    dataflow: list = field(default_factory=list)
    schedulers: dict = field(default_factory=dict)
    resource_estimate: dict = field(default_factory=dict)
    target_device: str = "xczu7ev-ffvc1156-2-e"
    note: str = ""
    simd: int = 8
    pe: int = 8
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "top_module": self.top_module,
            "input_width": self.input_width,
            "output_width": self.output_width,
            "engines": self.engines,
            "dataflow": [asdict(d) if isinstance(d, DataflowLink) else d
                         for d in self.dataflow],
            "schedulers": self.schedulers,
            "resource_estimate": self.resource_estimate,
            "target_device": self.target_device,
            "note": self.note,
            "simd": self.simd,
            "pe": self.pe,
            "config": self.config,
        }


def build_archdesc(qmodel, arch_cfg) -> ArchDesc:
    """由 QuantizedModel + ArchConfig 生成 ArchDesc。"""
    simd = arch_cfg.simd
    pe = arch_cfg.pe
    cfg = qmodel.config
    layers = cfg["num_layers"]
    hidden = cfg["hidden"]
    heads = cfg["num_heads"]
    hd = cfg["head_dim"]
    vocab = cfg["vocab_size"]
    act_bits = qmodel.bit_width

    desc = ArchDesc(simd=simd, pe=pe, config=qtok(cfg))
    desc.top_module = f"{cfg['name']}_accel"

    # ---- 引擎清单（共享控制器模型）----
    engines = [
        EngineSpec("embedding_engine", params={"vocab": vocab, "hidden": hidden,
                                               "rom": "wte_q.mem", "vec_bits": 24},
                   attrs={"id": "emb"}),
        EngineSpec("rmsnorm_engine", params={"hidden": hidden, "F": 12},
                   attrs={"id": "rms"}),
        EngineSpec("gemm_engine", params={"pe": pe, "simd": simd,
                                          "act_bits": act_bits, "w_bits": qmodel.bit_width,
                                          "out_bits": 24, "requant_s": 16},
                   attrs={"id": "gemm", "modes": ["gemv"], "engines": list(qmodel.engines.keys())}),
        EngineSpec("ropes_engine", params={"hidden": hidden, "head_dim": hd,
                                           "theta": float(cfg.get("rope_theta", 10000.0))},
                   attrs={"id": "rope"}),
        EngineSpec("attention_engine", params={"heads": heads, "head_dim": hd,
                                               "hidden": hidden, "causal": True,
                                               "scale": 1.0 / (hd ** 0.5)},
                   attrs={"id": "attn"}),
        EngineSpec("activation_engine", params={"slilu": True, "lut": "silu.mem"},
                   attrs={"id": "silu"}),
        EngineSpec("vector_op_engine", params={"hidden": hidden}, attrs={"id": "vec"}),
        EngineSpec("kv_memory", params={"num_layers": layers, "heads": heads,
                                        "head_dim": hd, "max_seq": cfg["max_seq_len"],
                                        "onchip": True}, attrs={"id": "kv"}),
    ]
    desc.engines = [asdict(e) for e in engines]

    # ---- 数据流 ----
    desc.dataflow = [
        {"src": "emb", "dst": "gemm", "fifo_depth": 64, "data_width": hidden, "protocol": "axis"},
        {"src": "gemm", "dst": "attn", "fifo_depth": hidden * 4, "data_width": hidden, "protocol": "axis"},
        {"src": "attn", "dst": "vec", "fifo_depth": hidden, "data_width": hidden, "protocol": "axis"},
        {"src": "vec", "dst": "gemm", "fifo_depth": hidden * 4, "data_width": hidden, "protocol": "axis"},
    ]

    # ---- 调度（decode 主；prefill 复用同一引擎集批量）----
    decode_sched = ["emb"]
    for L in range(layers):
        decode_sched += [f"rms{n1}({L})" for n1 in ["0", "1"]]
        decode_sched += [f"q({L})", f"k({L})", f"v({L})", f"ropeq({L})", f"ropek({L})",
                         f"attn({L})", f"h1({L})", f"g({L})", f"u({L})", f"silu({L})",
                         f"m({L})", f"d({L})", f"h({L})"]
    decode_sched += ["rmsf", "out"]
    desc.schedulers = {
        "decode": decode_sched,
        "prefill": decode_sched,
    }

    # ---- 资源估计（解析模型）----
    gemm_macs = sum(e.c_out * e.c_in for e in qmodel.engines.values())
    desc.resource_estimate = {
        "DSP": int(pe * simd),
        "BRAM": int(2 + (gemm_macs // 4096) + (vocab * hidden // 4096)),
        "LUT": int(pe * simd * 64),
        "gemm_macs_total": int(gemm_macs),
        "layers": layers,
        "hidden": hidden,
    }
    desc.note = "共享顶层控制器：顺序执行算子图；decode 单 token；prefill 复用同一引擎集。"
    return desc


def qtok(cfg: dict) -> dict:
    """把不可 JSON 序列化的配置项转为基础类型。"""
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in cfg.items()}
