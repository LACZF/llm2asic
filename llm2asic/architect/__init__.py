"""architect subpackage: 架构规划（ArchDesc 生成）。"""

from .archdesc import ArchDesc, build_archdesc, EngineSpec, DataflowLink
from .memory import budget_check
from .dse import explore

__all__ = ["ArchDesc", "build_archdesc", "EngineSpec", "DataflowLink",
           "budget_check", "explore", "generate"]


def generate(qmodel, arch_cfg, out_dir: str = "build_out") -> ArchDesc:
    """Architect 主入口：QuantizedModel -> ArchDesc + 报告落盘。"""
    import os
    from ..ir.serialize import dump_json

    adir = os.path.join(out_dir, "architect")
    os.makedirs(adir, exist_ok=True)
    desc = build_archdesc(qmodel, arch_cfg)
    dump_json(desc.to_dict(), os.path.join(adir, "archdesc.json"))

    dse = explore(qmodel)
    dump_json(dse[:5], os.path.join(adir, "dse_report.json"))

    budget = budget_check(desc, qmodel.config,
                          onchip_cap_bytes=arch_cfg.onchip_mem_bytes)
    dump_json(budget, os.path.join(adir, "memory_report.json"))
    return desc
