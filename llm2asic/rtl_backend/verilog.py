# llm2asic/rtl_backend/verilog.py
"""RTL 生成器：QuantizedModel + ROM(.mem) -> SystemVerilog 顶层与测试台。

算子与 reference.IntModel 逐位一致。模块：gemv / rmsnorm / attn；顶层负责调度。
用 sentinel 令牌 (``@@K@@``) + replace 拼接，避免与 Verilog 花括号冲突。
"""

from __future__ import annotations

import math
import os

import numpy as np

from . import numeric
from .numeric import F, REQUANT_S, ACT_BITS

ACTW = ACT_BITS    # 默认 16；generate() 前按 numeric.ACT_BITS 同步（模型可覆盖）


def _sync_actw() -> None:
    """让 ACTW 跟随 numeric.ACT_BITS（build.act_bits 每模型覆盖后刷新）。"""
    global ACTW
    ACTW = numeric.ACT_BITS
EF = 10
RR_DIV = 2 * F


def _san(k: str) -> str:
    return k.replace(".", "_").replace("-", "_")


def _hex_patterns(vals, bits: int = 24) -> list:
    """把整数值转成与 .mem 相同的位模式 hex 词（写 ROM 时 `v & mask` 截位，
    与 $readmemh 装载语义完全一致：越界值(如 recip 的 2^24)同样被截位）。"""
    nhex = (bits + 3) // 4
    return [f"{int(v) & ((1 << bits) - 1):0{nhex}x}" for v in vals]


def _case_rom_fn(name: str, tokens, out_bits: int = 24,
                 indent: str = "  ") -> str:
    """生成可综合的 case 常量 ROM 函数（无 initial/$readmemh）。

    任意综合器(DC/Genus/Vivado/Yosys/Verilator)都能综合的常数表：
    `function ... name(input logic [A-1:0] a); case (a) idx: name=<const>;
    ... default: name=0; endcase endfunction`。tokens 为位模式 hex 词，
    与 .mem 文件逐位一致（截位/符号语义与 $readmemh 装载完全相同）。
    """
    n = len(tokens)
    abits = max(1, (n - 1).bit_length())
    head = (f"{indent}function automatic logic signed [{out_bits - 1}:0] {name}"
            f"(input logic [{abits - 1}:0] a);")
    lines = [head, f"{indent}  case (a)"]
    for i, t in enumerate(tokens):
        lines.append(f"{indent}    {i}: {name} = {out_bits}'h{t};")
    lines.append(f"{indent}    default: {name} = {out_bits}'h0;")
    lines.append(f"{indent}  endcase")
    lines.append(f"{indent}endfunction")
    return "\n".join(lines)


def _precompute_rope_cs(cfg: dict) -> tuple:
    hd = cfg["head_dim"]
    seq = cfg["max_seq_len"]
    theta = float(cfg.get("rope_theta", 10000.0))
    cq = [[int(round(math.cos(pos / (theta ** (2.0 * i / hd))) * (2.0 ** F)))
           for i in range(hd // 2)] for pos in range(seq)]
    sq = [[int(round(math.sin(pos / (theta ** (2.0 * i / hd))) * (2.0 ** F)))
           for i in range(hd // 2)] for pos in range(seq)]
    return cq, sq


# 24-bit 饱和定点 rr（rshift_round + clamp）——与 reference.clamp_act 对齐
# 注意：负分支必须用 if/else（iverilog 对带符号三目混合分支求值有误，会把负数
# 饱和成 +8388607）。与 reference.rshift_round 逐位一致。
_RR_FN = r"""
  function automatic logic signed [23:0] rr(input logic signed [63:0] v, input int n);
    logic signed [63:0] p;
    if (v >= 0) p = (v + (1 <<< (n-1))) >>> n;
    else        p = -( ( (-v) + (1 <<< (n-1)) ) >>> n );
    if (p > 64'sd8388607) rr = 24'sd8388607;
    else if (p < -64'sd8388608) rr = -24'sd8388608;
    else rr = $signed(p[23:0]);
  endfunction
"""

GEMV_TEMPLATE = r'''
// gemv：顺序 GEMV。x 打包输入，yout 打包输出。
// 权重/尺度以 case 常量 ROM 函数内联（无 initial/$readmemh），
// 任意综合器可综合；每个引擎独立文件 gemv_<idx>.sv。
module @@GMOD@@ #(
  parameter C_IN=16, parameter C_OUT=16, parameter SIMD=8, parameter WW=4,
  parameter ACT=24, parameter RS=16, parameter WORDS=32, parameter NUM_BITS=24
)(
  input logic clk, input logic rst_n, input logic en,
  input logic signed [C_IN*ACT-1:0] x,
  output logic signed [C_OUT*ACT-1:0] yout,
  output logic done
);
@@RRFN@@
  // ---------- 权重 / 尺度 ROM（case 常量函数）----------
@@WROM_FN@@
@@NROM_FN@@
  localparam WPR = (C_IN + SIMD - 1) / SIMD;

  function automatic logic signed [63:0] chunk_mac(input int oo, input int cc);
    integer j; logic signed [31:0] w; logic signed [63:0] p;
    logic signed [WW*SIMD-1:0] wi;
    p = 0;
    for (j=0;j<SIMD;j=j+1) begin
      wi = wrom_lut(oo*WPR+cc);
      w = $signed({ {32-WW{ wi[j*WW+WW-1] }}, wi[j*WW +: WW] });
      p = p + w * $signed(x[(cc*SIMD+j)*ACT +: ACT]);
    end
    chunk_mac = p;
  endfunction

  localparam S_IDLE=0, S_MAC=1, S_REQ=2;
  reg [3:0] st;
  reg signed [63:0] acc;
  reg [15:0] o, ch;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin st<=S_IDLE; done<=0; yout<='0; end
    else begin
      done<=0;
      case (st)
        S_IDLE: if (en) begin st<=S_MAC; o<=0; ch<=0; acc<=0; end
        S_MAC: begin
          acc<=acc+chunk_mac(o,ch);
          if (ch==WPR-1) st<=S_REQ; else ch<=ch+1;
        end
        S_REQ: begin
          yout[o*ACT +: ACT] <= rr(acc*$signed({ {64-NUM_BITS{1'b0}}, nrom_lut(o)}), RS);
          if (o==C_OUT-1) begin st<=S_IDLE; done<=1; end
          else begin o<=o+1; ch<=0; acc<=0; st<=S_MAC; end
        end
      endcase
    end
  end
endmodule
'''

GEMV_TEMPLATE_BIAS = r'''
// gemv + bias（GPT-2）：yout[o] = clamp( rr(acc*num, RS) + bias_rom[o] )。
// 权重/尺度/bias 以 case 常量 ROM 函数内联（无 initial/$readmemh）。
module @@GMOD@@ #(
  parameter C_IN=16, parameter C_OUT=16, parameter SIMD=8, parameter WW=4,
  parameter ACT=24, parameter RS=16, parameter WORDS=32, parameter NUM_BITS=24
)(
  input logic clk, input logic rst_n, input logic en,
  input logic signed [C_IN*ACT-1:0] x,
  output logic signed [C_OUT*ACT-1:0] yout,
  output logic done
);
@@RRFN@@
  // ---------- 权重 / 尺度 / bias ROM（case 常量函数）----------
@@WROM_FN@@
@@NROM_FN@@
@@BROM_FN@@
  localparam WPR = (C_IN + SIMD - 1) / SIMD;

  function automatic logic signed [63:0] chunk_mac(input int oo, input int cc);
    integer j; logic signed [31:0] w; logic signed [63:0] p;
    logic signed [WW*SIMD-1:0] wi;
    p = 0;
    for (j=0;j<SIMD;j=j+1) begin
      wi = wrom_lut(oo*WPR+cc);
      w = $signed({ {32-WW{ wi[j*WW+WW-1] }}, wi[j*WW +: WW] });
      p = p + w * $signed(x[(cc*SIMD+j)*ACT +: ACT]);
    end
    chunk_mac = p;
  endfunction

  localparam S_IDLE=0, S_MAC=1, S_REQ=2;
  reg [3:0] st;
  reg signed [63:0] acc;
  reg [15:0] o, ch;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin st<=S_IDLE; done<=0; yout<='0; end
    else begin
      done<=0;
      case (st)
        S_IDLE: if (en) begin st<=S_MAC; o<=0; ch<=0; acc<=0; end
        S_MAC: begin
          acc<=acc+chunk_mac(o,ch);
          if (ch==WPR-1) st<=S_REQ; else ch<=ch+1;
        end
        S_REQ: begin
          begin : bb
            logic signed [63:0] bv;
            bv = rr(acc*$signed({ {64-NUM_BITS{1'b0}}, nrom_lut(o)}), RS)
                 + $signed(brom_lut(o));
            bv = (bv>64'sd8388607)?64'sd8388607:((bv<-64'sd8388608)?-64'sd8388608:bv);
            yout[o*ACT +: ACT] <= bv[ACT-1:0];
          end
          if (o==C_OUT-1) begin st<=S_IDLE; done<=1; end
          else begin o<=o+1; ch<=0; acc<=0; st<=S_MAC; end
        end
      endcase
    end
  end
endmodule
'''

RMSNORM_TEMPLATE = r'''
// rmsnorm：规约 + rsqrt LUT + 逐元素定点乘。x/g 打包，yout 打包。
module rmsnorm #(parameter H=16, ACT=24, F=12)(
  input logic clk, input logic rst_n, input logic en,
  input logic signed [H*ACT-1:0] x, g,
  output logic signed [H*ACT-1:0] yout,
  output logic done
);
@@RRFN@@
  logic signed [ACT-1:0] xa[H], ga[H];
  always_comb begin
    for (int i=0;i<H;i=i+1) begin
      xa[i]=$signed(x[i*ACT +: ACT]);
      ga[i]=$signed(g[i*ACT +: ACT]);
    end
  end
  localparam RMX=@@RSQRT_MAX@@;
  // H 为 2 的幂时用算术右移实现「÷H」（穷举一致，且不产生 $div 除法器）；
  // 否则保留除法表达式（当前全部模型 H∈{8,16}，综合器恒折叠为移位）。
  localparam HAS_SHIFT = (H & (H - 1)) == 0;
  localparam HSH = HAS_SHIFT ? $clog2(H) : 0;
  // ---------- rsqrt LUT（case 常量函数，可综合）----------
@@RSQRT_FN@@

  logic signed [63:0] rsum;
  logic signed [23:0] c;
  logic [15:0] ii;
  reg [2:0] st;
  localparam S0=0, SSUM=1, SMEAN=2, SELEM=3;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin st<=S0; done<=0; end
    else begin
      done<=0;
      case(st)
        S0: if(en) begin st<=SSUM; rsum<=0; ii<=0; end
        SSUM: begin
          if(ii<H) begin
            rsum<=rsum + $signed(xa[ii])*$signed(xa[ii]);
            ii<=ii+1;
          end else st<=SMEAN;
        end
        SMEAN: begin
          begin : um
            logic signed [63:0] mean, idx;
            mean = (HAS_SHIFT ? ((rsum + (H/2)) >>> HSH)
                              : ((rsum + (H/2)) / H)) + 2;
            idx = (mean<1) ? 1 : (mean>RMX) ? RMX : mean;
            c <= $signed(rsqrt_lut(idx[@@RSQRT_IDX@@:0]));
          end
          ii<=0; st<=SELEM;
        end
        SELEM: begin
          begin : sv
            logic signed [63:0] v;
            v = $signed(xa[ii])*$signed(ga[ii])*$signed(c);
            yout[ii*ACT +: ACT] <= rr(v, F);
          end
          if(ii==H-1) begin st<=S0; done<=1; end else ii<=ii+1;
        end
      endcase
    end
  end
endmodule
'''

LAYERNORM_TEMPLATE = r'''
// layernorm（GPT-2）：(x-mean)*rsqrt(1+Σ(x-mean)^2/H)*gamma + beta。x/g/b 打包。
module layernorm #(parameter H=16, ACT=24, F=12)(
  input logic clk, input logic rst_n, input logic en,
  input logic signed [H*ACT-1:0] x, g, b,
  output logic signed [H*ACT-1:0] yout,
  output logic done
);
@@RRFN@@
  logic signed [ACT-1:0] xa[H], ga[H], ba[H];
  always_comb begin
    for (int i=0;i<H;i=i+1) begin
      xa[i]=$signed(x[i*ACT +: ACT]);
      ga[i]=$signed(g[i*ACT +: ACT]);
      ba[i]=$signed(b[i*ACT +: ACT]);
    end
  end
  localparam RMX=@@RSQRT_MAX@@;
  // H 为 2 的幂时用算术右移实现「÷H」（穷举一致，且不产生 $div 除法器）；
  // 否则保留除法表达式（当前全部模型 H∈{8,16}，综合器恒折叠为移位）。
  localparam HAS_SHIFT = (H & (H - 1)) == 0;
  localparam HSH = HAS_SHIFT ? $clog2(H) : 0;
  // ---------- rsqrt LUT（case 常量函数，可综合）----------
@@RSQRT_FN@@

  logic signed [63:0] rsum, rsumc, meanv;
  logic signed [23:0] c;
  logic signed [ACT-1:0] xc[0:H-1];
  logic [15:0] ii;
  reg [2:0] st;
  localparam S0=0, SSUM=1, SMEAN=2, SELEM=3;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin st<=S0; done<=0; end
    else begin
      done<=0;
      case(st)
        S0: if(en) begin st<=SSUM; rsum<=0; ii<=0; end
        SSUM: begin
          rsum<=rsum + $signed(xa[ii]);
          xc[ii]<=xa[ii];
          if(ii==H-1) begin
            begin : mn
              logic signed [63:0] nsum;
              nsum = rsum + $signed(xa[ii]) + (H/2);
              meanv <= HAS_SHIFT ? (nsum >>> HSH)
                    : ((nsum>=0) ? (nsum / H) : ((nsum - (H - 1)) / H));
            end
            st<=SMEAN;
          end else ii<=ii+1;
        end
        SMEAN: begin
          begin : mc
            logic signed [63:0] s2, idx;
            s2=0;
            for (int j=0;j<H;j=j+1) begin
              xc[j]<=$signed(xa[j])-meanv;
              s2=s2 + ($signed(xa[j])-meanv)*($signed(xa[j])-meanv);
            end
            idx = (HAS_SHIFT ? ((s2 + (H/2)) >>> HSH)
                             : ((s2 + (H/2)) / H)) + 1;
            idx = (idx<1) ? 1 : ((idx>RMX) ? RMX : idx);
            c <= $signed(rsqrt_lut(idx[@@RSQRT_IDX@@:0]));
          end
          ii<=0; st<=SELEM;
        end
        SELEM: begin
          begin : sv
            logic signed [63:0] v;
            v = $signed(xc[ii])*$signed(ga[ii])*$signed(c);
            v = (v>=0) ? ((v + (1 <<< (F-1))) >>> F)
                       : -( ( (-v) + (1 <<< (F-1)) ) >>> F );
            v = v + $signed(ba[ii]);
            v = (v>64'sd8388607)?64'sd8388607:((v<-64'sd8388608)?-64'sd8388608:v);
            yout[ii*ACT +: ACT] <= v[ACT-1:0];
          end
          if(ii==H-1) begin st<=S0; done<=1; end else ii<=ii+1;
        end
      endcase
    end
  end
endmodule
'''

ATTN_TEMPLATE = r'''// 单 token 多头注意力：定点 softmax（exp LUT + recip LUT）。qr/kvk/kvv 打包。
module attn #(parameter H=16, HEADS=4, HD=4, SEQ=8, ACT=24, F=12, EF=10, RR=24, K=15)(
  input logic clk, input logic rst_n, input logic en,
  input logic signed [H*ACT-1:0] qr,
  input logic signed [SEQ*H*ACT-1:0] kvk, kvv,
  input logic [15:0] seq,
  output logic signed [H*ACT-1:0] yout,
  output logic done
);
@@RRFN@@
  logic signed [ACT-1:0] qa[H];
  logic signed [ACT-1:0] ka[SEQ][H];
  logic signed [ACT-1:0] va[SEQ][H];
  always_comb begin
    for (int i=0;i<H;i=i+1) begin
      qa[i]=$signed(qr[i*ACT +: ACT]);
      for (int p=0;p<SEQ;p=p+1) begin
        ka[p][i]=$signed(kvk[(p*H+i)*ACT +: ACT]);
        va[p][i]=$signed(kvv[(p*H+i)*ACT +: ACT]);
      end
    end
  end
  // ---------- exp(-x) / 1/x LUT（case 常量函数，可综合）----------
  localparam EXPMAX=@@EXPMAX@@;
  localparam RECMAX=@@RECMAX@@;
@@EXP_FN@@
@@RECIP_FN@@

  function automatic logic signed [63:0] banker(input logic signed [63:0] v, input int k);
    logic signed [63:0] q, rem;
    q   = v >>> k;
    rem = v - (q <<< k);
    if (rem > (1 <<< (k-1))) banker = q + 1;
    else if (rem == (1 <<< (k-1))) banker = (q[0]==0) ? q : q + 1;
    else banker = q;
  endfunction

  reg [4:0] st;
  localparam A0=0, ADOT=1, AEXP=2, ANORM=3, AOUT=4;
  reg [3:0] hh; reg [15:0] p, i;
  logic signed [63:0] me, ssum, rcp;
  logic signed [47:0] sc[0:SEQ-1];
  logic signed [31:0] e[0:SEQ-1];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin st<=A0; done<=0; end
    else begin
      done<=0;
      case(st)
        A0: if(en) begin st<=ADOT; hh<=0; end
        ADOT: begin
          begin : dd
            logic signed [63:0] dot;
            me = -64'sd4611686018427387904;
            for (int p2=0;p2<SEQ;p2=p2+1) begin
              dot=0;
              for (int i2=0;i2<HD;i2=i2+1)
                dot = dot + $signed(ka[p2][hh*HD+i2])*$signed(qa[hh*HD+i2]);
              if (p2 < seq) begin
                sc[p2] = banker(dot, K);
                if (sc[p2]>me) me=sc[p2];
              end else sc[p2]=0;
            end
          end
          p<=0; ssum<=1; st<=AEXP;
        end
        AEXP: begin
          if (p<seq) begin
            begin : ee
              logic signed [63:0] d;
              d = me - sc[p];
              d = (d<0) ? 0 : ((d>@@EXPMAX@@) ? @@EXPMAX@@ : d);
              e[p]   <= expneg_lut(d[15:0]);
              ssum   <= ssum + $signed(expneg_lut(d[15:0]));
            end
          end
          if (p==seq-1) st<=ANORM; else p<=p+1;
        end
        ANORM: begin
          begin : rn
            logic signed [63:0] sid;
            sid = (ssum<1) ? 1 : ((ssum>@@RECMAX@@) ? @@RECMAX@@ : ssum);
            rcp <= $signed(recip_lut(sid[15:0]));
          end
          i<=0; st<=AOUT;
        end
        AOUT: begin
          begin : oo
            logic signed [63:0] num, vv;
            num=0;
            for (int p2=0;p2<SEQ;p2=p2+1)
              if(p2<seq) num = num + $signed(e[p2])*$signed(va[p2][hh*HD+i]);
            vv = num*rcp;
            yout[(hh*HD+i)*ACT +: ACT] <= rr(vv, RR);
          end
          if ((hh==HEADS-1)&&(i==HD-1)) begin st<=A0; done<=1; end
          else if (i==HD-1) begin hh<=hh+1; i<=0; st<=ADOT; end
          else i<=i+1;
        end
      endcase
    end
  end
endmodule
'''


# --------------------------------------------------------------------------


def _build_master(engine_list, cfg):
    """返回 (state_defs_text, case_body_text)。engine_list: [(key,QWeight)]。"""
    LYR = cfg["num_layers"]
    eid_of = {k: i for i, (k, _) in enumerate(engine_list)}
    idx_of = lambda eng: eid_of[f"layers.{eng}"]

    names = ["S_IDLE", "S_DEC", "RMSF", "RMSFW", "OUT", "OUTW", "LOGW", "ADVT", "DONE"]
    for L in range(LYR):
        names += [f"R0{L}", f"R0{L}W",
                  f"q{L}", f"q{L}W", f"k{L}", f"k{L}W", f"v{L}", f"v{L}W",
                  f"RQ{L}", f"RK{L}", f"KV{L}",
                  f"AT{L}", f"AT{L}W", f"AH{L}",
                  f"R1{L}", f"R1{L}W",
                  f"g{L}", f"g{L}W", f"u{L}", f"u{L}W",
                  f"SU{L}", f"MU{L}", f"d{L}", f"d{L}W", f"AHH{L}"]
    sn = {n: i for i, n in enumerate(names)}
    state_defs = "\n".join(f"  localparam {n}={i};" for i, n in enumerate(names))

    def nxt_after_layer(L):
        return f"R0{L+1}" if L + 1 < LYR else "RMSF"

    C = []

    def emit(st, body):
        # body already includes S<=...
        C.append(f"        {sn[st]}: begin {body} end")

    # ---------- IDLE ----------
    kv_reset = "; ".join(
        f"for(int pi=0;pi<H;pi=pi+1) for(int qi=0;qi<SEQ;qi=qi+1) "
        f"begin kv_k[{L}][qi][pi]<='0; kv_v[{L}][qi][pi]<='0; end"
        for L in range(LYR))
    emit("S_IDLE",
         f"if(start) begin tokk<=0; {kv_reset}; S<={sn['S_DEC']}; end "
         f"else S<={sn['S_IDLE']};")

    # ---------- S_DEC ----------
    emit("S_DEC",
         f"for(int i=0;i<H;i=i+1) h[i]<=wte_lut(token_ram[tokk]*H+i); "
         f"S<={sn['R00']};")

    def gemv_en_wait(st_en, idx, st_done_next):
        st_w = st_en + "W"
        emit(st_en, f"gen_{idx}<=1; S<={sn[st_w]};")
        emit(st_w, f"gen_{idx}<=0; if(gd_{idx}) S<={sn[st_done_next]}; "
                   f"else S<={sn[st_w]};")

    def rms_en_wait(st_en, src, gsel, st_done_next, target):
        st_w = st_en + "W"
        emit(st_en, f"rms_en<=1; rms_src<={src}; rms_gsel<={gsel}; "
                    f"S<={sn[st_w]};")
        cap = (f"for(int i=0;i<H;i=i+1) {target}[i]"
               f"<=$signed(rms_yout[i*{ACTW} +: {ACTW}]);") if target else ""
        emit(st_w, f"rms_en<=0; if(rms_done) begin {cap} S<={sn[st_done_next]}; "
                   f"end else S<={sn[st_w]};")

    # ---------- layers ----------
    for L in range(LYR):
        rms_en_wait(f"R0{L}", 0, 2 * L, f"q{L}", "n1")
        gemv_en_wait(f"q{L}", idx_of(f"{L}.q"), f"k{L}")
        gemv_en_wait(f"k{L}", idx_of(f"{L}.k"), f"v{L}")
        gemv_en_wait(f"v{L}", idx_of(f"{L}.v"), f"KV{L}")
        emit(f"RQ{L}", f"S<={sn[f'RK{L}']};")
        emit(f"RK{L}", f"S<={sn[f'KV{L}']};")
        emit(f"KV{L}", (f"for(int i=0;i<H;i=i+1) begin "
                        f"kv_k[{L}][tokk][i]<=kr[i]; kv_v[{L}][tokk][i]<=vvec[i]; "
                        f"end S<={sn[f'AT{L}']};"))
        emit(f"AT{L}", f"att_en<=1; att_layer<={L}; att_seq<=tokk+1; "
                       f"S<={sn[f'AT{L}W']};")
        emit(f"AT{L}W",
             f"att_en<=0; if(att_done) begin "
             f"for(int i=0;i<H;i=i+1) att[i]<=$signed(att_yout[i*{ACTW} +: {ACTW}]); "
             f"S<={sn[f'AH{L}']}; end else S<={sn[f'AT{L}W']};")
        emit(f"AH{L}", f"S<={sn[f'R1{L}']};")
        rms_en_wait(f"R1{L}", 1, 2 * L + 1, f"g{L}", "n2")
        gemv_en_wait(f"g{L}", idx_of(f"{L}.g"), f"u{L}")
        gemv_en_wait(f"u{L}", idx_of(f"{L}.u"), f"d{L}")
        emit(f"SU{L}", f"S<={sn[f'MU{L}']};")
        emit(f"MU{L}", f"S<={sn[f'd{L}']};")
        gemv_en_wait(f"d{L}", idx_of(f"{L}.d"), f"AHH{L}")
        nl = nxt_after_layer(L)
        emit(f"AHH{L}",
             f"for(int i=0;i<H;i=i+1) begin "
             f"logic signed [63:0] sa, sc; "
             f"sa=$signed(h1[i])+$signed(dd[i]); "
             f"sc=(sa>64'sd8388607)?64'sd8388607:((sa<-64'sd8388608)?-64'sd8388608:sa); "
             f"h[i]<=sc[ACT-1:0]; "
             f"end S<={sn[nl]};")

    # ---------- final ----------
    rms_en_wait("RMSF", 0, 2 * LYR, "OUT", "nf")
    out_idx = eid_of["output_proj"]
    gemv_en_wait("OUT", out_idx, "LOGW")
    emit("LOGW",
         f"for(int vi=0;vi<VOCAB;vi=vi+1) logit_bank[tokk][vi]<=outvec[vi]; "
         f"S<={sn['ADVT']};")
    emit("ADVT",
         f"if(tokk==SEQ-1) S<={sn['DONE']}; else begin tokk<=tokk+1; "
         f"S<={sn['S_DEC']}; end")
    emit("DONE", f"done<=1; S<={sn['S_IDLE']};")

    case_body = "\n".join(C)
    return state_defs, case_body


_TOP_TEMPLATE = r'''
`timescale 1ns/1ps
// =====================================================================
// @@MODNAME@@ : 预填充（prefill）整数推理加速器
// 与 llm2asic.rtl_backend.reference.IntModel 逐位一致。
// =====================================================================
module @@MODNAME@@ #(
  parameter H=@@H@@, parameter HEADS=@@HEADS@@, parameter HD=@@HD@@,
  parameter LYR=@@LYR@@, parameter SEQ=@@SEQ@@, parameter VOCAB=@@VOCAB@@,
  parameter F=@@F@@, parameter RS=@@RS@@, parameter RR=@@RR@@, parameter ACT=@@ACT@@
)(
  input logic clk, input logic rst_n, input logic start,
  // 展平（flat）端口：token_ram[i] <=> token_flat[i*16 +: 16]
  input logic [SEQ*16-1:0] token_flat,
  output logic done,
  // logit_bank[t][v] <=> logit_flat[(t*VOCAB+v)*28 +: 28]
  output logic signed [SEQ*VOCAB*28-1:0] logit_flat
);
  // ---------- 端口解包（unpacked 视图）----------
  logic [15:0] token_ram [0:SEQ-1];
  logic signed [27:0] logit_bank [0:SEQ-1][0:VOCAB-1];
  genvar gt_f, gv_f, gi_f;
  generate
    for (gi_f=0; gi_f<SEQ; gi_f=gi_f+1) begin : gtok
      assign token_ram[gi_f] = token_flat[gi_f*16 +: 16];
    end
    for (gt_f=0; gt_f<SEQ; gt_f=gt_f+1) begin : glog_t
      for (gv_f=0; gv_f<VOCAB; gv_f=gv_f+1) begin : glog_v
        assign logit_flat[(gt_f*VOCAB+gv_f)*28 +: 28] = logit_bank[gt_f][gv_f];
      end
    end
  endgenerate

  // ---------- 激活向量 ----------
  logic signed [ACT-1:0] h   [0:H-1];
  logic signed [ACT-1:0] n1  [0:H-1];
  logic signed [ACT-1:0] n2  [0:H-1];
  logic signed [ACT-1:0] nf  [0:H-1];
  logic signed [ACT-1:0] qvec[0:H-1];
  logic signed [ACT-1:0] kvec[0:H-1];
  logic signed [ACT-1:0] vvec[0:H-1];
  logic signed [ACT-1:0] qr  [0:H-1];
  logic signed [ACT-1:0] kr  [0:H-1];
  logic signed [ACT-1:0] att [0:H-1];
  logic signed [ACT-1:0] gvec[0:H-1];
  logic signed [ACT-1:0] uvec[0:H-1];
  logic signed [ACT-1:0] us  [0:H-1];
  logic signed [ACT-1:0] mm  [0:H-1];
  logic signed [ACT-1:0] dd  [0:H-1];
  logic signed [ACT-1:0] outvec[0:VOCAB-1];
  logic signed [ACT-1:0] h1  [0:H-1];
  always_comb begin
    for (int i=0;i<H;i=i+1) begin
      logic signed [63:0] sa, sc;
      sa = $signed(h[i]) + $signed(att[i]);
      sc = (sa>64'sd8388607) ? 64'sd8388607
         : ((sa<-64'sd8388608) ? -64'sd8388608 : sa);
      h1[i] = sc[ACT-1:0];
    end
  end

  // ---------- KV 缓存 ----------
  logic signed [ACT-1:0] kv_k [0:LYR-1][0:SEQ-1][0:H-1];
  logic signed [ACT-1:0] kv_v [0:LYR-1][0:SEQ-1][0:H-1];
  // token 位置（master 维护）
  reg [15:0] tokk;

  // ---------- 嵌入 ROM（case 常量函数，可综合）----------
@@WTE_FN@@

  // ---------- gamma ROM ----------
@@GAMMA_ROMS@@

  // ---------- silu LUT（case 常量函数）----------
  localparam SILU_LO=@@SILU_LO@@;
@@SILU_FN@@

  // ---------- rope cos/sin 常量（case 常量函数）----------
@@ROPE_FNS@@

  // ---------- gemv 实例 ----------
@@GEMV_DECLS@@
@@GEMV_INSTS@@
@@GEMV_DRIVES@@
@@GEMV_LATCH@@

  // ---------- rmsnorm 实例 ----------
  logic rms_en, rms_done;
  logic [3:0] rms_src;
  logic [5:0] rms_gsel;
  logic signed [H*ACT-1:0] rms_xin, rms_gin, rms_yout;
  rmsnorm #(.H(H),.ACT(ACT),.F(F)) u_rms (
    .clk(clk),.rst_n(rst_n),.en(rms_en),.x(rms_xin),.g(rms_gin),
    .yout(rms_yout),.done(rms_done));
  always_comb begin
    for (int i=0;i<H;i=i+1)
      case (rms_src)
        0: rms_xin[i*ACT +: ACT] = h[i];
        1: rms_xin[i*ACT +: ACT] = h1[i];
        default: rms_xin[i*ACT +: ACT] = h[i];
      endcase
    case (rms_gsel)
@@RMS_GSEL@@
      default: rms_gin = '0;
    endcase
  end

  // ---------- attn 实例 ----------
  logic att_en, att_done;
  logic [15:0] att_seq;
  logic [7:0] att_layer;
  logic signed [H*ACT-1:0] att_qr;
  logic signed [SEQ*H*ACT-1:0] att_kvk, att_kvv;
  logic signed [H*ACT-1:0] att_yout;
  attn #(.H(H),.HEADS(HEADS),.HD(HD),.SEQ(SEQ),.ACT(ACT),.F(F),.EF(10),.RR(RR),.K(15))
    u_att (.clk(clk),.rst_n(rst_n),.en(att_en),.qr(att_qr),.kvk(att_kvk),
           .kvv(att_kvv),.seq(att_seq),.yout(att_yout),.done(att_done));
  always_comb begin
    for (int i=0;i<H;i=i+1) att_qr[i*ACT +: ACT] = qr[i];
    for (int p=0;p<SEQ;p=p+1)
      for (int i=0;i<H;i=i+1) begin
        att_kvk[(p*H+i)*ACT +: ACT] = kv_k[att_layer][p][i];
        att_kvv[(p*H+i)*ACT +: ACT] = kv_v[att_layer][p][i];
      end
  end

  // ---------- rope（组合，饱和）----------
  always_comb begin
    logic signed [63:0] ar, br, tr, tc, cc, ss;
    for (int start=0; start<H; start=start+HD)
      for (int j=0;j<HD/2;j=j+1) begin
        cc = rope_c_lut(tokk*(HD/2)+j);
        ss = rope_s_lut(tokk*(HD/2)+j);
        ar = $signed(qvec[start+j]); br = $signed(qvec[start+j+HD/2]);
        tr = (ar*cc - br*ss + (1 <<< (F-1))) >>> F;
        tc = (tr>64'sd8388607)?64'sd8388607:((tr<-64'sd8388608)?-64'sd8388608:tr);
        qr[start+j]   = tc[23:0];
        tr = (ar*ss + br*cc + (1 <<< (F-1))) >>> F;
        tc = (tr>64'sd8388607)?64'sd8388607:((tr<-64'sd8388608)?-64'sd8388608:tr);
        qr[start+j+HD/2] = tc[23:0];
        ar = $signed(kvec[start+j]); br = $signed(kvec[start+j+HD/2]);
        tr = (ar*cc - br*ss + (1 <<< (F-1))) >>> F;
        tc = (tr>64'sd8388607)?64'sd8388607:((tr<-64'sd8388608)?-64'sd8388608:tr);
        kr[start+j]   = tc[23:0];
        tr = (ar*ss + br*cc + (1 <<< (F-1))) >>> F;
        tc = (tr>64'sd8388607)?64'sd8388607:((tr<-64'sd8388608)?-64'sd8388608:tr);
        kr[start+j+HD/2] = tc[23:0];
      end
  end

  // ---------- silu / mul（组合）----------
  always_comb begin
    for (int i=0;i<H;i=i+1) begin
      begin : sb
        logic signed [63:0] xi; logic signed [23:0] u;
        integer sidx;
        xi = $signed(uvec[i]);
        xi = (xi<SILU_LO) ? SILU_LO : ((xi>@@SILU_HI@@) ? @@SILU_HI@@ : xi);
        sidx = xi - SILU_LO;
        u  = silu_lut(sidx);
        us[i] = u;
      end
      begin : mb
        logic signed [63:0] vv, vc;
        vv = $signed(gvec[i])*$signed(us[i]);
        vv = (vv>=0) ? ((vv + (1 <<< (F-1))) >>> F)
                     : -( ( (-vv) + (1 <<< (F-1)) ) >>> F );
        vc = (vv>64'sd8388607)?64'sd8388607:((vv<-64'sd8388608)?-64'sd8388608:vv);
        mm[i] = vc[23:0];
      end
    end
  end

  // ---------- master FSM ----------
  reg [7:0] S;
@@STATE_DEFS@@

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      S<=S_IDLE; done<=0; tokk<=0; rms_en<=0; att_en<=0; att_layer<=0;
    end else begin
      done<=0; rms_en<=0; att_en<=0;
      case (S)
@@MASTER_CASE@@
      endcase
    end
  end

endmodule
'''


def _emit_top(qmodel, cfg: dict) -> str:
    H = cfg["hidden"]; HEADS = cfg["num_heads"]; HD = cfg["head_dim"]
    LYR = cfg["num_layers"]; SEQ = cfg["max_seq_len"]; VOCAB = cfg["vocab_size"]
    simd = qmodel.simd; ww = qmodel.bit_width

    engine_list = sorted(qmodel.engines.items())
    eid_of = {k: i for i, (k, _) in enumerate(engine_list)}

    target_of = {}
    inp_of = {}
    for L in range(LYR):
        for nm in ["q", "k", "v"]:
            target_of[f'layers.{L}.{nm}'] = nm + "vec"
            inp_of[f'layers.{L}.{nm}'] = 'n1'
        for nm in ["g", "u"]:
            target_of[f'layers.{L}.{nm}'] = nm + "vec"
            inp_of[f'layers.{L}.{nm}'] = 'n2'
        target_of[f'layers.{L}.d'] = 'dd'
        inp_of[f'layers.{L}.d'] = 'mm'
    target_of['output_proj'] = 'outvec'
    inp_of['output_proj'] = 'nf'

    # gemv instances + pack input (genvar) + latch output
    gemv_insts, gemv_drives, gemv_latch, gemv_decls = [], [], [], []
    target_latch = {}
    for idx, (key, qw) in enumerate(engine_list):
        c_in, c_out = qw.c_in, qw.c_out
        gemv_decls.append(
            f"  logic gen_{idx}, gd_{idx};\n"
            f"  logic signed [{c_in*ACTW-1}:0] gx_{idx};\n"
            f"  logic signed [{c_out*ACTW-1}:0] gy_{idx};")
        words = c_out * ((c_in + simd - 1) // simd)
        gemv_insts.append(
            f"  gemv_{idx} #(.C_IN({c_in}),.C_OUT({c_out}),.SIMD({simd}),.WW({ww}),.ACT({ACTW}),"
            f".RS({REQUANT_S}),.WORDS({words}),.NUM_BITS(24)) "
            f"u_gv{idx}(.clk(clk),.rst_n(rst_n),.en(gen_{idx}),"
            f".x(gx_{idx}),.yout(gy_{idx}),.done(gd_{idx}));")
        src = inp_of[key]
        gemv_drives.append(
            f"  for (genvar G{idx}=0; G{idx}<{H}; G{idx}=G{idx}+1) "
            f"assign gx_{idx}[G{idx}*{ACTW} +: {ACTW}] = {src}[G{idx}];")
        tgt = target_of[key]; c_out = qw.c_out
        target_latch.setdefault(tgt, []).append((idx, c_out))

    _latch_blocks = []
    for tgt, items in target_latch.items():
        parts = []
        for k, (idx, c_out) in enumerate(items):
            kw = "if" if k == 0 else "else if"
            parts.append(f"{kw} (gd_{idx}) begin for (int ii=0; ii<{c_out}; ii=ii+1) {tgt}[ii] <= gy_{idx}[ii*{ACTW} +: {ACTW}]; end")
        parts.append("else ;")
        _latch_blocks.append("  always_ff @(posedge clk) begin " + " ".join(parts) + " end")
    gemv_latch = _latch_blocks

    # gamma ROMs（case 常量函数，可综合）
    gamma_fns = "\n".join(
        _case_rom_fn(f"{_san(k)}_g_lut", _hex_patterns(v.reshape(-1)))
        for k, v in qmodel.gammas.items())

    # rope cos/sin 常量函数
    cq, sq = _precompute_rope_cs(cfg)
    nc = SEQ * (HD // 2)
    rope_fns = "\n".join([
        _case_rom_fn("rope_c_lut",
                     _hex_patterns([cq[x // (HD // 2)][x % (HD // 2)]
                                    for x in range(nc)])),
        _case_rom_fn("rope_s_lut",
                     _hex_patterns([sq[x // (HD // 2)][x % (HD // 2)]
                                    for x in range(nc)])),
    ])

    # wte / silu 常量函数
    wte_fn = _case_rom_fn("wte_lut", _hex_patterns(qmodel.wte_q.reshape(-1)))
    silu_fn = _case_rom_fn("silu_lut", _hex_patterns(qmodel.luts.silu_gate))

    # rms_gin gamma select case
    rms_gsel = []
    norm_keys = []
    for L in range(LYR):
        norm_keys.append(f'layers.{L}.input_layernorm.gamma')
        norm_keys.append(f'layers.{L}.post_attention_layernorm.gamma')
    norm_keys.append('final_norm.gamma')
    for gi, gk in enumerate(norm_keys):
        arr = qmodel.gammas[gk]
        body = " ".join(
            f"rms_gin[{i}*{ACTW} +: {ACTW}] = {_san(gk)}_g_lut({i});"
            for i in range(H))
        rms_gsel.append(f"      {gi}: begin {body} end")
    rms_gsel = "\n".join(rms_gsel)

    state_defs, master_case = _build_master(engine_list, cfg)

    r = _TOP_TEMPLATE
    repl = {
        "MODNAME": f"{cfg['name']}_accel",
        "H": H, "HEADS": HEADS, "HD": HD, "LYR": LYR, "SEQ": SEQ, "VOCAB": VOCAB,
        "F": F, "RS": REQUANT_S, "RR": RR_DIV, "ACT": ACTW,
        "SILU_LO": -(1 << (cfg.get('silu_input_bits', qmodel.luts.silu_input_bits) - 1)),
        "SILU_HI": (1 << (cfg.get('silu_input_bits', qmodel.luts.silu_input_bits) - 1)) - 1,
    }
    for k, v in repl.items():
        r = r.replace(f"@@{k}@@", str(v))
    # computed spans
    lo = -(1 << (qmodel.luts.silu_input_bits - 1))
    hi = (1 << (qmodel.luts.silu_input_bits - 1)) - 1
    r = r.replace("@@SILU_SPAN@@", str(hi - lo))
    r = r.replace("@@GAMMA_ROMS@@", gamma_fns)
    r = r.replace("@@ROPE_FNS@@", rope_fns)
    r = r.replace("@@WTE_FN@@", wte_fn)
    r = r.replace("@@SILU_FN@@", silu_fn)
    r = r.replace("@@GEMV_DECLS@@", "\n".join(gemv_decls))
    r = r.replace("@@GEMV_INSTS@@", "\n".join(gemv_insts))
    r = r.replace("@@GEMV_DRIVES@@", "\n".join(gemv_drives))
    r = r.replace("@@GEMV_LATCH@@", "\n".join(gemv_latch))
    r = r.replace("@@RMS_GSEL@@", rms_gsel)
    r = r.replace("@@STATE_DEFS@@", state_defs)
    r = r.replace("@@MASTER_CASE@@", master_case)
    r = r.replace(
        "S<=S_IDLE; done<=0; tokk<=0; rms_en<=0; att_en<=0; att_layer<=0;",
        "S<=S_IDLE; done<=0; tokk<=0; rms_en<=0; att_en<=0; att_layer<=0;"
        + "".join(f" gen_{i}<=0;" for i in range(len(engine_list))))
    return r


# --------------------------------------------------------------------------
# GPT-2 RTL（LayerNorm + GELU + 绝对位置嵌入，无 RoPE；linear 带 bias）
# --------------------------------------------------------------------------

def _build_master_gpt2(engine_list, cfg):
    """GPT-2 master FSM。engine_list: [(key,QWeight)]。"""
    LYR = cfg["num_layers"]
    eid_of = {k: i for i, (k, _) in enumerate(engine_list)}
    idx_of = lambda eng: eid_of[f"layers.{eng}"]

    names = ["S_IDLE", "S_DEC", "RMSF", "RMSFW", "OUT", "OUTW", "LOGW", "ADVT", "DONE"]
    for L in range(LYR):
        names += [f"R0{L}", f"R0{L}W",
                  f"q{L}", f"q{L}W", f"k{L}", f"k{L}W", f"v{L}", f"v{L}W",
                  f"KV{L}", f"AT{L}", f"AT{L}W",
                  f"o{L}", f"o{L}W", f"AH{L}",
                  f"R1{L}", f"R1{L}W",
                  f"fc{L}", f"fc{L}W", f"GE{L}",
                  f"pr{L}", f"pr{L}W", f"AHH{L}"]
    sn = {n: i for i, n in enumerate(names)}
    state_defs = "\n".join(f"  localparam {n}={i};" for i, n in enumerate(names))

    def nxt_after_layer(L):
        return f"R0{L+1}" if L + 1 < LYR else "RMSF"

    C = []
    def emit(st, body):
        C.append(f"        {sn[st]}: begin {body} end")

    kv_reset = "; ".join(
        f"for(int pi=0;pi<H;pi=pi+1) for(int qi=0;qi<SEQ;qi=qi+1) "
        f"begin kv_k[{L}][qi][pi]<='0; kv_v[{L}][qi][pi]<='0; end"
        for L in range(LYR))
    emit("S_IDLE", f"if(start) begin tokk<=0; {kv_reset}; S<={sn['S_DEC']}; end "
                   f"else S<={sn['S_IDLE']};")
    # 位置嵌入：h[i] = embed_rom[token] + wpe_rom[tokk*H+i]
    emit("S_DEC",
         f"for(int i=0;i<H;i=i+1) h[i]<="
         f"$signed(wte_lut(token_ram[tokk]*H+i))+$signed(wpe_lut(tokk*H+i)); "
         f"S<={sn['R00']};")

    def gemv_en_wait(st_en, idx, st_done_next):
        st_w = st_en + "W"
        emit(st_en, f"gen_{idx}<=1; S<={sn[st_w]};")
        emit(st_w, f"gen_{idx}<=0; if(gd_{idx}) S<={sn[st_done_next]}; "
                   f"else S<={sn[st_w]};")

    def ln_en_wait(st_en, gsel, bsel, xsel, st_done_next, target):
        st_w = st_en + "W"
        emit(st_en, f"ln_en<=1; ln_gsel<={gsel}; ln_bsel<={bsel}; "
                    f"ln_xsel<={xsel}; S<={sn[st_w]};")
        emit(st_w, f"ln_en<=0; if(ln_done) begin "
                   f"for(int i=0;i<H;i=i+1) {target}[i]<="
                   f"$signed(ln_yout[i*{ACTW} +: {ACTW}]); S<={sn[st_done_next]}; "
                   f"end else S<={sn[st_w]};")

    for L in range(LYR):
        # ln_1: gsel=4L, bsel=4L+1（norm 数组排序：每层 ln_1.g/b, ln_2.g/b）
        ln_en_wait(f"R0{L}", 4 * L, 4 * L + 1, 0, f"q{L}", "n1")
        gemv_en_wait(f"q{L}", idx_of(f"{L}.q"), f"k{L}")
        gemv_en_wait(f"k{L}", idx_of(f"{L}.k"), f"v{L}")
        gemv_en_wait(f"v{L}", idx_of(f"{L}.v"), f"KV{L}")
        emit(f"KV{L}", (f"for(int i=0;i<H;i=i+1) begin "
                        f"kv_k[{L}][tokk][i]<=kvec[i]; kv_v[{L}][tokk][i]<=vvec[i]; "
                        f"end S<={sn[f'AT{L}']};"))
        emit(f"AT{L}", f"att_en<=1; att_layer<={L}; att_seq<=tokk+1; "
                       f"S<={sn[f'AT{L}W']};")
        emit(f"AT{L}W", f"att_en<=0; if(att_done) begin "
                        f"for(int i=0;i<H;i=i+1) att[i]<="
                        f"$signed(att_yout[i*{ACTW} +: {ACTW}]); S<={sn[f'o{L}']}; "
                        f"end else S<={sn[f'AT{L}W']};")
        gemv_en_wait(f"o{L}", idx_of(f"{L}.o"), f"AH{L}")
        emit(f"AH{L}", (f"for(int i=0;i<H;i=i+1) begin "
                        f"logic signed [63:0] sa, sc; "
                        f"sa=$signed(h[i])+$signed(ovec[i]); "
                        f"sc=(sa>64'sd8388607)?64'sd8388607:"
                        f"((sa<-64'sd8388608)?-64'sd8388608:sa); "
                        f"h1[i]<=sc[ACT-1:0]; end S<={sn[f'R1{L}']};"))
        ln_en_wait(f"R1{L}", 4 * L + 2, 4 * L + 3, 1, f"fc{L}", "n2")
        gemv_en_wait(f"fc{L}", idx_of(f"{L}.fc"), f"GE{L}")
        # GELU（组合，1 拍）：fcvec -> gvec
        emit(f"GE{L}", (f"for(int i=0;i<NINNER;i=i+1) begin "
                        f"logic signed [63:0] xi; integer sidx; "
                        f"xi=$signed(fcvec[i]); "
                        f"xi=(xi<GELU_LO)?GELU_LO:((xi>GELU_HI)?GELU_HI:xi); "
                        f"sidx=xi-GELU_LO; gvec[i]<=gelu_lut(sidx); "
                        f"end S<={sn[f'pr{L}']};"))
        gemv_en_wait(f"pr{L}", idx_of(f"{L}.proj"), f"AHH{L}")
        nl = nxt_after_layer(L)
        emit(f"AHH{L}", (f"for(int i=0;i<H;i=i+1) begin "
                         f"logic signed [63:0] sa, sc; "
                         f"sa=$signed(h1[i])+$signed(pvec[i]); "
                         f"sc=(sa>64'sd8388607)?64'sd8388607:"
                         f"((sa<-64'sd8388608)?-64'sd8388608:sa); "
                         f"h[i]<=sc[ACT-1:0]; end S<={sn[nl]};"))

    # final norm（gsel=4*LYR, bsel=4*LYR+1）
    ln_en_wait("RMSF", 4 * LYR, 4 * LYR + 1, 0, "OUT", "nf")
    out_idx = eid_of["output_proj"]
    gemv_en_wait("OUT", out_idx, "LOGW")
    emit("LOGW", f"for(int vi=0;vi<VOCAB;vi=vi+1) logit_bank[tokk][vi]<=outvec[vi]; "
                 f"S<={sn['ADVT']};")
    emit("ADVT", f"if(tokk==SEQ-1) S<={sn['DONE']}; else begin tokk<=tokk+1; "
                 f"S<={sn['S_DEC']}; end")
    emit("DONE", f"done<=1; S<={sn['S_IDLE']};")

    return state_defs, "\n".join(C)


_TOP_TEMPLATE_GPT2 = r'''
`timescale 1ns/1ps
// =====================================================================
// @@MODNAME@@ : GPT-2 预填充（prefill）整数推理加速器
// LayerNorm + GELU + 绝对位置嵌入（无 RoPE）；linear 带 bias。
// 与 llm2asic.rtl_backend.reference.IntModel.run_decode_step_gpt2 逐位一致。
// =====================================================================
module @@MODNAME@@ #(
  parameter H=@@H@@, parameter HEADS=@@HEADS@@, parameter HD=@@HD@@,
  parameter LYR=@@LYR@@, parameter SEQ=@@SEQ@@, parameter VOCAB=@@VOCAB@@,
  parameter NINNER=@@NINNER@@,
  parameter F=@@F@@, parameter RS=@@RS@@, parameter RR=@@RR@@, parameter ACT=@@ACT@@
)(
  input logic clk, input logic rst_n, input logic start,
  input logic [SEQ*16-1:0] token_flat,
  output logic done,
  output logic signed [SEQ*VOCAB*28-1:0] logit_flat
);
  // ---------- 端口解包 ----------
  logic [15:0] token_ram [0:SEQ-1];
  logic signed [27:0] logit_bank [0:SEQ-1][0:VOCAB-1];
  genvar gt_f, gv_f, gi_f;
  generate
    for (gi_f=0; gi_f<SEQ; gi_f=gi_f+1) begin : gtok
      assign token_ram[gi_f] = token_flat[gi_f*16 +: 16];
    end
    for (gt_f=0; gt_f<SEQ; gt_f=gt_f+1) begin : glog_t
      for (gv_f=0; gv_f<VOCAB; gv_f=gv_f+1) begin : glog_v
        assign logit_flat[(gt_f*VOCAB+gv_f)*28 +: 28] = logit_bank[gt_f][gv_f];
      end
    end
  endgenerate

  // ---------- 激活向量 ----------
  logic signed [ACT-1:0] h   [0:H-1];
  logic signed [ACT-1:0] n1  [0:H-1];
  logic signed [ACT-1:0] n2  [0:H-1];
  logic signed [ACT-1:0] nf  [0:H-1];
  logic signed [ACT-1:0] qvec[0:H-1];
  logic signed [ACT-1:0] kvec[0:H-1];
  logic signed [ACT-1:0] vvec[0:H-1];
  logic signed [ACT-1:0] att [0:H-1];
  logic signed [ACT-1:0] ovec[0:H-1];
  logic signed [ACT-1:0] fcvec[0:NINNER-1];
  logic signed [ACT-1:0] gvec[0:NINNER-1];
  logic signed [ACT-1:0] pvec[0:H-1];
  logic signed [ACT-1:0] outvec[0:VOCAB-1];
  logic signed [ACT-1:0] h1  [0:H-1];

  // ---------- KV 缓存 ----------
  logic signed [ACT-1:0] kv_k [0:LYR-1][0:SEQ-1][0:H-1];
  logic signed [ACT-1:0] kv_v [0:LYR-1][0:SEQ-1][0:H-1];
  reg [15:0] tokk;

  // ---------- 嵌入 + 位置嵌入 ROM（case 常量函数，可综合）----------
@@WTE_FN@@
@@WPE_FN@@

  // ---------- 归一化（gamma/beta）ROM（case 常量函数）----------
@@NORM_ROMS@@

  // ---------- gelu LUT ----------
  localparam GELU_LO=@@GELU_LO@@;
  localparam GELU_HI=@@GELU_HI@@;
  // ---------- gelu LUT（case 常量函数）----------
@@GELU_FN@@

  // ---------- gemv 实例 ----------
@@GEMV_DECLS@@
@@GEMV_INSTS@@
@@GEMV_DRIVES@@
@@GEMV_LATCH@@

  // ---------- layernorm 实例 ----------
  logic ln_en, ln_done;
  logic [5:0] ln_gsel, ln_bsel;
  logic ln_xsel;
  logic signed [H*ACT-1:0] ln_xin, ln_gin, ln_bin, ln_yout;
  layernorm #(.H(H),.ACT(ACT),.F(F)) u_ln (
    .clk(clk),.rst_n(rst_n),.en(ln_en),.x(ln_xin),.g(ln_gin),.b(ln_bin),
    .yout(ln_yout),.done(ln_done));
  always_comb begin
    for (int i=0;i<H;i=i+1)
      case (ln_xsel)
        1: ln_xin[i*ACT +: ACT] = h1[i];
        default: ln_xin[i*ACT +: ACT] = h[i];
      endcase
    case (ln_gsel)
@@LN_GSEL@@
      default: ln_gin = '0;
    endcase
    case (ln_bsel)
@@LN_BSEL@@
      default: ln_bin = '0;
    endcase
  end

  // ---------- attn 实例 ----------
  logic att_en, att_done;
  logic [15:0] att_seq;
  logic [7:0] att_layer;
  logic signed [H*ACT-1:0] att_qr;
  logic signed [SEQ*H*ACT-1:0] att_kvk, att_kvv;
  logic signed [H*ACT-1:0] att_yout;
  attn #(.H(H),.HEADS(HEADS),.HD(HD),.SEQ(SEQ),.ACT(ACT),.F(F),.EF(10),.RR(RR),.K(15))
    u_att (.clk(clk),.rst_n(rst_n),.en(att_en),.qr(att_qr),.kvk(att_kvk),
           .kvv(att_kvv),.seq(att_seq),.yout(att_yout),.done(att_done));
  always_comb begin
    for (int i=0;i<H;i=i+1) att_qr[i*ACT +: ACT] = qvec[i];
    for (int p=0;p<SEQ;p=p+1)
      for (int i=0;i<H;i=i+1) begin
        att_kvk[(p*H+i)*ACT +: ACT] = kv_k[att_layer][p][i];
        att_kvv[(p*H+i)*ACT +: ACT] = kv_v[att_layer][p][i];
      end
  end

  // ---------- master FSM ----------
  reg [7:0] S;
@@STATE_DEFS@@

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      S<=S_IDLE; done<=0; tokk<=0; ln_en<=0; ln_xsel<=0; att_en<=0; att_layer<=0;
    end else begin
      done<=0; ln_en<=0; att_en<=0;
      case (S)
@@MASTER_CASE@@
      endcase
    end
  end

endmodule
'''


def _emit_top_gpt2(qmodel, cfg: dict) -> str:
    H = cfg["hidden"]; HEADS = cfg["num_heads"]; HD = cfg["head_dim"]
    LYR = cfg["num_layers"]; SEQ = cfg["max_seq_len"]; VOCAB = cfg["vocab_size"]
    simd = qmodel.simd; ww = qmodel.bit_width

    engine_list = sorted(qmodel.engines.items())
    eid_of = {k: i for i, (k, _) in enumerate(engine_list)}

    target_of = {}
    inp_of = {}
    for L in range(LYR):
        for nm in ["q", "k", "v"]:
            target_of[f'layers.{L}.{nm}'] = nm + "vec"
            inp_of[f'layers.{L}.{nm}'] = 'n1'
        target_of[f'layers.{L}.o'] = 'ovec'
        inp_of[f'layers.{L}.o'] = 'att'
        target_of[f'layers.{L}.fc'] = 'fcvec'
        inp_of[f'layers.{L}.fc'] = 'n2'
        target_of[f'layers.{L}.proj'] = 'pvec'
        inp_of[f'layers.{L}.proj'] = 'gvec'
    target_of['output_proj'] = 'outvec'
    inp_of['output_proj'] = 'nf'

    gemv_insts, gemv_drives, gemv_latch, gemv_decls = [], [], [], []
    target_latch = {}
    for idx, (key, qw) in enumerate(engine_list):
        c_in, c_out = qw.c_in, qw.c_out
        gemv_decls.append(
            f"  logic gen_{idx}, gd_{idx};\n"
            f"  logic signed [{c_in*ACTW-1}:0] gx_{idx};\n"
            f"  logic signed [{c_out*ACTW-1}:0] gy_{idx};")
        words = c_out * ((c_in + simd - 1) // simd)
        gemv_insts.append(
            f"  gemv_{idx} #(.C_IN({c_in}),.C_OUT({c_out}),.SIMD({simd}),.WW({ww}),.ACT({ACTW}),"
            f".RS({REQUANT_S}),.WORDS({words}),.NUM_BITS(24)) "
            f"u_gv{idx}(.clk(clk),.rst_n(rst_n),.en(gen_{idx}),"
            f".x(gx_{idx}),.yout(gy_{idx}),.done(gd_{idx}));")
        src = inp_of[key]
        gemv_drives.append(
            f"  for (genvar G{idx}=0; G{idx}<{c_in}; G{idx}=G{idx}+1) "
            f"assign gx_{idx}[G{idx}*{ACTW} +: {ACTW}] = {src}[G{idx}];")
        tgt = target_of[key]
        target_latch.setdefault(tgt, []).append((idx, c_out))

    _latch_blocks = []
    for tgt, items in target_latch.items():
        parts = []
        for k, (idx, c_out) in enumerate(items):
            kw = "if" if k == 0 else "else if"
            parts.append(f"{kw} (gd_{idx}) begin for (int ii=0; ii<{c_out}; ii=ii+1) "
                         f"{tgt}[ii] <= gy_{idx}[ii*{ACTW} +: {ACTW}]; end")
        parts.append("else ;")
        _latch_blocks.append("  always_ff @(posedge clk) begin "
                             + " ".join(parts) + " end")
    gemv_latch = _latch_blocks

    # 归一化：gsel/bsel 排序 = 每层 [ln_1.g, ln_1.b, ln_2.g, ln_2.b]，再 final [g,b]
    norm_keys = []
    for L in range(LYR):
        norm_keys += [f'layers.{L}.ln_1.gamma', f'layers.{L}.ln_1.beta',
                      f'layers.{L}.ln_2.gamma', f'layers.{L}.ln_2.beta']
    norm_keys += ['final_norm.gamma', 'final_norm.beta']
    norm_fns = "\n".join(
        _case_rom_fn(f"{_san(k)}_g_lut", _hex_patterns(v.reshape(-1)))
        for k, v in qmodel.gammas.items())
    wte_fn = _case_rom_fn("wte_lut", _hex_patterns(qmodel.wte_q.reshape(-1)))
    wpe_fn = _case_rom_fn("wpe_lut", _hex_patterns(qmodel.wpe_q.reshape(-1)))
    gelu_fn = _case_rom_fn("gelu_lut", _hex_patterns(qmodel.luts.gelu))

    # gamma / beta 选择器
    def _sel(suffix):
        rows = []
        for gi, gk in enumerate(norm_keys):
            if not gk.endswith(suffix):
                continue
            tgt = 'ln_gin' if suffix == 'gamma' else 'ln_bin'
            body = " ".join(f"{tgt}[{j}*{ACTW} +: {ACTW}] = {_san(gk)}_g_lut({j});"
                            for j in range(H))
            rows.append(f"      {gi}: begin {body} end")
        return "\n".join(rows)

    ln_gsel = _sel("gamma")
    ln_bsel = _sel("beta")

    state_defs, master_case = _build_master_gpt2(engine_list, cfg)

    r = _TOP_TEMPLATE_GPT2
    lo = -(1 << (qmodel.luts.gelu_input_bits - 1))
    hi = (1 << (qmodel.luts.gelu_input_bits - 1)) - 1
    repl = {
        "MODNAME": f"{cfg['name']}_accel",
        "H": H, "HEADS": HEADS, "HD": HD, "LYR": LYR, "SEQ": SEQ, "VOCAB": VOCAB,
        "NINNER": int(cfg.get("n_inner", H * 4)),
        "F": F, "RS": REQUANT_S, "RR": RR_DIV, "ACT": ACTW,
        "SILU_LO": lo, "SILU_HI": hi,
        "GELU_LO": lo, "GELU_HI": hi, "GELU_SPAN": hi - lo,
    }
    for k, v in repl.items():
        r = r.replace(f"@@{k}@@", str(v))
    r = r.replace("@@NORM_ROMS@@", norm_fns)
    r = r.replace("@@WTE_FN@@", wte_fn)
    r = r.replace("@@WPE_FN@@", wpe_fn)
    r = r.replace("@@GELU_FN@@", gelu_fn)
    r = r.replace("@@LN_GSEL@@", ln_gsel)
    r = r.replace("@@LN_BSEL@@", ln_bsel)
    r = r.replace("@@GEMV_DECLS@@", "\n".join(gemv_decls))
    r = r.replace("@@GEMV_INSTS@@", "\n".join(gemv_insts))
    r = r.replace("@@GEMV_DRIVES@@", "\n".join(gemv_drives))
    r = r.replace("@@GEMV_LATCH@@", "\n".join(gemv_latch))
    r = r.replace("@@STATE_DEFS@@", state_defs)
    r = r.replace("@@MASTER_CASE@@", master_case)
    r = r.replace(
        "S<=S_IDLE; done<=0; tokk<=0; ln_en<=0; ln_xsel<=0; att_en<=0; att_layer<=0;",
        "S<=S_IDLE; done<=0; tokk<=0; ln_en<=0; ln_xsel<=0; att_en<=0; att_layer<=0;"
        + "".join(f" gen_{i}<=0;" for i in range(len(engine_list))))
    return r


def _emit_tb(qmodel, cfg: dict, modname: str) -> str:
    SEQ = cfg["max_seq_len"]

    SEQ = cfg["max_seq_len"]
    VOCAB = cfg["vocab_size"]
    return rf'''
`timescale 1ns/1ps
module tb;
  localparam SEQ={SEQ}, VOCAB={VOCAB};
  logic clk=0, rst_n=0, start=0;
  logic [15:0] token_ram [0:SEQ-1];
  logic done;
  logic signed [27:0] logit_bank [0:SEQ-1][0:VOCAB-1];
  // flat 端口（DUT 端口已展平，便于综合）
  logic [SEQ*16-1:0] token_flat;
  logic signed [SEQ*VOCAB*28-1:0] logit_flat;
  genvar gi_f, gt_f, gv_f;
  generate
    for (gi_f=0; gi_f<SEQ; gi_f=gi_f+1) begin : gtok
      assign token_flat[gi_f*16 +: 16] = token_ram[gi_f];
    end
    for (gt_f=0; gt_f<SEQ; gt_f=gt_f+1) begin : glog_t
      for (gv_f=0; gv_f<VOCAB; gv_f=gv_f+1) begin : glog_v
        assign logit_bank[gt_f][gv_f] = logit_flat[(gt_f*VOCAB+gv_f)*28 +: 28];
      end
    end
  endgenerate
  always #5 clk = ~clk;

  {modname} dut (
    .clk(clk), .rst_n(rst_n), .start(start), .token_flat(token_flat),
    .done(done), .logit_flat(logit_flat)
  );

  int fid;
  initial begin
    $readmemh("tokens.mem", token_ram);
    rst_n <= 0; #20; rst_n <= 1;
    #20; start <= 1; #20; start <= 0;
    wait (done);
    #20;
    fid = $fopen("sim_logits.txt", "w");
    for (int t=0;t<SEQ;t=t+1)
      for (int v=0;v<VOCAB;v=v+1)
        $fwrite(fid, "%0d\n", logit_bank[t][v]);
    $fclose(fid);
    $display("SIM_DONE");
    $finish;
  end
endmodule
'''


def _fill_module(s: str, qmodel, **kw) -> str:
    for k, v in kw.items():
        s = s.replace("@@" + k + "@@", str(v))
    luts = qmodel.luts
    rsqrt_idx = (luts.rsqrt.shape[0] - 1).bit_length() - 1
    rsqrt_fn = _case_rom_fn("rsqrt_lut", _hex_patterns(luts.rsqrt))
    exp_fn = _case_rom_fn("expneg_lut", _hex_patterns(luts.exp_neg))
    recip_fn = _case_rom_fn("recip_lut", _hex_patterns(luts.recip2))
    return (s.replace("@@RSQRT_MAX@@", str(luts.rsqrt.shape[0] - 1))
             .replace("@@RSQRT_IDX@@", str(rsqrt_idx))
             .replace("@@EXPMAX@@", str(luts.exp_neg.shape[0] - 1))
             .replace("@@RECMAX@@", str(luts.recip2.shape[0] - 1))
             .replace("@@RRFN@@", _RR_FN.strip())
             .replace("@@RSQRT_FN@@", rsqrt_fn)
             .replace("@@EXP_FN@@", exp_fn)
             .replace("@@RECIP_FN@@", recip_fn))


def _write_idem(path: str, content: str) -> bool:
    """幂等写：内容不变时保持 mtime（避免 SDC/综合把 RTL 视为变更重跑）。"""
    if os.path.exists(path) and open(path, "r").read() == content:
        return False
    with open(path, "w") as f:
        f.write(content)
    return True


def _read_hex_tokens(path: str) -> list:
    """读取 .mem 文件每行的 hex 词（wrom/nrom/brom 只取自量化产物，
    与 $readmemh 时代逐位等价）。"""
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def generate(qmodel, cfg: dict, out_dir: str, tokens: np.ndarray,
             backend_dir: str = "rtl", single_file: bool = False) -> str:
    """生成 RTL 文件到 out_dir/rtl。返回顶层模块名。"""
    modname = f"{cfg['name']}_accel"
    rdir = os.path.join(out_dir, backend_dir)
    os.makedirs(rdir, exist_ok=True)
    _sync_actw()
    is_gpt2 = cfg.get("architecture") == "gpt2"
    top = _emit_top_gpt2(qmodel, cfg) if is_gpt2 else _emit_top(qmodel, cfg)

    # gemv 权重/尺度/偏置：从量化产物 .mem 取数，内联为 case 常量函数。
    # （不再用 $readmemh；同一份 hex 词保证与解析器/黄金参考逐位一致。）
    wrom_dir = os.path.join(out_dir, "quantizer", "weights_rom")
    if not os.path.isdir(wrom_dir):
        raise FileNotFoundError(
            f"未找到量化产物目录 {wrom_dir}，无法生成可综合权重 ROM")

    # 每个 gemv 引擎一个独立模块（文件名硬编码，兼容 Yosys）
    engine_list = sorted(qmodel.engines.items())
    gemv_files = {}
    simd = qmodel.simd
    ww = qmodel.bit_width
    for idx, (key, qw) in enumerate(engine_list):
        wrom_fn = _case_rom_fn(
            "wrom_lut", _read_hex_tokens(os.path.join(wrom_dir, qw.rom_file)),
            out_bits=ww * simd)
        nrom_fn = _case_rom_fn(
            "nrom_lut",
            _read_hex_tokens(os.path.join(wrom_dir, qw.scale_rom_file)),
            out_bits=24)
        has_bias = getattr(qw, "bias_q", None) is not None
        if is_gpt2 and has_bias:
            brom_fn = _case_rom_fn(
                "brom_lut",
                _read_hex_tokens(os.path.join(wrom_dir, qw.bias_rom_file)))
            gemv_tpl = GEMV_TEMPLATE_BIAS
        else:
            brom_fn = ""
            gemv_tpl = GEMV_TEMPLATE
        gemv_files[f"gemv_{idx}.sv"] = _fill_module(
            gemv_tpl, qmodel,
            GMOD=f"gemv_{idx}",
            WROM_FN=wrom_fn, NROM_FN=nrom_fn, BROM_FN=brom_fn)

    files = {
        f"{modname}.sv": top,
        **gemv_files,
        "rmsnorm.sv": _fill_module(RMSNORM_TEMPLATE, qmodel),
        "attn.sv": _fill_module(ATTN_TEMPLATE, qmodel),
        "sim_tb.sv": _emit_tb(qmodel, cfg, modname),
    }
    if is_gpt2:
        files["layernorm.sv"] = _fill_module(LAYERNORM_TEMPLATE, qmodel)
    for fn, content in files.items():
        _write_idem(os.path.join(rdir, fn), content)
    path = os.path.join(rdir, "tokens.mem")
    tcontent = "".join(f"{int(t):x}\n" for t in np.asarray(tokens).reshape(-1))
    if not (os.path.exists(path) and open(path, "r").read() == tcontent):
        with open(path, "w") as f:
            f.write(tcontent)

    if single_file:
        sf = f"{modname}_single.sv"
        # 依赖在前：gemv 引擎 -> layernorm/rmsnorm -> attn -> 顶层
        order = list(gemv_files) + ["layernorm.sv" if is_gpt2 else "rmsnorm.sv",
                                    "attn.sv", f"{modname}.sv"]
        _write_idem(os.path.join(rdir, sf),
                    "\n".join(files[n].rstrip() for n in order) + "\n")
    return modname
