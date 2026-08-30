# llm2asic/rtl_backend/verilog.py
"""RTL 生成器：QuantizedModel + ROM(.mem) -> SystemVerilog 顶层与测试台。

算子与 reference.IntModel 逐位一致。模块：gemv / rmsnorm / attn；顶层负责调度。
用 sentinel 令牌 (``@@K@@``) + replace 拼接，避免与 Verilog 花括号冲突。
"""

from __future__ import annotations

import math
import os

import numpy as np

from .numeric import F, REQUANT_S, ACT_BITS

EF = 10
RR_DIV = 2 * F


def _san(k: str) -> str:
    return k.replace(".", "_").replace("-", "_")


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
// ROM 文件名以字面量硬编码（Yosys 不支持 string 类型参数），故每个引擎
// 输出独立文件 gemv_<idx>.sv 且模块名唯一。
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
  localparam WPR = (C_IN + SIMD - 1) / SIMD;
  logic [WW*SIMD-1:0] wrom [0:WORDS-1];
  logic [NUM_BITS-1:0] nrom [0:C_OUT-1];
  initial begin $readmemh("@@WF@@", wrom); $readmemh("@@SF@@", nrom); end

  function automatic logic signed [63:0] chunk_mac(input int oo, input int cc);
    integer j; logic signed [31:0] w; logic signed [63:0] p;
    p = 0;
    for (j=0;j<SIMD;j=j+1) begin
      w = $signed({ {32-WW{ wrom[oo*WPR+cc][j*WW+WW-1] }}, wrom[oo*WPR+cc][j*WW +: WW] });
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
          yout[o*ACT +: ACT] <= rr(acc*$signed({ {64-NUM_BITS{1'b0}}, nrom[o]}), RS);
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
  logic signed [23:0] rsqrt_mem [0:RMX];
  initial $readmemh("rsqrt.mem", rsqrt_mem);

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
            mean = (rsum + (H/2)) / H + 2;
            idx = (mean<1) ? 1 : (mean>RMX) ? RMX : mean;
            c <= $signed(rsqrt_mem[idx[20:0]]);
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

ATTN_TEMPLATE = r'''
// 单 token 多头注意力：定点 softmax（exp LUT + recip LUT）。qr/kvk/kvv 打包。
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
  logic signed [23:0] expneg_mem [0:@@EXPMAX@@];
  logic signed [23:0] recip_mem  [0:@@RECMAX@@];
  initial $readmemh("exp_neg.mem", expneg_mem);
  initial $readmemh("recip.mem", recip_mem);

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
              e[p]   <= expneg_mem[d[15:0]];
              ssum   <= ssum + $signed(expneg_mem[d[15:0]]);
            end
          end
          if (p==seq-1) st<=ANORM; else p<=p+1;
        end
        ANORM: begin
          begin : rn
            logic signed [63:0] sid;
            sid = (ssum<1) ? 1 : ((ssum>@@RECMAX@@) ? @@RECMAX@@ : ssum);
            rcp <= $signed(recip_mem[sid[15:0]]);
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
         f"for(int i=0;i<H;i=i+1) h[i]<=embed_rom[token_ram[tokk]*H+i]; "
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
               f"<=$signed(rms_yout[i*24 +: 24]);") if target else ""
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
             f"for(int i=0;i<H;i=i+1) att[i]<=$signed(att_yout[i*24 +: 24]); "
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

  // ---------- 嵌入 ROM ----------
  logic signed [23:0] embed_rom [0:VOCAB*H-1];
  initial $readmemh("wte_q.mem", embed_rom);

  // ---------- gamma ROM ----------
@@GAMMA_ROMS@@

  // ---------- silu LUT ----------
  localparam SILU_LO=@@SILU_LO@@;
  logic signed [23:0] silu_mem [0:@@SILU_SPAN@@];
  initial $readmemh("silu.mem", silu_mem);

  // ---------- rope cos/sin 常量 ----------
  logic signed [23:0] rope_c [0:SEQ*HD/2 -1];
  logic signed [23:0] rope_s [0:SEQ*HD/2 -1];
@@ROPE_CINIT@@
@@ROPE_SINIT@@

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
        cc = rope_c[tokk*(HD/2)+j];
        ss = rope_s[tokk*(HD/2)+j];
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
        u  = silu_mem[sidx];
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
            f"  logic signed [{c_in*24-1}:0] gx_{idx};\n"
            f"  logic signed [{c_out*24-1}:0] gy_{idx};")
        words = c_out * ((c_in + simd - 1) // simd)
        gemv_insts.append(
            f"  gemv_{idx} #(.C_IN({c_in}),.C_OUT({c_out}),.SIMD({simd}),.WW({ww}),.ACT(24),"
            f".RS({REQUANT_S}),.WORDS({words}),.NUM_BITS(24)) "
            f"u_gv{idx}(.clk(clk),.rst_n(rst_n),.en(gen_{idx}),"
            f".x(gx_{idx}),.yout(gy_{idx}),.done(gd_{idx}));")
        src = inp_of[key]
        gemv_drives.append(
            f"  for (genvar G{idx}=0; G{idx}<{H}; G{idx}=G{idx}+1) "
            f"assign gx_{idx}[G{idx}*24 +: 24] = {src}[G{idx}];")
        tgt = target_of[key]; c_out = qw.c_out
        target_latch.setdefault(tgt, []).append((idx, c_out))

    _latch_blocks = []
    for tgt, items in target_latch.items():
        parts = []
        for k, (idx, c_out) in enumerate(items):
            kw = "if" if k == 0 else "else if"
            parts.append(f"{kw} (gd_{idx}) begin for (int ii=0; ii<{c_out}; ii=ii+1) {tgt}[ii] <= gy_{idx}[ii*24 +: 24]; end")
        parts.append("else ;")
        _latch_blocks.append("  always_ff @(posedge clk) begin " + " ".join(parts) + " end")
    gemv_latch = _latch_blocks

    # gamma roms
    gamma_roms = "\n".join(
        f"  logic signed [23:0] {_san(k)}_g [0:{int(v.size)-1}];\n"
        f"  initial $readmemh(\"{_san(k)}.mem\", {_san(k)}_g);"
        for k, v in qmodel.gammas.items())

    # rope cos/sin init
    cq, sq = _precompute_rope_cs(cfg)
    nc = SEQ * (HD // 2)
    def _f(v):
        return f"-24'sd{abs(v)}" if v < 0 else f"24'sd{v}"
    rope_cinit = "\n".join(
        f"  rope_c[{x}] = {_f(cq[x//(HD//2)][x%(HD//2)])};" for x in range(nc))
    rope_sinit = "\n".join(
        f"  rope_s[{x}] = {_f(sq[x//(HD//2)][x%(HD//2)])};" for x in range(nc))

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
            f"rms_gin[{i}*24 +: 24] = {_san(gk)}_g[{i}];" for i in range(H))
        rms_gsel.append(f"      {gi}: begin {body} end")
    rms_gsel = "\n".join(rms_gsel)

    state_defs, master_case = _build_master(engine_list, cfg)

    r = _TOP_TEMPLATE
    repl = {
        "MODNAME": f"{cfg['name']}_accel",
        "H": H, "HEADS": HEADS, "HD": HD, "LYR": LYR, "SEQ": SEQ, "VOCAB": VOCAB,
        "F": F, "RS": REQUANT_S, "RR": RR_DIV, "ACT": ACT_BITS,
        "SILU_LO": -(1 << (cfg.get('silu_input_bits', qmodel.luts.silu_input_bits) - 1)),
        "SILU_HI": (1 << (cfg.get('silu_input_bits', qmodel.luts.silu_input_bits) - 1)) - 1,
    }
    for k, v in repl.items():
        r = r.replace(f"@@{k}@@", str(v))
    # computed spans
    lo = -(1 << (qmodel.luts.silu_input_bits - 1))
    hi = (1 << (qmodel.luts.silu_input_bits - 1)) - 1
    r = r.replace("@@SILU_SPAN@@", str(hi - lo))
    r = r.replace("@@GAMMA_ROMS@@", gamma_roms)
    r = r.replace("@@ROPE_CINIT@@", "  initial begin\n" + rope_cinit + "\n  end")
    r = r.replace("@@ROPE_SINIT@@", "  initial begin\n" + rope_sinit + "\n  end")
    r = r.replace("@@GEMV_DECLS@@", "\n".join(gemv_decls))
    r = r.replace("@@GEMV_INSTS@@", "\n".join(gemv_insts))
    r = r.replace("@@GEMV_DRIVES@@", "\n".join(gemv_drives))
    r = r.replace("@@GEMV_LATCH@@", "\n".join(gemv_latch))
    r = r.replace("@@RMS_GSEL@@", rms_gsel)
    r = r.replace("@@STATE_DEFS@@", state_defs)
    r = r.replace("@@MASTER_CASE@@", master_case)
    return r



def _emit_tb(qmodel, cfg: dict, modname: str) -> str:
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
    return (s.replace("@@RSQRT_MAX@@", str(qmodel.luts.rsqrt.shape[0] - 1))
             .replace("@@EXPMAX@@", str(qmodel.luts.exp_neg.shape[0] - 1))
             .replace("@@RECMAX@@", str(qmodel.luts.recip2.shape[0] - 1))
             .replace("@@RRFN@@", _RR_FN.strip()))


def generate(qmodel, cfg: dict, out_dir: str, tokens: np.ndarray,
             backend_dir: str = "rtl") -> str:
    """生成 RTL 文件到 out_dir/rtl。返回顶层模块名。返回主 .sv 路径。"""
    modname = f"{cfg['name']}_accel"
    rdir = os.path.join(out_dir, backend_dir)
    os.makedirs(rdir, exist_ok=True)
    top = _emit_top(qmodel, cfg)

    # 每个 gemv 引擎一个独立模块（文件名硬编码，兼容 Yosys）
    engine_list = sorted(qmodel.engines.items())
    gemv_files = {}
    for idx, (key, qw) in enumerate(engine_list):
        gemv_files[f"gemv_{idx}.sv"] = _fill_module(
            GEMV_TEMPLATE, qmodel,
            GMOD=f"gemv_{idx}", WF=qw.rom_file, SF=qw.scale_rom_file)

    files = {
        f"{modname}.sv": top,
        **gemv_files,
        "rmsnorm.sv": _fill_module(RMSNORM_TEMPLATE, qmodel),
        "attn.sv": _fill_module(ATTN_TEMPLATE, qmodel),
        "sim_tb.sv": _emit_tb(qmodel, cfg, modname),
    }
    for fn, content in files.items():
        with open(os.path.join(rdir, fn), "w") as f:
            f.write(content)
    with open(os.path.join(rdir, "tokens.mem"), "w") as f:
        for t in np.asarray(tokens).reshape(-1):
            f.write(f"{int(t):x}\n")
    return modname
