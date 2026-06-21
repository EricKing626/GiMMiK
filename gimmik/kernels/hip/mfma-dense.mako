<%doc>
  mfma-dense  (CDNA3 / gfx94x, MI300X)
  ===================================
  COMPUTE-bound dense kernel: A is densified (zeros included) and run through
  the f64 Matrix Core via v_mfma_f64_16x16x4f64. Intended ONLY for dense,
  compute-limited operators (high-order tet) where the vector-ALU GiMMiK
  kernels are FLOP-bound and lose to rocBLAS. Gated by _mfma_dense_ok in hip.py.

  v_mfma_f64_16x16x4f64 semantics (one wavefront = 64 lanes):
    D[16x16] += A[16x4] * B[4x16]
  Operand lane layout (per AMD matrix-instruction calculator):
    A: lane l holds A[l%16][l//16]      (16 rows x 4 k)
    B: lane l holds B[l//16][l%16]      (4 k x 16 cols)
    D: 4 f64 per lane; acc index a holds D[(l//16)*4 + a][l%16]

  A is constant, so it is baked as a compile-time lane-ordered .global array
  (Ag) and pulled once. B is streamed from global; C is written non-temporally.

  NOTE: this is a baseline MFMA kernel (single wavefront per 16-col N tile,
  no warp-specialised producer/consumer overlap). It exists to test whether the
  Matrix-Core path can approach rocBLAS on dense tet; if it is close but still
  short, the next step is a warp-specialised version (LDS-staged B double-
  buffer + s_barrier), mirroring the Hopper TMA/MMA design but on CDNA3.

  Verify on-device for accuracy (MFMA operand layout) and that A's densified
  layout matches Ag before trusting results.
</%doc>
<%inherit file='base'/>

<%
import numpy as np
Adense = np.asarray(A, dtype=float)
m_, k_ = Adense.shape
MT = 16   # MFMA M tile
KT = 4    # MFMA K tile
NT = 16   # MFMA N tile (cols of C per wavefront)
m_tiles = -(-m_ // MT)
k_tiles = -(-k_ // KT)
# Pad A up to (m_tiles*16) x (k_tiles*4)
Apad = np.zeros((m_tiles*MT, k_tiles*KT), dtype=float)
Apad[:m_, :k_] = Adense
# Lane-ordered flatten: for each (m_tile, k_tile) block, lane l -> A[l%16][l//16]
a_vals = []
for mt in range(m_tiles):
    for kt in range(k_tiles):
        for l in range(64):
            r = mt*MT + (l % 16)
            c = kt*KT + (l // 16)
            a_vals.append(repr(float(Apad[r, c])))
%>

__device__ static const ${dtype} ${kname}_Ag[${m_tiles*k_tiles*64}] = {
    ${', '.join(a_vals)}
};

__global__ __launch_bounds__(${blockx}) void
% if n is None:
${kname}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
% else:
${kname}(const ${dtype}* __restrict__ b, ${dtype}* __restrict__ c)
{
    const int n = ${n};
    const int ldb = ${ldb};
    const int ldc = ${ldc};
% endif
    typedef ${dtype} f64x4 __attribute__((ext_vector_type(4)));

    const int lane   = threadIdx.x & 63;
    const int wave   = (blockDim.x*blockIdx.x + threadIdx.x) >> 6;
    const int n_base = wave*${NT};                 // first C/B column for this wavefront
    if (n_base >= n) return;

    const int brow = lane >> 4;                    // 0..3  (k within tile)
    const int bcol = lane & 15;                    // 0..15 (n within tile)
    const int n_col = n_base + bcol;
    const bool ncol_ok = (n_col < n);

    // Load A constants for this wavefront's lanes (one f64 per (mt,kt) block).
% for mt in range(m_tiles):
%  for kt in range(k_tiles):
    ${dtype} a_${mt}_${kt} = ${kname}_Ag[${(mt*k_tiles + kt)*64} + lane];
%  endfor
% endfor

    // Accumulators: m_tiles blocks, each 4 f64 per lane.
% for mt in range(m_tiles):
    f64x4 acc${mt} = {0, 0, 0, 0};
% endfor

    // Stream B from global and issue MFMA per k-tile.
% for kt in range(k_tiles):
    {
        ${dtype} bval = 0;
        const int b_k = ${kt*KT} + brow;
        if (b_k < ${k_} && ncol_ok)
            bval = b[b_k*ldb + n_col];
%  for mt in range(m_tiles):
        acc${mt} = __builtin_amdgcn_mfma_f64_16x16x4f64(a_${mt}_${kt}, bval, acc${mt}, 0, 0, 0);
%  endfor
    }
% endfor

    // Write C non-temporally. acc index a of lane l -> row (l//16)*4 + a.
% for mt in range(m_tiles):
    {
      % for a in range(4):
        {
            const int crow = ${mt*MT} + brow*4 + ${a};
            if (crow < ${m_} && ncol_ok)
            {
% if beta == 0:
                __builtin_nontemporal_store(acc${mt}[${a}], &c[crow*ldc + n_col]);
% elif beta == 1:
                __builtin_nontemporal_store(c[crow*ldc + n_col] + acc${mt}[${a}], &c[crow*ldc + n_col]);
% else:
                __builtin_nontemporal_store(acc${mt}[${a}] + ${beta}*c[crow*ldc + n_col], &c[crow*ldc + n_col]);
% endif
            }
        }
      % endfor
    }
% endfor
}
