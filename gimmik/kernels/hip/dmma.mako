<%inherit file='base'/>

## dmma: MFMA-based dense matmul for AMD CDNA2/CDNA3 (MI200/MI300 series).
## Uses __builtin_amdgcn_mfma_f64_16x16x4f64 (wavefront-64 only).
##
## Tile layout (mirrors PTX dmma-astream-v1 with mma.sync.aligned.m8n8k4):
##   MFMA m16n16k4: C[16,16], A[16,4], B[4,16]
##   Each wavefront (64 lanes) owns the full C tile.
##   Lane geometry: lane = r_div4*4 + r_mod4
##     r_div4 = lane >> 2  (0..15) -> selects C row pair
##     r_mod4 = lane &  3  (0..3)  -> selects k sub-row of A/B
##
## A is a constant matrix embedded as a .global array (gA[m_tiles][k_tiles][16][4]),
## stored in row-major order matching the MFMA lane layout.
## Each lane reads: gA[mt][kt][r_div4][r_mod4] = A[mt*16 + r_div4, kt*4 + r_mod4]

<%
import numpy as np

# Tile dimensions for mfma_f64_16x16x4
MROW = 16   # C rows per tile
KCOL = 4    # k cols per A/B tile (k dimension of MFMA)
NCOL = 16   # C/B cols per tile

# Number of m-tiles and k-tiles needed to cover the full A matrix
m_tiles  = -(-m // MROW)        # ceil(m / 16)
k_tiles  = -(-k // KCOL)        # ceil(k / 4)
m_pad    = m_tiles * MROW
k_pad    = k_tiles * KCOL

# Number of N tiles per warp (each warp handles n_per_warp consecutive N cols)
# blockx must be a multiple of 64 (wavefront size)
n_per_warp = 1   # each wavefront handles one 16-wide N tile

# Pad A to (m_pad, k_pad) with zeros
Apad = np.zeros((m_pad, k_pad), dtype=A.dtype)
Apad[:m, :k] = A

# Build the gA array in the correct lane order:
# gA[mt][kt][row_in_tile][k_in_tile] — row_in_tile=0..15, k_in_tile=0..3
gA_flat = []
for mt in range(m_tiles):
    for kt in range(k_tiles):
        tile = Apad[mt*MROW:(mt+1)*MROW, kt*KCOL:(kt+1)*KCOL]
        for row in range(MROW):
            for ki in range(KCOL):
                gA_flat.append(tile[row, ki])

# Encode as u64 hex literals (preserves f64 bit pattern exactly)
import struct
def f64_hex(v):
    return '0x{:016X}'.format(struct.unpack('Q', struct.pack('d', float(v)))[0])

a_u64 = [f64_hex(v) for v in gA_flat]

# C accumulator layout: each lane holds 4 values (two pairs of rows)
# The MFMA instruction produces:
#   {c0, c1, c2, c3} where c0,c1 are row r_div4 cols (r_mod4*2, r_mod4*2+1)
#                     and c2,c3 are row r_div4+8 cols (r_mod4*2, r_mod4*2+1)
# This means each lane covers:
#   C[r_div4,     r_mod4*2], C[r_div4,     r_mod4*2+1]
#   C[r_div4+8,   r_mod4*2], C[r_div4+8,   r_mod4*2+1]

# B layout: lane holds B[r_mod4, col] where col is a function of warp/n base
# For k_tiles > 1 the B fragment stride is ldb*KCOL*itemsize

dwidth_i  = 8 if dtype == 'double' else 4
n_per_cta = blockx   # one wavefront per block for simplicity
%>

.global .align 16 .b64 ${kname}_Ag[${len(a_u64)}] = {
    ${', '.join(a_u64)}
};

__global__ __launch_bounds__(${blockx}, 1) void
${kname}(const ${dtype}* __restrict__ b, ${dtype}* __restrict__ c)
{
    const int n   = ${n};
    const int ldb = ${ldb};
    const int ldc = ${ldc};

    const int lane    = __lane_id();            /* 0..63 */
    const int r_div4  = lane >> 2;              /* 0..15: row index within tile */
    const int r_mod4  = lane &  3;              /* 0..3 : k sub-index */
    const int warp_n  = blockIdx.x;             /* one warp per CTA covers one N tile */
    const int n_base  = warp_n * ${NCOL};       /* first N column for this warp */

    if (n_base >= n)
        return;

    /* Pointer to this lane's A fragment base in constant global memory */
    const ${dtype}* ag_lane = (const ${dtype}*)${kname}_Ag + lane;

    /* B base pointer: b[r_mod4, n_base + r_div4] in col-major layout */
    const ${dtype}* b_lane = b + (long long)r_mod4 * ldb + n_base + r_div4;

    /* C base pointer: c[r_div4, n_base + r_mod4*2] */
    ${dtype}* c_lane = c + (long long)r_div4 * ldc + n_base + r_mod4 * 2;

% for mt in range(m_tiles):
    {
        /* Accumulator for C tile mt (rows ${mt*MROW}..${min((mt+1)*MROW-1, m-1)}) */
        typedef double double4 __attribute__((ext_vector_type(4)));
% if beta == 0:
        double4 acc = {0.0, 0.0, 0.0, 0.0};
% else:
        /* Pre-load C and scale by beta */
        double4 acc;
        acc[0] = ${beta} * c_lane[${mt * MROW * '+ ldc*8'[:-6] if mt else ''}              ];
        acc[1] = ${beta} * c_lane[${mt * MROW * '+ ldc*8'[:-6] if mt else ''}          + 1 ];
        acc[2] = ${beta} * c_lane[${mt * MROW * '+ ldc*8'[:-6] if mt else ''} + ldc * 8    ];
        acc[3] = ${beta} * c_lane[${mt * MROW * '+ ldc*8'[:-6] if mt else ''} + ldc * 8 + 1];
% endif

% for kt in range(k_tiles):
        {
            /* Load A fragment: gA[mt][kt][r_div4][r_mod4] */
            ${dtype} a_f = ag_lane[${(mt * k_tiles + kt) * MROW * KCOL}];
            /* Load B fragment: b[kt*KCOL + r_mod4, n_base + r_div4] */
            ${dtype} b_f = b_lane[${kt * KCOL} * ldb];
            acc = __builtin_amdgcn_mfma_f64_16x16x4f64(a_f, b_f, acc, 0, 0, 0);
        }
% endfor

        /* Store the 4 accumulator values back to C */
<%
    row0 = mt * MROW
    row8 = mt * MROW + 8
    in_bounds_r0 = row0 < m
    in_bounds_r8 = row8 < m
%>
% if in_bounds_r0:
        c_lane[ldc * ${mt * MROW}    ] = acc[0];
        c_lane[ldc * ${mt * MROW} + 1] = acc[1];
% endif
% if in_bounds_r8:
        c_lane[ldc * ${mt * MROW + 8}    ] = acc[2];
        c_lane[ldc * ${mt * MROW + 8} + 1] = acc[3];
% endif
    }
% endfor
}
