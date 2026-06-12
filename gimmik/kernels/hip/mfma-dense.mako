<%inherit file='base'/>

## mfma-dense — MI300X / CDNA3 Matrix-Core dense kernel (f64).
##
## Abandons sparsity: A (the constant operator) is densified, padded with
## zeros, and baked into a __device__ constant array in the lane-order layout
## expected by mfma_f64_16x16x4f64 (16x16 <- 16x4 . 4x16, 64 lanes each holding
## one f64 of the 16x4 A fragment). Each 64-lane wavefront owns one 16-wide N
## tile and streams B fragments through the matrix core. For dense / near-dense
## operators (e.g. tetrahedral) the ~256 TFLOPS f64 MFMA throughput beats the
## vector-ALU path despite multiplying through the zeros. NOTE: this is a
## compute-bound kernel, not a bandwidth optimisation.
##
## Requires gfx90a+ (CDNA2/CDNA3) and f64. Gate m,k <= 256.
<%
import numpy as np
mt = -(-m // 16)
kt = -(-k // 4)
dense = np.zeros((mt*16, kt*4))
dense[:m, :k] = np.asarray(A, dtype=float)
# Lane-order A: lane l holds dense[16*mtile + (l & 15)][4*ktile + (l >> 4)].
aflat = []
for mtile in range(mt):
    for ktile in range(kt):
        for l in range(64):
            aflat.append(dense[16*mtile + (l & 15)][4*ktile + (l >> 4)])
%>
typedef ${dtype} f64x4 __attribute__((ext_vector_type(4)));

__device__ static const ${dtype} Ag[${mt*kt*64}] = {
    ${', '.join(repr(float(v)) for v in aflat)}
};

__global__ void __launch_bounds__(64)
% if n is None:
${kname}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
% else:
${kname}(const ${dtype}* __restrict__ b, ${dtype}* __restrict__ c)
{
    const int n = ${n};
    const ${'long long' if k*ldb >= 2**31 else 'int'} ldb = ${ldb};
    const ${'long long' if m*ldc >= 2**31 else 'int'} ldc = ${ldc};
% endif
    const int lane   = threadIdx.x & 63;
    const int wave   = (blockIdx.x*blockDim.x + threadIdx.x) >> 6;
    const int n_base = wave*16;
    if (n_base >= n) return;

    const int b_row = lane >> 4;          // which of the 4 k-rows this lane reads
    const int b_col = n_base + (lane & 15);

% for mtile in range(mt):
    f64x4 acc${mtile} = {0, 0, 0, 0};
% endfor

% if beta != 0:
    // beta: preload C into the accumulators (MFMA then adds onto them)
  % for mtile in range(mt):
    % for r in range(4):
        if (${16*mtile} + (lane>>4) + ${4*r} < ${m})
            acc${mtile}[${r}] = ${'' if beta == 1 else f'{beta}*'}c[(${16*mtile} + (lane>>4) + ${4*r})*ldc + n_base + (lane & 15)];
    % endfor
  % endfor
% endif

    // Stream B fragments, pull A fragments from the constant array, MFMA.
% for ktile in range(kt):
<%  ktail = (k % 4 != 0) and (ktile == kt - 1) %>
    {
        ${dtype} bf = ${f'(b_row + {4*ktile} < {k}) ? ' if ktail else ''}b[(${4*ktile} + b_row)*ldb + b_col]${' : 0.0' if ktail else ''};
  % for mtile in range(mt):
        acc${mtile} = __builtin_amdgcn_mfma_f64_16x16x4f64(Ag[${(mtile*kt + ktile)}*64 + lane], bf, acc${mtile}, 0, 0, 0);
  % endfor
    }
% endfor

    // Write C: each lane owns 4 rows x 1 column of the 16x16 output tile.
% for mtile in range(mt):
  % for r in range(4):
        if (${16*mtile} + (lane>>4) + ${4*r} < ${m})
            c[(${16*mtile} + (lane>>4) + ${4*r})*ldc + n_base + (lane & 15)] = acc${mtile}[${r}];
  % endfor
% endfor
}
