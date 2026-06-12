<%inherit file='base'/>

## bstream-cu-coop — MI300X / CDNA3 A-stationary, Infinity-Cache-friendly.
##
## A is a constant FR operator, so the whole work-group cooperatively stages
## A's non-zero values into LDS once and reuses them for every column it
## streams. B is streamed through the 256 MB Infinity Cache (shared across the
## 8 XCDs) and is highly reused across work-groups, so keeping it cache-
## resident is the win. C is written with __builtin_nontemporal_store so the
## write-once output does not evict reusable B lines from the cache hierarchy.
<%
bix_list = list(bix)
# Flatten A's non-zeros row-major; aoff[j] is the start of row j in Aval[].
aoff, flat = [], []
for j in range(m):
    aoff.append(len(flat))
    for kx in range(k):
        if A[j, kx] != 0:
            flat.append((j, kx, A[j, kx]))
nnz = len(flat)
# Per (row,col) -> index into the LDS Ash[] array.
aidx = {(j, kx): t for t, (j, kx, v) in enumerate(flat)}
%>
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
    const ${'long long' if k*ldb >= 2**31 else 'int'} ldb = ${ldb};
    const ${'long long' if m*ldc >= 2**31 else 'int'} ldc = ${ldc};
% endif
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    // A's non-zeros baked as a compile-time constant, staged into LDS once.
    static const ${dtype} Aval[${nnz}] = { ${', '.join(repr(float(v)) for (_, _, v) in flat)} };
    __shared__ ${dtype} Ash[${nnz}];   // A non-zeros, shared per work-group
    ${dtype} bv, csub[${m}];

    // Cooperatively load A's non-zeros into LDS (constant operator, reused for
    // every streamed column -> stays resident, off the HBM critical path).
    for (int t = threadIdx.x; t < ${nnz}; t += ${blockx})
        Ash[t] = Aval[t];
    __syncthreads();

    if (i < n)
    {
% if beta != 0:
  % for j in range(m):
    % if afix[j] != -1:
        csub[${j}] = ${'' if beta == 1 else f'{beta}*'}c[i + ${j}*ldc];
    % endif
  % endfor
% endif
% for kx in bix_list:
        bv = b[i + ${kx}*ldb];   // streamed via Infinity Cache, reused across work-groups
  % for j in range(m):
    % if A[j, kx] != 0 and kx == afix[j] and beta == 0:
        csub[${j}] = Ash[${aidx[(j, kx)]}]*bv;
    % elif A[j, kx] != 0:
        csub[${j}] += Ash[${aidx[(j, kx)]}]*bv;
    % endif
    % if kx == alix[j]:
        __builtin_nontemporal_store(csub[${j}], &c[i + ${j}*ldc]);
    % endif
  % endfor
% endfor
% if beta == 0:
  % for j in range(m):
    % if afix[j] == -1:
        __builtin_nontemporal_store(make_zero(), &c[i + ${j}*ldc]);
    % endif
  % endfor
% endif
    }
}
