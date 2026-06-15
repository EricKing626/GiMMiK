<%doc>
  bstream-cu-coop — CU-cooperative bstream (MI300X / CDNA3, gfx940+).

  Two ideas, both aimed at cutting redundant HBM reads on DENSE operators:
    (1) A is a constant operator shared by the whole work-group, so the
        work-group cooperatively stages A's non-zeros into LDS once (Aval ->
        Ash) and every thread then hits LDS instead of re-reading A. For dense
        A this also avoids the code-size blow-up of baking every non-zero as an
        immediate (I-cache pressure).
    (2) B is reused across work-groups -> let it ride in the 256 MB Infinity
        Cache, and write C with the base.mako nt_store_c (non-temporal) so the
        write-once output does not evict the resident B.

  Only worth it when A is dense enough; gated by _cucoop_ok() in hip.py.
</%doc>
<%inherit file='base'/>

<%
# Flatten A's non-zeros row-major; aidx maps (row, col) -> index into Ash.
flat = []
aidx = {}
for j in range(m):
    for kx in range(k):
        if A[j, kx] != 0:
            aidx[(j, kx)] = len(flat)
            flat.append(A[j, kx])
nnz = len(flat)
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

    // A's non-zeros, baked as a compile-time constant and staged into LDS once.
    static const ${dtype} Aval[${nnz}] = { ${', '.join(str(float(v)) for v in flat)} };
    __shared__ ${dtype} Ash[${nnz}];
    ${dtype} bv, csub[${m}];

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
% for kx in bix:
        bv = b[i + ${kx}*ldb];   // streamed via Infinity Cache, reused across work-groups
  % for j in range(m):
    % if A[j, kx] != 0 and kx == afix[j] and beta == 0:
        csub[${j}] = Ash[${aidx[(j, kx)]}]*bv;
    % elif A[j, kx] != 0:
        csub[${j}] += Ash[${aidx[(j, kx)]}]*bv;
    % endif
    % if kx == alix[j]:
        nt_store_c(&c[i + ${j}*ldc], csub[${j}]);
    % endif
  % endfor
% endfor
% if beta == 0:
  % for j in range(m):
    % if afix[j] == -1:
        nt_store_c(&c[i + ${j}*ldc], make_zero());
    % endif
  % endfor
% endif
    }
}
