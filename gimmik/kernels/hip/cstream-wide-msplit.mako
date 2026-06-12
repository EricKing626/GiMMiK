<%inherit file='base'/>

## cstream-wide-msplit — MI300X / CDNA3 orthogonal combination of two axes.
##
## Two independent optimisations on different loop dimensions:
##   * N axis (vector width): each thread processes vw consecutive output
##     columns as one wide vector (double4 / float4), so each B/C access is a
##     single global_load_dwordx4 that fills whole 128 B cache lines.
##   * M axis (m-split): the rows of C are partitioned across threadIdx.y
##     groups, so each thread only holds m/msplit accumulators instead of m.
## Because the two act on orthogonal dimensions they compose cleanly: low
## register pressure (from the row split) *and* full cache-line utilisation
## (from the wide vectors) at the same time. No LDS, so occupancy stays high.
##
## Requires N % vw == 0 and ldb/ldc % vw == 0.
<%
mx = partition(A, into=msplit, by='rows')
vt = {2: dtype + '2', 4: dtype + '4'}[vw]
comps = ['x', 'y', 'z', 'w'][:vw]
%>
__device__ __forceinline__ ${vt} fmav(${vt} acc, ${dtype} s, ${vt} b)
{
    ${' '.join(f'acc.{c} += s*b.{c};' for c in comps)} return acc;
}

__global__ __launch_bounds__(${blockx*msplit}) void
% if n is None:
${kname}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
    n = ((n + ${vw} - 1) / ${vw});
    const int ldb${vw} = ldb / ${vw}, ldc${vw} = ldc / ${vw};
% else:
${kname}(const ${dtype}* __restrict__ b, ${dtype}* __restrict__ c)
{
    const int n = ${-(-n // vw)};
    const int ldb${vw} = ${ldb // vw}, ldc${vw} = ${ldc // vw};
% endif
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const ${vt}* __restrict__ b${vw} = reinterpret_cast<const ${vt}*>(b);
    ${vt}* __restrict__ c${vw} = reinterpret_cast<${vt}*>(c);
    ${vt} dotp;

    if (i >= n) return;

% for cid in range(msplit):
    if (threadIdx.y == ${cid})
    {
  % for j in mx[cid]:
<%    nz = [(kx, v) for kx, v in enumerate(A[j]) if v != 0] %>
    % if nz:
        dotp = ${vt}{${', '.join('0' for _ in comps)}};
      % for kx, v in nz:
        dotp = fmav(dotp, ${v}, b${vw}[i + ${kx}*ldb${vw}]);
      % endfor
    % else:
        dotp = ${vt}{${', '.join('0' for _ in comps)}};
    % endif
    % if beta == 0:
        c${vw}[i + ${j}*ldc${vw}] = dotp;
    % elif beta == 1:
        { ${vt} cc = c${vw}[i + ${j}*ldc${vw}];
          ${' '.join(f'dotp.{c} += cc.{c};' for c in comps)} }
        c${vw}[i + ${j}*ldc${vw}] = dotp;
    % else:
        { ${vt} cc = c${vw}[i + ${j}*ldc${vw}];
          ${' '.join(f'dotp.{c} += {beta}*cc.{c};' for c in comps)} }
        c${vw}[i + ${j}*ldc${vw}] = dotp;
    % endif
  % endfor
    }
% endfor
}
