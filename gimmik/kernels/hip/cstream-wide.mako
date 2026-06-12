<%inherit file='base'/>

## cstream-wide — MI300X / CDNA3 bandwidth-saturating variant of cstream.
##
## Idea: each thread owns `vw` consecutive columns of B/C along the N axis and
## loads/stores them as a single wide vector (double4 / float4). One
## global_load_dwordx4 moves 32 B, so a 64-lane wavefront issues a contiguous
## 64*32 = 2 KiB burst that lands as whole 128 B HBM3 cache lines with zero
## gather. Launch 256 threads/block (4 wavefronts/CU) to keep enough in-flight
## loads to hide HBM latency and saturate the 5.3 TB/s bus.
##
## Requires N % vw == 0 and ldb/ldc % vw == 0 (the wrapper pads N up to vw).
<%
vt = {2: dtype + '2', 4: dtype + '4'}[vw]
comps = ['x', 'y', 'z', 'w'][:vw]
def vop(acc, val, b, first):
    op = '=' if first else '+='
    rhs = ' '.join(f'{acc}.{c} {("=" if first else "+=")} {val}*{b}.{c};' for c in comps)
    return rhs
%>
__global__ __launch_bounds__(${blockx}) void
% if n is None:
${kname}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
    n   = ((n + ${vw} - 1) / ${vw});
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

    if (i < n)
    {
% for j, jx in enumerate(A):
<%  nz = [(kx, v) for kx, v in enumerate(jx) if v != 0] %>
  % if nz:
    % for t, (kx, v) in enumerate(nz):
        ${' '.join(f'dotp.{c} {"=" if t == 0 else "+="} {v}*b{vw}[i + {kx}*ldb{vw}].{c};' for c in comps)}
    % endfor
  % else:
        ${' '.join(f'dotp.{c} = make_zero();' for c in comps)}
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
}
