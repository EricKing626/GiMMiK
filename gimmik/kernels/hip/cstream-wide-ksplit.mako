<%inherit file='base'/>

## cstream-wide-ksplit — MI300X / CDNA3 orthogonal combination of two axes.
##
## Two independent optimisations on different loop dimensions:
##   * N axis (vector width): each thread processes vw consecutive output
##     columns as one wide vector (double4 / float4), so every B/C access is a
##     single global_load_dwordx4 that fills whole 128 B cache lines.
##   * K axis (k-split): the contraction dimension is partitioned across
##     threadIdx.y groups; each lane computes a partial wide dot product and
##     the partials are summed through LDS before the final store.
## The two act on orthogonal dimensions so they compose: full cache-line
## utilisation on the N reads *and* shorter dependency chains / more ILP from
## splitting K. B loads are non-temporal (streamed once, bypass L1); the LDS
## partial-sum buffer holds the reused data. Needs N (and leading dims) % vw==0.
<%
kparts = partition(A, ksplit, by='cols')
cchunks = chunk(range(m), csz)
vt = {2: dtype + '2', 4: dtype + '4'}[vw]
comps = ['x', 'y', 'z', 'w'][:vw]
loaded = set()
%>
__device__ __forceinline__ ${vt} fmav(${vt} acc, ${dtype} s, ${vt} b)
{
    ${' '.join(f'acc.{c} += s*b.{c};' for c in comps)} return acc;
}
__device__ __forceinline__ ${vt} addv(${vt} a, ${vt} b)
{
    ${' '.join(f'a.{c} += b.{c};' for c in comps)} return a;
}

__global__ __launch_bounds__(${blockx*ksplit}) void
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
    ${vt} cv[${-(-csz // ksplit)}], bv[${-(-k // ksplit)}], dotp;
    __shared__ ${vt} csub[${ksplit - 1}][${csz}][${blockx}];
% for cchunk in cchunks:
  % for bid, kbx in enumerate(kparts):
    if (i < n && threadIdx.y == ${bid})
    {
    % for j in cchunk:
      ## non-temporal wide load of B (streamed once -> bypass L1)
      % for kx in kbx:
        % if A[j, kx] != 0 and kx not in loaded:
        bv[${loop.index}] = __builtin_nontemporal_load(&b${vw}[i + ${kx}*ldb${vw}]); <% loaded.add(kx) %>
        % endif
      % endfor
<%      nz = [(kpos, A[j, kx]) for kpos, kx in enumerate(kbx) if A[j, kx] != 0] %>
      % if nz:
        dotp = ${vt}{${', '.join('0' for _ in comps)}};
        % for kpos, v in nz:
        dotp = fmav(dotp, ${v}, bv[${kpos}]);
        % endfor
      % else:
        dotp = ${vt}{${', '.join('0' for _ in comps)}};
      % endif
      % if loop.index % ksplit == bid:
        cv[${loop.index // ksplit}] = dotp;
      % else:
        csub[${bid - (bid > loop.index % ksplit)}][${loop.index}][threadIdx.x] = dotp;
      % endif
    % endfor
    }
  % endfor
    __syncthreads();
  % for bid, kbx in enumerate(kparts):
    if (i < n && threadIdx.y == ${bid})
    {
    % for j in cchunk:
      % if loop.index % ksplit == bid:
        dotp = cv[${loop.index // ksplit}];
        % for ii in range(ksplit - 1):
        dotp = addv(dotp, csub[${ii}][${loop.index}][threadIdx.x]);
        % endfor
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
      % endif
    % endfor
    }
  % endfor
    __syncthreads();
% endfor
}
