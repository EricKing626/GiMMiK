<%inherit file='base'/>

## cstream-ksplit-nt — MI300X / CDNA3 bandwidth variant of cstream-ksplit.
##
## cstream-ksplit splits the contraction dimension K across threadIdx.y; each
## lane computes a partial dot product, partials are reduced through LDS, then
## C is written. Two bandwidth tweaks for CDNA3:
##   1. B loads use __builtin_nontemporal_load (GLC/SLC) so the streamed B
##      operands bypass L1 -- they are read once per lane and never reused, so
##      caching them only evicts the partial-sum data that *is* reused.
##   2. C is written with __builtin_nontemporal_store for the same reason.
## The LDS reduction of partial sums is unchanged. gfx940+.
<%
kparts = partition(A, ksplit, by='cols')
cchunks = chunk(range(m), csz)
loaded = set()
%>
__global__ __launch_bounds__(${blockx*ksplit}) void
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
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    ${dtype} cv[${-(-csz // ksplit)}], bv[${-(-k // ksplit)}], dotp;
    __shared__ ${dtype} csub[${ksplit - 1}][${csz}][${blockx}];
% for cchunk in cchunks:
  % for bid, kbx in enumerate(kparts):
    if (i < n && threadIdx.y == ${bid})
    {
    % for j in cchunk:
      ## nontemporal load of B (streamed once, no reuse -> bypass L1)
      % for kx in kbx:
        % if A[j, kx] != 0 and kx not in loaded:
        bv[${loop.index}] = __builtin_nontemporal_load(&b[i + ${kx}*ldb]); <% loaded.add(kx) %>
        % endif
      % endfor
      % if (dotex := dot(lambda kx: f'bv[{kx}]', A[j, kbx])) != '0.0':
        dotp = ${dotex};
      % else:
        dotp = make_zero();
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
        dotp = cv[${loop.index // ksplit}] + ${' + '.join(f'csub[{ii}][{loop.index}][threadIdx.x]'
                                                          for ii in range(ksplit - 1))};
        % if beta == 0:
        __builtin_nontemporal_store(dotp, &c[i + ${j}*ldc]);
        % elif beta == 1:
        c[i + ${j}*ldc] += dotp;
        % else:
        c[i + ${j}*ldc] = dotp + ${beta}*c[i + ${j}*ldc];
        % endif
      % endif
    % endfor
    }
  % endfor
    __syncthreads();
% endfor
}
