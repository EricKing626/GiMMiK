<%inherit file='base'/>
<%doc>
  bstream-msplit-ntile: per-thread n-tiling for memory-level parallelism.
  Each thread owns ${ntile} independent B columns (spaced blockDim.x apart so
  each tile stays coalesced) and issues their loads together -> more outstanding
  HBM requests in flight, which helps saturate bandwidth on memory-bound
  operators. The tile dimension is UNROLLED in mako so every bv[t]/csub[t][j]
  uses compile-time-constant indices (-> registers, not a runtime-indexed local
  array which can trip the AMDGPU register coalescer). `pos` captures the chunk
  index so the unroll does not clobber mako's loop.index. LDS (bsub) and
  registers scale x ntile; emit() filters combos exceeding the LDS budget.
  UNVERIFIED on hardware -- validate accuracy + tune on MI300X.
</%doc>

<%
mx = partition(A, into=msplit, by='rows')
bchunks = chunk(bix, bsz)
mrows = -(-m // msplit)
%>

__global__ __launch_bounds__(${blockx*msplit}) void
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
    const int col0 = blockDim.x*${ntile}*blockIdx.x + threadIdx.x;

    ${dtype} bv[${ntile}], csub[${ntile}][${mrows}];
    __shared__ ${dtype} bsub[2][${bsz}][${blockx*ntile}];

## Fill the initial shared memory block (all ntile tiles)
% for cid in range(msplit):
    if (threadIdx.y == ${cid})
    {
  % for kx in bchunks[0]:
    % if loop.index % msplit == cid:
<% pos = loop.index %>
      % for t in range(ntile):
        if (col0 + ${t}*blockDim.x < n)
            bsub[0][${pos}][${t}*blockDim.x + threadIdx.x] = b[col0 + ${t}*blockDim.x + ${kx}*ldb];
      % endfor
    % endif
  % endfor
    }
% endfor
    __syncthreads();

## Iterate over each row-chunk of B
% for bb in range(len(bchunks)):
  % for cid, mcx in enumerate(mx):
    if (threadIdx.y == ${cid})
    {
    ## Prefetch the next chunk (all tiles)
    % if not loop.parent.last:
      % for kx in bchunks[bb + 1]:
        % if loop.index % msplit == cid:
<% pos = loop.index %>
          % for t in range(ntile):
        if (col0 + ${t}*blockDim.x < n)
            bsub[${(bb + 1) % 2}][${pos}][${t}*blockDim.x + threadIdx.x] = b[col0 + ${t}*blockDim.x + ${kx}*ldb];
          % endfor
        % endif
      % endfor
    % endif
    ## Load all ntile B values, then accumulate
    % for kx in bchunks[bb]:
<% pos = loop.index %>
      % for t in range(ntile):
        bv[${t}] = bsub[${bb % 2}][${pos}][${t}*blockDim.x + threadIdx.x];
      % endfor
      % for j, jx in enumerate(A[mcx, kx]):
        % if jx != 0 and kx == afix[mcx[j]]:
          % for t in range(ntile):
        csub[${t}][${j}] = ${jx}*bv[${t}];
          % endfor
        % elif jx != 0:
          % for t in range(ntile):
        csub[${t}][${j}] += ${jx}*bv[${t}];
          % endfor
        % endif
        % if kx == alix[mcx[j]] and beta == 0:
          % for t in range(ntile):
        if (col0 + ${t}*blockDim.x < n)
            nt_store_c(&c[col0 + ${t}*blockDim.x + ${mcx[j]}*ldc], csub[${t}][${j}]);
          % endfor
        % elif kx == alix[mcx[j]] and beta == 1:
          % for t in range(ntile):
        if (col0 + ${t}*blockDim.x < n)
            nt_store_c(&c[col0 + ${t}*blockDim.x + ${mcx[j]}*ldc], nt_load_c(&c[col0 + ${t}*blockDim.x + ${mcx[j]}*ldc]) + csub[${t}][${j}]);
          % endfor
        % elif kx == alix[mcx[j]]:
          % for t in range(ntile):
        if (col0 + ${t}*blockDim.x < n)
            nt_store_c(&c[col0 + ${t}*blockDim.x + ${mcx[j]}*ldc], csub[${t}][${j}] + ${beta}*nt_load_c(&c[col0 + ${t}*blockDim.x + ${mcx[j]}*ldc]));
          % endfor
        % endif
      % endfor
    % endfor
    ## Handle rows of A which are all zero
    % if loop.parent.last:
      % for j, jx in enumerate(afix):
        % if jx == -1 and j % msplit == cid and beta == 0:
          % for t in range(ntile):
        if (col0 + ${t}*blockDim.x < n)
            nt_store_c(&c[col0 + ${t}*blockDim.x + ${j}*ldc], make_zero());
          % endfor
        % elif jx == -1 and j % msplit == cid and beta != 1:
          % for t in range(ntile):
        if (col0 + ${t}*blockDim.x < n)
            nt_store_c(&c[col0 + ${t}*blockDim.x + ${j}*ldc], nt_load_c(&c[col0 + ${t}*blockDim.x + ${j}*ldc])*${beta});
          % endfor
        % endif
      % endfor
    % endif
    }
  % endfor
    __syncthreads();
% endfor
}
