<%inherit file='base'/>

## bstream-msplit-nt — MI300X / CDNA3 bandwidth variant of bstream-msplit.
##
## bstream-msplit double-buffers chunks of B through LDS. That fill path is
## pure streaming: each B element is read once and never reused, so routing it
## through L1 only pollutes the cache. Two changes:
##   1. Fill bsub via __builtin_amdgcn_global_load_lds (gfx940+): a direct
##      global->LDS DMA that bypasses the VGPR file entirely (global->VGPR->LDS
##      becomes global->LDS). This cuts VGPR pressure -> higher wavefront
##      occupancy, and the async fill overlaps with VALU, like PTX cp.async.
##   2. Write C with __builtin_nontemporal_store (GLC/SLC) so the write-once
##      output does not evict reusable data from L1/L2.
## Falls back to plain msplit semantics where the builtins are unavailable.
<%
mx = partition(A, into=msplit, by='rows')
bchunks = chunk(bix, bsz)
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
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    ${dtype} bv, csub[${-(-m // msplit)}];
    __shared__ ${dtype} bsub[2][${bsz}][${blockx}];

## Macro: direct global->LDS fill (no VGPR round-trip)
<%def name="lds_fill(buf, idx, kx)">\
        if (i < n)
            __builtin_amdgcn_global_load_lds(
                reinterpret_cast<const uint32_t*>(&b[i + ${kx}*ldb]),
                reinterpret_cast<uint32_t*>(&bsub[${buf}][${idx}][threadIdx.x]),
                sizeof(${dtype}), 0, 0);
</%def>\
## Fill the initial shared-memory block directly from global memory
% for cid in range(msplit):
    if (threadIdx.y == ${cid})
    {
  % for kx in bchunks[0]:
    % if loop.index % msplit == cid:
${lds_fill(0, loop.index, kx)}\
    % endif
  % endfor
    }
% endfor
    __syncthreads();

## Main loop: consume current buffer, prefetch next directly into LDS
% for cid in range(msplit):
    if (threadIdx.y == ${cid})
    {
  % for bb, bchunk in enumerate(bchunks):
    ## Prefetch next chunk (global -> LDS) while we compute on this one
    % if bb < len(bchunks) - 1:
      % for kx in bchunks[bb + 1]:
        % if loop.index % msplit == cid:
${lds_fill((bb + 1) % 2, loop.index, kx)}\
        % endif
      % endfor
    % endif
    % for ci, kx in enumerate(bchunk):
        bv = bsub[${bb % 2}][${ci}][threadIdx.x];
      <% rows = [r for r in mx[cid]] %>
      % for jj, j in enumerate(rows):
        % if A[j, kx] != 0 and kx == afix[j]:
        csub[${jj}] = ${A[j, kx]}*bv;
        % elif A[j, kx] != 0:
        csub[${jj}] += ${A[j, kx]}*bv;
        % endif
        % if kx == alix[j] and beta == 0:
        __builtin_nontemporal_store(csub[${jj}], &c[i + ${j}*ldc]);
        % elif kx == alix[j] and beta == 1:
        c[i + ${j}*ldc] += csub[${jj}];
        % elif kx == alix[j]:
        c[i + ${j}*ldc] = csub[${jj}] + ${beta}*c[i + ${j}*ldc];
        % endif
      % endfor
    % endfor
        __syncthreads();
  % endfor
  % if beta == 0:
    % for j in mx[cid]:
      % if afix[j] == -1:
        __builtin_nontemporal_store(make_zero(), &c[i + ${j}*ldc]);
      % endif
    % endfor
  % endif
    }
% endfor
}
