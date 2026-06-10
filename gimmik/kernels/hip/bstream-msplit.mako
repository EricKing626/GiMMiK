<%inherit file='base'/>

<%
mx = partition(A, into=msplit, by='rows')
bchunks = chunk(bix, bsz)
vec = (width == 2)
vtype = dtype + '2' if vec else dtype
# use_lds_async: direct global->LDS async copy via __builtin_amdgcn_global_load_lds.
# Available on GFX940+ (MI300X/MI300A). Mirrors PTX cp.async.ca.shared::cta.global.
# Caller passes use_lds_async=True only when gcn_arch starts with gfx94.
use_lds_async = locals().get('use_lds_async', False) and not vec
dsize_bytes   = 8 if dtype == 'double' else 4
%>

__global__ __launch_bounds__(${blockx*msplit}, 1) void
% if n is None:
${kname}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
  % if width > 1:
    n = ((n + ${width} - 1) / ${width}) * ${width};
    ldb /= ${width};
    ldc /= ${width};
  % endif
% else:
${kname}(const ${dtype}* __restrict__ b, ${dtype}* __restrict__ c)
{
    const int n = ${-(-n // width)};
    const ${'long long' if k*ldb >= width*2**31 else 'int'} ldb = ${ldb // width};
    const ${'long long' if m*ldc >= width*2**31 else 'int'} ldc = ${ldc // width};
% endif
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= n)
        return;

    ${vtype} bv;
    ${dtype} csub_x[${-(-m // msplit)}];
% if vec:
    ${dtype} csub_y[${-(-m // msplit)}];
% endif
    __shared__ ${vtype} bsub[2][${bsz}][${blockx}];

## Iterate over each row-chunk of C
% for cid, mcx in enumerate(mx):
    if (threadIdx.y == ${cid})
    {
  ## Iterate over each row-chunk of B
  % for bb in range(len(bchunks)):
    ## Fill the initial shared memory block
    % if loop.first:
      % for kx in bchunks[0]:
        % if loop.index % msplit == cid:
          % if vec:
        bsub[0][${loop.index}][threadIdx.x] = {
            __builtin_nontemporal_load(b + i*2     + ${kx}*ldb),
            __builtin_nontemporal_load(b + i*2 + 1 + ${kx}*ldb)};
          % elif use_lds_async:
        __builtin_amdgcn_global_load_lds(
            b + i + ${kx}*ldb,
            &bsub[0][${loop.index}][threadIdx.x],
            ${dsize_bytes}, 0, 0);
          % else:
        bsub[0][${loop.index}][threadIdx.x] = __builtin_nontemporal_load(b + i + ${kx}*ldb);
          % endif
        % endif
      % endfor
      % if use_lds_async:
        __builtin_amdgcn_s_waitcnt(0);
      % endif
        __syncthreads();
    % endif
    ## Start filling the next shared memory block (prefetch into alternate buf)
    % if not loop.last:
      % for kx in bchunks[bb + 1]:
        % if loop.index % msplit == cid:
          % if vec:
        bsub[${(bb + 1) % 2}][${loop.index}][threadIdx.x] = {
            __builtin_nontemporal_load(b + i*2     + ${kx}*ldb),
            __builtin_nontemporal_load(b + i*2 + 1 + ${kx}*ldb)};
          % elif use_lds_async:
        __builtin_amdgcn_global_load_lds(
            b + i + ${kx}*ldb,
            &bsub[${(bb + 1) % 2}][${loop.index}][threadIdx.x],
            ${dsize_bytes}, 0, 0);
          % else:
        bsub[${(bb + 1) % 2}][${loop.index}][threadIdx.x] = __builtin_nontemporal_load(b + i + ${kx}*ldb);
          % endif
        % endif
      % endfor
    % endif
    ## Accumulate our dot products
    % for kx in bchunks[bb]:
        bv = bsub[${bb % 2}][${loop.index}][threadIdx.x];
      % for j, jx in enumerate(A[mcx, kx]):
        % if jx != 0 and kx == afix[mcx[j]]:
          % if vec:
        csub_x[${j}] = ${jx}*bv.x;
        csub_y[${j}] = ${jx}*bv.y;
          % else:
        csub_x[${j}] = ${jx}*bv;
          % endif
        % elif jx != 0:
          % if vec:
        csub_x[${j}] += ${jx}*bv.x;
        csub_y[${j}] += ${jx}*bv.y;
          % else:
        csub_x[${j}] += ${jx}*bv;
          % endif
        % endif
        ## If we're done with this dot product then store to global
        % if kx == alix[mcx[j]] and beta == 0:
          % if vec:
        __builtin_nontemporal_store(csub_x[${j}], c + i*2     + ${mcx[j]}*ldc);
        __builtin_nontemporal_store(csub_y[${j}], c + i*2 + 1 + ${mcx[j]}*ldc);
          % else:
        __builtin_nontemporal_store(csub_x[${j}], c + i + ${mcx[j]}*ldc);
          % endif
        % elif kx == alix[mcx[j]] and beta == 1:
          % if vec:
        c[i*2     + ${mcx[j]}*ldc] += csub_x[${j}];
        c[i*2 + 1 + ${mcx[j]}*ldc] += csub_y[${j}];
          % else:
        c[i + ${mcx[j]}*ldc] += csub_x[${j}];
          % endif
        % elif kx == alix[mcx[j]]:
          % if vec:
        c[i*2     + ${mcx[j]}*ldc] = csub_x[${j}] + ${beta}*c[i*2     + ${mcx[j]}*ldc];
        c[i*2 + 1 + ${mcx[j]}*ldc] = csub_y[${j}] + ${beta}*c[i*2 + 1 + ${mcx[j]}*ldc];
          % else:
        c[i + ${mcx[j]}*ldc] = csub_x[${j}] + ${beta}*c[i + ${mcx[j]}*ldc];
          % endif
        % endif
      % endfor
    % endfor
      % if use_lds_async and not loop.last:
        __builtin_amdgcn_s_waitcnt(0);
      % endif
        __syncthreads();
  % endfor
  ## Handle rows of A which are all zero
  % for j, jx in enumerate(afix):
    % if jx == -1 and j % msplit == cid and beta == 0:
      % if vec:
        __builtin_nontemporal_store(make_zero(), c + i*2     + ${j}*ldc);
        __builtin_nontemporal_store(make_zero(), c + i*2 + 1 + ${j}*ldc);
      % else:
        __builtin_nontemporal_store(make_zero(), c + i + ${j}*ldc);
      % endif
    % elif jx == -1 and j % msplit == cid and beta != 1:
        c[i + ${j}*ldc] *= ${beta};
      % if vec:
        c[i*2 + 1 + ${j}*ldc] *= ${beta};
      % endif
    % endif
  % endfor
    }
% endfor
}
