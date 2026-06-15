<%inherit file='base'/>
<%doc>
  bstream-msplit-preload-c-lds
  ============================
  = repo 的 bstream-msplit-preload-c(scalar,無 width),B 雙緩衝填充改用
  __builtin_amdgcn_load_to_lds(global -> LDS,繞過 VGPR)。preload-c 沿用
  repo 版骨架。
  注意:load_to_lds 為非同步 DMA;本檔依賴既有的 __syncthreads() 作為完成屏障。
  真實 hipcc/MI300X 上請確認 LDS 讀取前 DMA 已完成(必要時補 waitcnt/fence)。
  load_to_lds 支援 4/8/12/16 byte:scalar double=8B、double2=16B 皆可。僅 CDNA(gfx94x)。
</%doc>

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

    ${dtype} bv, csub[${-(-m // msplit)}];
    __shared__ ${dtype} bsub[2][${bsz}][${blockx}];

## Fill the initial shared memory block
% for cid in range(msplit):
    if (i < n && threadIdx.y == ${cid})
    {
  % for kx in bchunks[0]:
    % if loop.index % msplit == cid:
        __builtin_amdgcn_load_to_lds((const ${dtype}*)(b + i + ${kx}*ldb), (${dtype}*)&bsub[0][${loop.index}][threadIdx.x], sizeof(${dtype}), 0, 0);
    % endif
  % endfor

  ## Preload C values for active rows owned by this m-split lane
  % for j, jx in enumerate(mx[cid]):
    % if afix[jx] != -1:
      % if beta == 0:
        csub[${j}] = make_zero();
      % elif beta == 1:
        csub[${j}] = c[i + ${jx}*ldc];
      % else:
        csub[${j}] = ${beta}*c[i + ${jx}*ldc];
      % endif
    % endif
  % endfor
    }
% endfor
    __syncthreads();

## Iterate over each row-chunk of B
% for bb in range(len(bchunks)):
  ## Iterate over each row-chunk of C
  % for cid, mcx in enumerate(mx):
    if (i < n && threadIdx.y == ${cid})
    {
    ## Start filling the next shared memory block
    % if not loop.parent.last:
      % for kx in bchunks[bb + 1]:
        % if loop.index % msplit == cid:
        __builtin_amdgcn_load_to_lds((const ${dtype}*)(b + i + ${kx}*ldb), (${dtype}*)&bsub[${(bb + 1) % 2}][${loop.index}][threadIdx.x], sizeof(${dtype}), 0, 0);
        % endif
      % endfor
    % endif
    ## Accumulate our dot products
    % for kx in bchunks[bb]:
        bv = bsub[${bb % 2}][${loop.index}][threadIdx.x];
      % for j, jx in enumerate(A[mcx, kx]):
        % if jx != 0:
        csub[${j}] += ${jx}*bv;
        % endif
        ## If we're done with this dot product then store to global
        % if kx == alix[mcx[j]]:
        nt_store_c(&c[i + ${mcx[j]}*ldc], csub[${j}]);
        % endif
      % endfor
    % endfor
    ## Handle rows of A which are all zero
    % if loop.parent.last:
      % for j, jx in enumerate(afix):
        % if jx == -1 and j % msplit == cid and beta == 0:
        nt_store_c(&c[i + ${j}*ldc], make_zero());
        % elif jx == -1 and j % msplit == cid and beta != 1:
        nt_store_c(&c[i + ${j}*ldc], ${beta}*c[i + ${j}*ldc]);
        % endif
      % endfor
    % endif
    }
  % endfor
    __syncthreads();
% endfor
}
