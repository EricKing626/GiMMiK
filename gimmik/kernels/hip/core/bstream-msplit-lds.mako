<%doc>
  bstream-msplit-lds — bstream-msplit 的頻寬變體 (MI300X / CDNA3, gfx940+)
  B 的雙緩衝填充改用 __builtin_amdgcn_global_load_lds:直接 global -> LDS,
  繞過 VGPR (原本是 global -> VGPR -> LDS)。省一趟暫存器來回、降 VGPR 壓力、
  DMA 與 VALU 重疊。C 寫出沿用 base.mako 的 nt_store_c (non-temporal)。
  限制:global_load_lds 為 dword 粒度,僅用於 width==1 的純量路徑。
  注意:global_load_lds 為非同步 DMA。本 kernel 依賴既有的 __syncthreads()
  作為完成屏障;在真實 hipcc/MI300X 上請確認 LDS 讀取前 DMA 已完成
  (必要時於 __syncthreads() 前插入適當的 waitcnt / fence)。
</%doc>
<%inherit file='base'/>

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
        __builtin_amdgcn_global_load_lds((const uint32_t*)(b + i + ${kx}*ldb), (uint32_t*)&bsub[0][${loop.index}][threadIdx.x], sizeof(${dtype}), 0, 0);
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
        __builtin_amdgcn_global_load_lds((const uint32_t*)(b + i + ${kx}*ldb), (uint32_t*)&bsub[${(bb + 1) % 2}][${loop.index}][threadIdx.x], sizeof(${dtype}), 0, 0);
        % endif
      % endfor
    % endif
    ## Accumulate our dot products
    % for kx in bchunks[bb]:
        bv = bsub[${bb % 2}][${loop.index}][threadIdx.x];
      % for j, jx in enumerate(A[mcx, kx]):
        % if jx != 0 and kx == afix[mcx[j]]:
        csub[${j}] = ${jx}*bv;
        % elif jx != 0:
        csub[${j}] += ${jx}*bv;
        % endif
        ## If we're done with this dot product then store to global
        % if kx == alix[mcx[j]] and beta == 0:
        nt_store_c(&c[i + ${mcx[j]}*ldc], csub[${j}]);
        % elif kx == alix[mcx[j]] and beta == 1:
        nt_store_c(&c[i + ${mcx[j]}*ldc], c[i + ${mcx[j]}*ldc] + csub[${j}]);
        % elif kx == alix[mcx[j]]:
        nt_store_c(&c[i + ${mcx[j]}*ldc], csub[${j}] + ${beta}*c[i + ${mcx[j]}*ldc]);
        % endif
      % endfor
    % endfor
    ## Handle rows of A which are all zero
    % if loop.parent.last:
      % for j, jx in enumerate(afix):
        % if jx == -1 and j % msplit == cid and beta == 0:
        nt_store_c(&c[i + ${j}*ldc], make_zero());
        % elif jx == -1 and j % msplit == cid and beta != 1:
        nt_store_c(&c[i + ${j}*ldc], c[i + ${j}*ldc]*${beta});
        % endif
      % endfor
    % endif
    }
  % endfor
    __syncthreads();
% endfor
}
