<%doc>
  bstream-msplit-lds-sb — 單緩衝 (single-buffer) 版。
  與雙緩衝版相同 (B 經 __builtin_amdgcn_load_to_lds 直送 global->LDS、
  nt_store_c), scalar、無 preload-c。
  差別:bsub 由 [2] 改為 [1],不 prefetch 下一塊,改成每塊
  「填充 -> __syncthreads -> 計算 -> __syncthreads」。LDS 佔用減半
  (bsz*blockx*dsize[*width]),當 occupancy 受 LDS 限制時可換更多常駐
  wavefront,代價是失去 DMA/計算重疊、每塊多一次 barrier。
  注意:load_to_lds 為非同步 DMA,靠 __syncthreads() 作完成屏障;
  上機請驗證 LDS 讀取前 DMA 已完成。僅 CDNA (gfx94x)。
</%doc>
<%inherit file='base'/>

<%
mx = partition(A, into=msplit, by='rows')
bchunks = chunk(bix, bsz)
ndw = {'double': 2, 'double2': 4, 'double4': 8, 'float': 1, 'float2': 2, 'float4': 4}.get(dtype, max(1, len(dtype)))
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
    __shared__ ${dtype} bsub[1][${bsz}][${blockx}];

% for bb in range(len(bchunks)):
  ## Fill the single shared block with this chunk
  % for cid in range(msplit):
    if (i < n && threadIdx.y == ${cid})
    {
    % for kx in bchunks[bb]:
      % if loop.index % msplit == cid:
        for (int dw = 0; dw < ${ndw}; ++dw)
            __builtin_amdgcn_load_to_lds((void*)((const char*)(b + i + ${kx}*ldb) + 4*dw), (void*)((char*)&bsub[0][${loop.index}][threadIdx.x] + 4*dw), 4, 0, 0);
      % endif
    % endfor
    }
  % endfor
    __syncthreads();
  ## Accumulate our dot products from the single block
  % for cid, mcx in enumerate(mx):
    if (i < n && threadIdx.y == ${cid})
    {
    % for kx in bchunks[bb]:
        bv = bsub[0][${loop.index}][threadIdx.x];
      % for j, jx in enumerate(A[mcx, kx]):
        % if jx != 0 and kx == afix[mcx[j]]:
        csub[${j}] = ${jx}*bv;
        % elif jx != 0:
        csub[${j}] += ${jx}*bv;
        % endif
        % if kx == alix[mcx[j]] and beta == 0:
        nt_store_c(&c[i + ${mcx[j]}*ldc], csub[${j}]);
        % elif kx == alix[mcx[j]] and beta == 1:
        nt_store_c(&c[i + ${mcx[j]}*ldc], c[i + ${mcx[j]}*ldc] + csub[${j}]);
        % elif kx == alix[mcx[j]]:
        nt_store_c(&c[i + ${mcx[j]}*ldc], csub[${j}] + ${beta}*c[i + ${mcx[j]}*ldc]);
        % endif
      % endfor
    % endfor
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
