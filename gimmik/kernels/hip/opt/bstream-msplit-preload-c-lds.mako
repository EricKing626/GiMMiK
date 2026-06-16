<%doc>
  bstream-msplit-preload-c-lds
  以對應基底為本,只把 B 的雙緩衝填充改成 __builtin_amdgcn_load_to_lds:
  global -> LDS 直送,繞過 VGPR round-trip。其餘(preload-c 讀回、accumulate、
  nt_store_c 寫出、全零列處理)與基底完全一致。
  load_to_lds 支援 4/8/12/16B:double(8B)與 double2(16B)皆可直送
  (需 ROCm 6.x+)。非同步,依賴既有 __syncthreads();16B 傳輸需 16B 對齊。
  僅 CDNA(gfx90a/gfx94x)。
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
    __shared__ ${dtype} bsub[2][${bsz}][${blockx}];

## Fill the initial shared memory block
% for cid in range(msplit):
    if (i < n && threadIdx.y == ${cid})
    {
  % for kx in bchunks[0]:
    % if loop.index % msplit == cid:
        for (int dw = 0; dw < ${ndw}; ++dw)
            __builtin_amdgcn_load_to_lds((void*)((const char*)(b + i + ${kx}*ldb) + 4*dw), (void*)((char*)&bsub[0][${loop.index}][threadIdx.x] + 4*dw), 4, 0, 0);
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
        for (int dw = 0; dw < ${ndw}; ++dw)
            __builtin_amdgcn_load_to_lds((void*)((const char*)(b + i + ${kx}*ldb) + 4*dw), (void*)((char*)&bsub[${(bb + 1) % 2}][${loop.index}][threadIdx.x] + 4*dw), 4, 0, 0);
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
    __s