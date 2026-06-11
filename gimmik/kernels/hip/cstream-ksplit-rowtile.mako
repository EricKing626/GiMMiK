<%inherit file='base'/>

<%
kparts = partition(A, ksplit, by='cols')

# 1. 計算每個 Block (Row-Tile) 負責的總列數
tile_size = -(-m // rowtiles)

# 2. 建立雙層結構：Tiles (給 Block) -> Chunks (給內部迴圈)
tiles = []
for cid in range(rowtiles):
    start = cid * tile_size
    end = min((cid + 1) * tile_size, m)
    if start >= end:
        continue
    
    # 將這個 Block 負責的列，依照 csz 切成多個時間分段 (Temporal Chunks)
    chunks_in_tile = []
    for c_start in range(start, end, csz):
        c_end = min(c_start + csz, end)
        chunks_in_tile.append(list(range(c_start, c_end)))
        
    tiles.append((cid, chunks_in_tile))
%>

__global__ __launch_bounds__(${blockx*ksplit}) void
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

    ## cv 與 csub 的大小現在嚴格綁定在 csz (Chunk Size)，大幅降低硬體壓力
    ${dtype} cv[${-(-csz // ksplit)}], bv[${-(-k // ksplit)}], dotp;
    __shared__ ${dtype} csub[${ksplit - 1}][${csz}][${blockx}];

## 1. 外層：迭代所有的 Row Tiles (由不同的 Block 平行處理)
% for cid, chunks_in_tile in tiles:
  if (blockIdx.y == ${cid})
  {
  ## 2. 內層：迭代這個 Block 內部的時間分段 (Chunks)
  % for cidx, cchunk in enumerate(chunks_in_tile):
    <% loaded = set() %>
    ## 計算部分內積
    % for bid, kbx in enumerate(kparts):
      if (i < n && threadIdx.y == ${bid})
      {
      % for j in cchunk:
        % for kx in kbx:
          % if A[j, kx] != 0 and kx not in loaded:
          bv[${loop.index}] = b[i + ${kx}*ldb]; <% loaded.add(kx) %>
          % endif
        % endfor
        
        % if (dotex := dot(lambda kx: f'bv[{kx}]', A[j, kbx])) != '0.0':
          dotp = ${dotex};
        % else:
          dotp = make_zero();
        % endif
        
        ## 寫入暫存器或 Shared Memory (索引基於 local chunk index: loop.index)
        % if loop.index % ksplit == bid:
          cv[${loop.index // ksplit}] = dotp;
        % else:
          csub[${bid - (bid > loop.index % ksplit)}][${loop.index}][threadIdx.x] = dotp;
        % endif
      % endfor
      }
    % endfor
    
      __syncthreads();
    
    ## 合併並寫出這個 Chunk 的結果
    % for bid, kbx in enumerate(kparts):
      if (i < n && threadIdx.y == ${bid})
      {
      % for j in cchunk:
        % if loop.index % ksplit == bid:
          dotp = cv[${loop.index // ksplit}] + ${' + '.join(f'csub[{i}][{loop.index}][threadIdx.x]'
                                                            for i in range(ksplit - 1))};
          % if beta == 0:
          c[i + ${j}*ldc] = dotp;
          % elif beta == 1:
          c[i + ${j}*ldc] += dotp;
          % else:
          c[i + ${j}*ldc] = dotp + ${beta}*c[i + ${j}*ldc];
          % endif
        % endif
      % endfor
      }
    % endfor
    
    ## 如果這個 Block 還有下一個 Chunk 要算，就必須 Sync 以免下一輪提早覆寫 Shared Memory
    % if cidx < len(chunks_in_tile) - 1:
      __syncthreads();
    % endif
  % endfor
  }
% endfor
}