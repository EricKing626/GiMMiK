<%inherit file='base'/>

% if width == 2:
static inline __device__ ${dtype}
gimmik_vmul(${dtype[:-1]} a, ${dtype} b)
{
    return make_${dtype}(a*b.x, a*b.y);
}

static inline __device__ ${dtype}
gimmik_vadd(${dtype} a, ${dtype} b)
{
    return make_${dtype}(a.x + b.x, a.y + b.y);
}

static inline __device__ ${dtype}
gimmik_vmadd(${dtype} acc, ${dtype[:-1]} a, ${dtype} b)
{
    return make_${dtype}(acc.x + a*b.x, acc.y + a*b.y);
}
% elif width == 4:
static inline __device__ ${dtype}
gimmik_vmul(${dtype[:-1]} a, ${dtype} b)
{
    return make_${dtype}(a*b.x, a*b.y, a*b.z, a*b.w);
}

static inline __device__ ${dtype}
gimmik_vadd(${dtype} a, ${dtype} b)
{
    return make_${dtype}(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static inline __device__ ${dtype}
gimmik_vmadd(${dtype} acc, ${dtype[:-1]} a, ${dtype} b)
{
    return make_${dtype}(acc.x + a*b.x, acc.y + a*b.y, acc.z + a*b.z, acc.w + a*b.w);
}
% else:
#error "cstream_ksplit_rowtile_width only supports width=2 or width=4"
% endif

<%
kparts = partition(A, ksplit, by='cols')

# Row-Tile 雙層解耦結構
tile_size = -(-m // rowtiles)
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
    n = (n + ${width} - 1) / ${width};
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

    ${dtype} cv[${-(-csz // ksplit)}], bv[${-(-k // ksplit)}], dotp;
    __shared__ ${dtype} csub[${ksplit - 1}][${csz}][${blockx}];

% for cid, chunks_in_tile in tiles:
  if (blockIdx.y == ${cid})
  {
  % for cidx, cchunk in enumerate(chunks_in_tile):
    <% loaded = set() %>
    % for bid, kbx in enumerate(kparts):
      if (i < n && threadIdx.y == ${bid})
      {
      % for j in cchunk:
        % for kx in kbx:
          % if A[j, kx] != 0 and kx not in loaded:
          bv[${loop.index}] = b[i + ${kx}*ldb]; <% loaded.add(kx) %>
          % endif
        % endfor
        
        ## 展開向量化的內積算式
        <%
        nzixs = []
        for l_idx, kx in enumerate(kbx):
            if A[j, kx] != 0:
                nzixs.append((l_idx, kx))
                
        if not nzixs:
            dotex = 'make_zero()'
        else:
            first_l_idx, first_kx = nzixs[0]
            dotex = f"gimmik_vmul({A[j, first_kx]}, bv[{first_l_idx}])"
            for l_idx, kx in nzixs[1:]:
                dotex = f"gimmik_vmadd({dotex}, {A[j, kx]}, bv[{l_idx}])"
        %>
        dotp = ${dotex};

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
          ## 展開向量化的 Shared Memory 合併算式
          <%
          sum_expr = f"cv[{loop.index // ksplit}]"
          for s_idx in range(ksplit - 1):
              sum_expr = f"gimmik_vadd({sum_expr}, csub[{s_idx}][{loop.index}][threadIdx.x])"
          %>
          dotp = ${sum_expr};

          ## 寫回 Global Memory
          % if beta == 0:
          c[i + ${j}*ldc] = dotp;
          % elif beta == 1:
          c[i + ${j}*ldc] = gimmik_vadd(c[i + ${j}*ldc], dotp);
          % else:
          c[i + ${j}*ldc] = gimmik_vadd(dotp, gimmik_vmul(${beta}, c[i + ${j}*ldc]));
          % endif
        % endif
      % endfor
      }
    % endfor
    
    ## 避免提早覆寫 Shared Memory
    % if cidx < len(chunks_in_tile) - 1:
      __syncthreads();
    % endif
  % endfor
  }
% endfor
}