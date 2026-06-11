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
    // Keep the multiply-add expression visible to the compiler.
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
    // Keep the multiply-add expression visible to the compiler.
    return make_${dtype}(acc.x + a*b.x, acc.y + a*b.y, acc.z + a*b.z, acc.w + a*b.w);
}
% else:
#error "cstream_width only supports width=2 or width=4"
% endif

__global__ __launch_bounds__(${blockx}) void
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
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    ${dtype} bv, dotp;

    if (i < n)
    {
% for j, row in enumerate(A):
  <%
  # 預先過濾出該列的非零元素與其索引
  nzixs = [kx for kx, val in enumerate(row) if val != 0]
  %>
  % if not nzixs:
        dotp = make_zero();
  % else:
    <% first_kx = nzixs[0] %>
        bv = b[i + ${first_kx}*ldb];
        dotp = gimmik_vmul(${row[first_kx]}, bv);
    % for kx in nzixs[1:]:
        bv = b[i + ${kx}*ldb];
        dotp = gimmik_vmadd(dotp, ${row[kx]}, bv);
    % endfor
  % endif

  ## 根據 beta 參數進行輸出邏輯判定
  % if beta == 0:
        c[i + ${j}*ldc] = dotp;
  % elif beta == 1 and nzixs:
        c[i + ${j}*ldc] = gimmik_vadd(c[i + ${j}*ldc], dotp);
  % else:
        c[i + ${j}*ldc] = gimmik_vadd(dotp, gimmik_vmul(${beta}, c[i + ${j}*ldc]));
  % endif
% endfor
    }
}