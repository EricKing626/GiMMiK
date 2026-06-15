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
#error "grouped_bstream_msplit_width_preload_c only supports width=2 or width=4"
% endif

__global__ __launch_bounds__(${blockx*msplit}) void
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

    if (i < n)
    {
% for gi, (cols, rows) in enumerate(colset_groups):
        if (threadIdx.y == ${gi % msplit})
        {
  % for ci, col in enumerate(cols):
            ${dtype} bv${ci} = b[i + ${col}*ldb];
  % endfor
  % for ri, row in enumerate(rows):
            ${dtype} csub${ri} = gimmik_vmul(${A[row, cols[0]]}, bv0);
    % for ci, col in enumerate(cols[1:], start=1):
            csub${ri} = gimmik_vmadd(csub${ri}, ${A[row, col]}, bv${ci});
    % endfor
    % if beta == 0:
            nt_store_c(&c[i + ${row}*ldc], csub${ri});
    % elif beta == 1:
            ${dtype} cval${ri} = c[i + ${row}*ldc];
            nt_store_c(&c[i + ${row}*ldc], gimmik_vadd(cval${ri}, csub${ri}));
    % else:
            ${dtype} cval${ri} = c[i + ${row}*ldc];
            nt_store_c(&c[i + ${row}*ldc], gimmik_vadd(csub${ri}, gimmik_vmul(${beta}, cval${ri})));
    % endif
  % endfor
        }
% endfor
    }
}
