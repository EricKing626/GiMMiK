<%inherit file='base'/>

__global__ __launch_bounds__(${blockx}) void
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
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < n)
    {
% for gi, (cols, rows) in enumerate(colset_groups):
        {
  % for ci, col in enumerate(cols):
            ${dtype} bv${ci} = b[i + ${col}*ldb];
  % endfor
  % for ri, row in enumerate(rows):
            ${dtype} csub${ri} = ${' + '.join(f'{A[row, col]}*bv{ci}' for ci, col in enumerate(cols))};
    % if beta == 0:
            nt_store_c(&c[i + ${row}*ldc], csub${ri});
    % elif beta == 1:
            nt_store_c(&c[i + ${row}*ldc], c[i + ${row}*ldc] + csub${ri});
    % else:
            nt_store_c(&c[i + ${row}*ldc], csub${ri} + ${beta}*c[i + ${row}*ldc]);
    % endif
  % endfor
        }
% endfor
    }
}
