<%inherit file='base'/>

## cstream-w2: each thread processes two consecutive N columns (i*2 and i*2+1).
## Only valid for f64 (double) with n fixed at compile time and alignment >= 2.
## Mirrors ptx/cstream-w2.mako but using HIP C double2 vector loads.

<% ksplit = 2 if m < 36 else 1 %>

__global__ __launch_bounds__(128, 2) void
${kname}(const ${dtype}* __restrict__ b, ${dtype}* __restrict__ c)
{
    const int n  = ${-(-n // 2)};
    const ${'long long' if k*ldb >= 2**31 else 'int'} ldb = ${ldb};
    const ${'long long' if m*ldc >= 2**31 else 'int'} ldc = ${ldc};
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    ${dtype} dotp_a, dotp_b;

    if (i < n)
    {
% for j, jx in enumerate(A):
  % if (dotex := dot(lambda kx: f'__builtin_nontemporal_load(b + i*2 + {kx}*ldb)', jx, maxsplit=ksplit)) != '0.0':
        ## row ${j}: load pairs and compute two dot products simultaneously
        dotp_a = ${dotex};
        dotp_b = ${dot(lambda kx: f'__builtin_nontemporal_load(b + i*2 + 1 + {kx}*ldb)', jx, maxsplit=ksplit)};
  % else:
        dotp_a = make_zero();
        dotp_b = make_zero();
  % endif
  % if beta == 0:
        __builtin_nontemporal_store(dotp_a, c + i*2     + ${j}*ldc);
        __builtin_nontemporal_store(dotp_b, c + i*2 + 1 + ${j}*ldc);
  % elif beta == 1 and dotex != '0.0':
        c[i*2     + ${j}*ldc] += dotp_a;
        c[i*2 + 1 + ${j}*ldc] += dotp_b;
  % else:
        c[i*2     + ${j}*ldc] = dotp_a + ${beta}*c[i*2     + ${j}*ldc];
        c[i*2 + 1 + ${j}*ldc] = dotp_b + ${beta}*c[i*2 + 1 + ${j}*ldc];
  % endif
% endfor
    }
}
