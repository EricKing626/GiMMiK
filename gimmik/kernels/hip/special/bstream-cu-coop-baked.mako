<%doc>
  bstream-cu-coop-baked — CONTROL / 對照組 for bstream-cu-coop.

  Identical to bstream-cu-coop in every respect (dense-A only, B streamed via
  the 256 MB Infinity Cache, C written non-temporally with nt_store_c, scalar
  width=1, same blockx, same _cucoop_ok() gating) EXCEPT A's non-zeros are baked
  inline as compile-time immediates instead of being cooperatively staged into
  LDS (Ash[]) and read back.

  Purpose: isolate the single variable that defines cu-coop — does staging A's
  non-zeros in LDS actually beat plain immediate-baking? — by holding everything
  else (B/C handling, register csub[m], block config, gating) constant. cu-coop
  only wins if the LDS reads + one-time cooperative staging beat the I-cache /
  immediate pressure of baking the constants inline. Uses zero shared memory.
</%doc>
<%inherit file='base'/>
__global__ __launch_bounds__(${blockx}) void
% if n is None:
${kname}(int n,
         const ${dtype}* __restrict__ b, int ldb,
         ${dtype}* __restrict__ c, int ldc)
{
% else:
${kname}(const ${dtype}* __restrict__ b, ${dtype}* __restrict__ c)
{
    const int n = ${n};
    const ${'long long' if k*ldb >= 2**31 else 'int'} ldb = ${ldb};
    const ${'long long' if m*ldc >= 2**31 else 'int'} ldc = ${ldc};
% endif
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    ${dtype} bv, csub[${m}];

    if (i < n)
    {
% if beta != 0:
  % for j in range(m):
    % if afix[j] != -1:
        csub[${j}] = ${'' if beta == 1 else f'{beta}*'}c[i + ${j}*ldc];
    % endif
  % endfor
% endif
% for kx in bix:
        bv = b[i + ${kx}*ldb];   // streamed via Infinity Cache, reused across work-groups
  % for j in range(m):
    % if A[j, kx] != 0 and kx == afix[j] and beta == 0:
        csub[${j}] = ${A[j, kx]}*bv;   // A baked as immediate (control, no LDS)
    % elif A[j, kx] != 0:
        csub[${j}] += ${A[j, kx]}*bv;
    % endif
    % if kx == alix[j]:
        nt_store_c(&c[i + ${j}*ldc], csub[${j}]);
    % endif
  % endfor
% endfor
% if beta == 0:
  % for j in range(m):
    % if afix[j] == -1:
        nt_store_c(&c[i + ${j}*ldc], make_zero());
    % endif
  % endfor
% endif
    }
}
