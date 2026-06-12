% if dtype.endswith('4'):
static inline __device__ ${dtype} make_zero()
{ return make_${dtype}(0, 0, 0, 0); }
% elif dtype.endswith('2'):
static inline __device__ ${dtype} make_zero()
{ return make_${dtype}(0, 0); }
% else:
static inline __device__ ${dtype} make_zero()
{ return 0; }
% endif

static inline __device__ void
nt_store_c(${dtype}* p, ${dtype} v)
{
    __builtin_nontemporal_store(v, p);
}

${next.body()}
