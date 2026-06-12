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
% if dtype.endswith('4'):
    __builtin_nontemporal_store(v.x, &p->x);
    __builtin_nontemporal_store(v.y, &p->y);
    __builtin_nontemporal_store(v.z, &p->z);
    __builtin_nontemporal_store(v.w, &p->w);
% elif dtype.endswith('2'):
    __builtin_nontemporal_store(v.x, &p->x);
    __builtin_nontemporal_store(v.y, &p->y);
% else:
    __builtin_nontemporal_store(v, p);
% endif
}

${next.body()}
