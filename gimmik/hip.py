# -*- coding: utf-8 -*-

import numpy as np

from gimmik.base import MatMul

# GCN architectures that are CDNA-family (MI100/MI200/MI300 series)
_CDNA_PREFIXES = ('gfx908', 'gfx90a', 'gfx940', 'gfx941', 'gfx942')

# GFX940+ supports __builtin_amdgcn_global_load_lds (direct global->LDS async)
_LDS_ASYNC_PREFIXES = ('gfx940', 'gfx941', 'gfx942')

# GFX90a+ (MI200) and GFX940+ (MI300) support MFMA f64 16x16x4
_MFMA_F64_PREFIXES = ('gfx90a', 'gfx940', 'gfx941', 'gfx942')


def _is_cdna(gcn_arch):
    return gcn_arch is not None and gcn_arch.startswith(_CDNA_PREFIXES)


def _has_lds_async(gcn_arch):
    return gcn_arch is not None and gcn_arch.startswith(_LDS_ASYNC_PREFIXES)


def _has_mfma_f64(gcn_arch):
    return gcn_arch is not None and gcn_arch.startswith(_MFMA_F64_PREFIXES)


class HIPMatMul(MatMul):
    platform = 'hip'
    basemeta = {'block': (128, 1, 1), 'width': 1, 'shared': 0}

    @staticmethod
    def is_suitable(arr):
        nnz = np.count_nonzero(arr)
        nuq = len(np.unique(np.abs(arr[arr != 0])))
        density = nnz / arr.size
        # Unrolled FMA trees are only efficient when A has few unique
        # coefficient magnitudes (so constants can be shared) or is very
        # sparse (so few operations are emitted per B element).
        return (nuq <= 28) or (density <= 0.15)

    def _kernel_generators(self, dtype, dsize, *, gcn_arch=None, warp_size=64):
        cdna      = _is_cdna(gcn_arch)
        lds_async = _has_lds_async(gcn_arch)
        mfma_f64  = _has_mfma_f64(gcn_arch) and dtype == 'double'
        K_used    = len(self.bix)

        # B loading, C streaming kernel
        yield ('cstream', {}, {'desc': 'cstream'})

        # cstream-w2: two consecutive N columns per thread (f64 only, fixed n).
        # Mirrors ptx/cstream-w2; requires aligned n (aligne divisible by 2).
        if dtype == 'double' and self.n is not None and self.aligne and self.aligne % 2 == 0:
            yield ('cstream-w2', {'width': 2},
                   {'width': 2, 'desc': 'cstream-w2'})

        # B streaming, C accumulation kernel
        yield ('bstream', {}, {'desc': 'bstream'})

        # Four-way m-split B streaming, C accumulation kernel.
        # MI300X (gfx942): wave64, 64 KB LDS per CU, large register file —
        # use a larger bsz and 8-way msplit to fill 64-wide waves efficiently.
        ms   = 8 if cdna else 4
        bsz  = 32 if (cdna and dtype == 'float') else 24
        blkx = 64
        args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
        meta = {'block': (blkx, ms, 1), 'shared': 2*bsz*blkx*dsize,
                'desc': f'bstream-msplit/m{ms}-b{bsz}-x{blkx}'}
        yield ('bstream-msplit', args, meta)

        # width=2 variant: each thread processes 2 consecutive N indices using
        # two __builtin_nontemporal_load calls instead of reinterpret_cast.
        args_w2 = {'msplit': ms, 'bsz': bsz, 'blockx': blkx, 'width': 2}
        meta_w2 = {'block': (blkx, ms, 1), 'shared': 2*bsz*blkx*dsize,
                   'width': 2, 'desc': f'bstream-msplit/m{ms}-b{bsz}-x{blkx}-w2'}
        yield ('bstream-msplit', args_w2, meta_w2)

        # LDS-async variant (GFX940+ / MI300): uses __builtin_amdgcn_global_load_lds
        # to pipeline global->LDS transfer with VALU, mirroring PTX cp.async.
        if lds_async:
            args_la = {'msplit': ms, 'bsz': bsz, 'blockx': blkx,
                       'use_lds_async': True}
            meta_la = {'block': (blkx, ms, 1), 'shared': 2*bsz*blkx*dsize,
                       'desc': f'bstream-msplit/m{ms}-b{bsz}-x{blkx}-ldsasync'}
            yield ('bstream-msplit', args_la, meta_la)

        # Standard four-way m-split as an additional candidate on CDNA.
        if cdna:
            ms2, bsz2, blkx2 = 4, 24, 64
            args2 = {'msplit': ms2, 'bsz': bsz2, 'blockx': blkx2}
            meta2 = {'block': (blkx2, ms2, 1), 'shared': 2*bsz2*blkx2*dsize,
                     'desc': f'bstream-msplit/m{ms2}-b{bsz2}-x{blkx2}'}
            yield ('bstream-msplit', args2, meta2)

        # Single-wavefront bstream-msplit for large-K operators.
        # msplit=1 means all 64 threads cooperate to stage a bigger B tile
        # (bsz=32 rows) in shared memory, halving the number of sync barriers
        # compared to the multi-way msplit variants.  Only worthwhile when
        # K is large enough for the double-buffering to hide latency.
        if cdna and K_used >= 64:
            ms1, bsz1, blkx1 = 1, 32, 64
            args1 = {'msplit': ms1, 'bsz': bsz1, 'blockx': blkx1}
            meta1 = {'block': (blkx1, ms1, 1), 'shared': 2*bsz1*blkx1*dsize,
                     'desc': f'bstream-msplit/m{ms1}-b{bsz1}-x{blkx1}'}
            yield ('bstream-msplit', args1, meta1)

        # Four-way k-split B loading, C streaming kernel (CDNA / large K).
        # Emitting 4-way only makes sense when K is large enough that the
        # extra partial-result shared memory and sync cost is amortised.
        if cdna and K_used >= 64:
            ks, csz, blkx = 4, 24, 64
            args = {'ksplit': ks, 'csz': csz, 'blockx': blkx}
            meta = {'block': (blkx, ks, 1), 'shared': (ks - 1)*csz*blkx*dsize,
                    'desc': f'cstream-ksplit/k{ks}-c{csz}-x{blkx}'}
            yield ('cstream-ksplit', args, meta)

        # Two-way k-split — default for all targets, always a candidate.
        ks2, csz2, blkx2 = 2, 24, 64
        args2 = {'ksplit': ks2, 'csz': csz2, 'blockx': blkx2}
        meta2 = {'block': (blkx2, ks2, 1), 'shared': (ks2 - 1)*csz2*blkx2*dsize,
                 'desc': f'cstream-ksplit/k{ks2}-c{csz2}-x{blkx2}'}
        yield ('cstream-ksplit', args2, meta2)

        # MFMA-based dense matmul (GFX90a+ f64 only).
        # Uses mfma_f64_16x16x4 (wavefront-64). A embedded as .global constant.
        # One wavefront per CTA, each covering a 16-wide N tile.
        if mfma_f64 and self.n is not None:
            NCOL = 16
            blkx_mfma = 64
            n_ctas = -(-self.n // NCOL)
            args_mfma = {'blockx': blkx_mfma}
            meta_mfma = {'block': (blkx_mfma, 1, 1),
                         'grid':  (n_ctas, 1, 1),
                         'shared': 0,
                         'desc': 'dmma/mfma-f64-16x16x4'}
            yield ('dmma', args_mfma, meta_mfma)

    def _process_meta(self, meta):
        if self.n is not None and 'grid' not in meta:
            div = meta['block'][0]*meta['width']
            meta['grid'] = (-(-self.n // div), 1, 1)
