# -*- coding: utf-8 -*-

import numpy as np

from gimmik.base import MatMul

class HIPMatMul(MatMul):
    platform = 'hip'
    basemeta = {'block': (128, 1, 1), 'width': 1, 'shared': 0}

    def _colset_groups(self):
        groups = {}

        for j, row in enumerate(self.A):
            cols = tuple(i for i, v in enumerate(row) if v != 0)
            groups.setdefault(cols, []).append(j)

        return [(cols, rows) for cols, rows in groups.items()]

    def _has_grouped_bstream_pattern(self):
        groups = self._colset_groups()

        return (
            groups and
            all(len(cols) == 2 for cols, rows in groups) and
            all(len(rows) > 1 for cols, rows in groups)
        )

    def _cucoop_ok(self, dsize):
        # cu-coop stages A's non-zeros in LDS and keeps csub[m] in registers.
        # Only worthwhile when A is dense enough that baking it as immediates
        # would bloat the code; and only feasible when the LDS / register
        # footprint fits. nnz*dsize must fit a slice of LDS and m must be small
        # enough that csub[m] does not blow up VGPRs.
        nnz = int(np.count_nonzero(self.A))
        density = nnz / (self.A.shape[0] * self.A.shape[1])
        return (
            density >= 0.4
            and nnz * dsize <= 16 * 1024
            and self.m <= 256
        )

    def _kernel_generators(self, dtype, dsize, *, gcn_arch=None, warp_size=64):
        max_block_threads = 1024
        max_shared = 64 * 1024

        # Optional single-strategy filter, set by the bench harness via the
        # GIMMIK_ONLY env var (e.g. GIMMIK_ONLY=bstream-msplit-width-lds). When
        # set, only kernels whose template name / desc match are emitted; PyFR
        # then still benchmarks and tunes the parameter sweep *within* that one
        # strategy. Matches the tplname ('opt/foo'), its basename ('foo'), the
        # desc head ('foo' of 'foo/w2-m4-...'), the full desc, or any desc
        # prefix. Unset -> all strategies (default behaviour).
        import os
        # Fallback: if the caller didn't pass gcn_arch (some PyFR gimmik
        # providers call kernels(dtype) without it), read GIMMIK_GCN_ARCH so
        # the CDNA-gated strategies (*-lds, cu-coop*) still get emitted,
        # e.g. GIMMIK_GCN_ARCH=gfx942.
        if gcn_arch is None:
            gcn_arch = os.environ.get('GIMMIK_GCN_ARCH') or None
        _strat_only = os.environ.get('GIMMIK_ONLY') or None
        def _strat_match(name, desc):
            if not _strat_only:
                return True
            base = name.split('/')[-1]              # 'opt/foo' -> 'foo'
            head = desc.split('/')[0] if desc else ''   # 'foo/w2-..' -> 'foo'
            # Exact match only (no loose prefix): so 'cstream-width' selects
            # cstream-width and NOT cstream-width-preload-c. Pass a full desc
            # like 'bstream-msplit-width-lds/w2-m4-b16-x128' to pin one config.
            return _strat_only in (name, base, head, desc)

        # --- 1. Parameter Pool Definition ---
        
        # P_BLKX: Number of threads per block in the X dimension. 
        # Dictates the base occupancy and shared memory tile width.
        # Original default/base: [128]
        P_BLKX = [64, 128]

        # P_KS: K-split factor. 
        # Divides the inner product dimension (K) across threads to reduce register pressure.
        # Original default/base: [2]
        P_KS = [2, 4]

        # P_MS: M-split factor. 
        # Divides the matrix C rows (M) across threads in bstream to reduce register pressure.
        # Original default/base: [4]
        P_MS = [4, 8]             

        # P_CSZ: C chunk size. 
        # Number of C matrix row elements processed per thread in temporal loops.
        # Original default/base: [24] (adjusted to [12, 24] for sweep)
        P_CSZ = [8, 12, 24]       

        # P_BSZ: B chunk size. 
        # Number of B matrix elements loaded into shared memory per iteration in bstream.
        # Original default/base: [24] (adjusted to [16, 24] for sweep)
        P_BSZ = [8, 16, 24]       

        # P_W: Vectorization width (Instruction level parallelism).
        # Locked to [2] to focus on memory bound optimization.
        P_W = []
        if self.aligne is not None and self.aligne % 2 == 0:
            P_W = [2]

        # P_LDS: enable the global_load_lds B-fill variant of bstream-msplit.
        # Stages B straight from HBM into LDS (global -> LDS), bypassing the
        # VGPR round-trip of the default fill. CDNA only (gfx940+); scalar path
        # only (the builtin is dword-granular, so not combined with width>1).
        # Robust CDNA detection: real gcn_arch strings often carry feature
        # suffixes (e.g. 'gfx942:sramecc+:xnack-'), so match the gfxNNN token
        # only. load_to_lds / nontemporal exist on CDNA2 (gfx90a) and CDNA3
        # (gfx94x). NOTE: the width+lds path uses __builtin_amdgcn_load_to_lds,
        # which needs ROCm 6.x+ / a recent LLVM; if your toolchain is older,
        # restrict P_LDS to the scalar path or bump the toolchain.
        def _is_cdna(arch):
            if arch is None:
                return False
            tok = str(arch).split(':', 1)[0]
            return tok.startswith('gfx9') and tok[3:] in (
                '90a', '940', '941', '942', '950')
        is_cdna = _is_cdna(gcn_arch)
        P_LDS = [True] if is_cdna else []

        # --- 2. Dispatch Helper ---
        def emit(name, args, meta):
            # Optional single-strategy filter (GIMMIK_ONLY env var)
            if not _strat_match(name, meta.get('desc', '')):
                return
            # Unified hardware resource validation to prevent compilation failure
            blk = meta['block']
            threads = blk[0] * blk[1]
            shared = meta.get('shared', 0)
            if threads <= max_block_threads and shared <= max_shared:
                yield (name, args, meta)

        # --- 3. Core Templates ---
        for x in P_BLKX:
            yield from emit('core/cstream', {'blockx': x}, 
                            {'block': (x, 1, 1), 'desc': f'cstream/x{x}'})
            yield from emit('core/bstream', {'blockx': x}, 
                            {'block': (x, 1, 1), 'desc': f'bstream/x{x}'})

        for ms in P_MS:
            for bsz in P_BSZ:
                for x in P_BLKX:
                    shared = 2 * bsz * x * dsize
                    base_args = {'msplit': ms, 'bsz': bsz, 'blockx': x}
                    yield from emit('core/bstream-msplit', base_args, 
                                    {'block': (x, ms, 1), 'shared': shared, 'desc': f'bstream-msplit/m{ms}-b{bsz}-x{x}'})
                    # Bandwidth variant: B filled via global_load_lds (global->LDS,
                    # bypassing VGPRs). Same shared footprint; CDNA gfx940+ only.
                    for _ in P_LDS:
                        yield from emit('core/bstream-msplit-lds', base_args,
                                        {'block': (x, ms, 1), 'shared': shared, 'desc': f'bstream-msplit-lds/m{ms}-b{bsz}-x{x}'})

        for ks in P_KS:
            for csz in P_CSZ:
                for x in P_BLKX:
                    shared = (ks - 1) * csz * x * dsize
                    base_args = {'ksplit': ks, 'csz': csz, 'blockx': x}
                    yield from emit('core/cstream-ksplit', base_args, 
                                    {'block': (x, ks, 1), 'shared': shared, 'desc': f'cstream-ksplit/k{ks}-c{csz}-x{x}'})

        # --- 4. Opt Templates ---
        for x in P_BLKX:
            yield from emit('opt/cstream-preload-c', {'blockx': x},
                            {'block': (x, 1, 1), 'desc': f'cstream-preload-c/x{x}'})
            yield from emit('opt/bstream-preload-c', {'blockx': x},
                            {'block': (x, 1, 1), 'desc': f'bstream-preload-c/x{x}'})

            for w in P_W:
                w_args = {'dtype': f'{dtype}{w}', 'width': w, 'blockx': x}
                yield from emit('opt/cstream-width-preload-c', w_args,
                                {'block': (x, 1, 1), 'width': w, 'desc': f'cstream-width-preload-c/w{w}-x{x}'})
                yield from emit('opt/bstream-width-preload-c', w_args,
                                {'block': (x, 1, 1), 'width': w, 'desc': f'bstream-width-preload-c/w{w}-x{x}'})
                # naked width (no preload-c): pure double2/4 vectorized streaming
                yield from emit('opt/cstream-width', w_args,
                                {'block': (x, 1, 1), 'width': w, 'desc': f'cstream-width/w{w}-x{x}'})
                yield from emit('opt/bstream-width', w_args,
                                {'block': (x, 1, 1), 'width': w, 'desc': f'bstream-width/w{w}-x{x}'})

        # bstream-msplit
        for ms in P_MS:
            for bsz in P_BSZ:
                for x in P_BLKX:
                    shared = 2 * bsz * x * dsize
                    base_args = {'msplit': ms, 'bsz': bsz, 'blockx': x}
                    yield from emit('opt/bstream-msplit-preload-c', base_args,
                                    {'block': (x, ms, 1), 'shared': shared, 'desc': f'bstream-msplit-preload-c/m{ms}-b{bsz}-x{x}'})
                    # preload-c + LDS B-fill (load_to_lds, global->LDS). CDNA gfx94x only.
                    for _ in P_LDS:
                        yield from emit('opt/bstream-msplit-preload-c-lds', base_args,
                                        {'block': (x, ms, 1), 'shared': shared, 'desc': f'bstream-msplit-preload-c-lds/m{ms}-b{bsz}-x{x}'})

                    for w in P_W:
                        w_args = {**base_args, 'dtype': f'{dtype}{w}', 'width': w}
                        yield from emit('opt/bstream-msplit-width-preload-c', w_args,
                                        {'block': (x, ms, 1), 'width': w, 'shared': shared * w, 'desc': f'bstream-msplit-width-preload-c/w{w}-m{ms}-b{bsz}-x{x}'})
                        # width + preload-c + LDS B-fill. load_to_lds handles double2 (16B). CDNA only.
                        for _ in P_LDS:
                            yield from emit('opt/bstream-msplit-width-preload-c-lds', w_args,
                                            {'block': (x, ms, 1), 'width': w, 'shared': shared * w, 'desc': f'bstream-msplit-width-preload-c-lds/w{w}-m{ms}-b{bsz}-x{x}'})
                        # naked width (no preload-c)
                        yield from emit('opt/bstream-msplit-width', w_args,
                                        {'block': (x, ms, 1), 'width': w, 'shared': shared * w, 'desc': f'bstream-msplit-width/w{w}-m{ms}-b{bsz}-x{x}'})
                        # naked width + LDS B-fill (load_to_lds, no preload-c). CDNA only.
                        for _ in P_LDS:
                            yield from emit('opt/bstream-msplit-width-lds', w_args,
                                            {'block': (x, ms, 1), 'width': w, 'shared': shared * w, 'desc': f'bstream-msplit-width-lds/w{w}-m{ms}-b{bsz}-x{x}'})
        # cstream-ksplit
        for ks in P_KS:
            for csz in P_CSZ:
                for x in P_BLKX:
                    shared = (ks - 1) * csz * x * dsize
                    base_args = {'ksplit': ks, 'csz': csz, 'blockx': x}
                    yield from emit('opt/cstream-ksplit-preload-c', base_args,
                                    {'block': (x, ks, 1), 'shared': shared, 'desc': f'cstream-ksplit-preload-c/k{ks}-c{csz}-x{x}'})

                    for w in P_W:
                        w_args = {**base_args, 'dtype': f'{dtype}{w}', 'width': w}
                        yield from emit('opt/cstream-ksplit-width-preload-c', w_args,
                                        {'block': (x, ks, 1), 'width': w, 'shared': shared * w,
                                        'desc': f'cstream-ksplit-width-preload-c/w{w}-k{ks}-c{csz}-x{x}'})
                        # naked width (no preload-c)
                        yield from emit('opt/cstream-ksplit-width', w_args,
                                        {'block': (x, ks, 1), 'width': w, 'shared': shared * w,
                                        'desc': f'cstream-ksplit-width/w{w}-k{ks}-c{csz}-x{x}'})

        # --- 5. Special Templates ---
        if self._has_grouped_bstream_pattern():
            colset_groups = self._colset_groups()

            for x in P_BLKX:
                g_args = {'blockx': x, 'colset_groups': colset_groups}
                yield from emit('special/grouped-bstream-preload-c',
                                g_args,
                                {'block': (x, 1, 1),
                                 'desc': f'grouped-bstream-preload-c/x{x}'})

                for w in P_W:
                    gw_args = {
                        **g_args, 'dtype': f'{dtype}{w}', 'width': w
                    }
                    yield from emit('special/grouped-bstream-width-preload-c',
                                    gw_args,
                                    {'block': (x, 1, 1), 'width': w,
                                     'desc': f'grouped-bstream-width-preload-c/w{w}-x{x}'})

            for ms in P_MS:
                for x in P_BLKX:
                    g_args = {
                        'msplit': ms, 'blockx': x,
                        'colset_groups': colset_groups
                    }
                    yield from emit('special/grouped-bstream-msplit-preload-c',
                                    g_args,
                                    {'block': (x, ms, 1),
                                     'desc': f'grouped-bstream-msplit-preload-c/m{ms}-x{x}'})

                    for w in P_W:
                        gw_args = {
                            **g_args, 'dtype': f'{dtype}{w}', 'width': w
                        }
                        yield from emit('special/grouped-bstream-msplit-width-preload-c',
                                        gw_args,
                                        {'block': (x, ms, 1), 'width': w,
                                         'desc': f'grouped-bstream-msplit-width-preload-c/w{w}-m{ms}-x{x}'})

        # cu-coop: dense-A variant. Stage A's non-zeros in LDS (shared by the
        # whole work-group), stream B via Infinity Cache, write C non-temporally.
        # CDNA gfx940+ and dense A only (see _cucoop_ok).
        if is_cdna and self._cucoop_ok(dsize):
            nnz = int(np.count_nonzero(self.A))
            for x in P_BLKX:
                yield from emit('special/bstream-cu-coop', {'blockx': x},
                                {'block': (x, 1, 1), 'shared': nnz * dsize,
                                 'desc': f'bstream-cu-coop/x{x}'})
                # control / 對照: same kernel but A baked as immediates (no LDS staging)
                yield from emit('special/bstream-cu-coop-baked', {'blockx': x},
                                {'block': (x, 1, 1),
                                 'desc': f'bstream-cu-coop-baked/x{x}'})

    def _process_meta(self, meta):
        if self.n is not None:
            div = meta['block'][0]*meta['width']
            meta['grid'] = (-(-self.n // div), 1, 1)
