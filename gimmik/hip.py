# -*- coding: utf-8 -*-

import numpy as np

from gimmik.base import MatMul


class HIPMatMul(MatMul):
    platform = 'hip'
    basemeta = {'block': (128, 1, 1), 'width': 1, 'shared': 0}

    def _kernel_generators(self, dtype, dsize, *, gcn_arch=None, warp_size=64):
        max_block_threads = 1024
        max_shared = 64 * 1024

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

        # P_RT: Row-tile factor. 
        # Decouples the M dimension into spatial tiles across different blocks.
        # Original default/base: [2]
        P_RT = [2]

        # P_W: Vectorization width (Instruction level parallelism).
        # Locked to [2] to focus on memory bound optimization.
        P_W = []
        if self.aligne is not None and self.aligne % 2 == 0:
            P_W = [2]

        # --- 2. Dispatch Helper ---
        def emit(name, args, meta):
            # Unified hardware resource validation to prevent compilation failure
            threads = meta['block'][0] * meta['block'][1]
            shared = meta.get('shared', 0)
            if threads <= max_block_threads and shared <= max_shared:
                yield (name, args, meta)

        # --- 3. Sweep Combinations ---

        # Base Stream Series (cstream / bstream)
        for x in P_BLKX:
            yield from emit('cstream', {'blockx': x},
                            {'block': (x, 1, 1), 'desc': f'cstream/x{x}'})
            yield from emit('bstream', {'blockx': x},
                            {'block': (x, 1, 1), 'desc': f'bstream/x{x}'})
            yield from emit('cstream-preload-c', {'blockx': x},
                            {'block': (x, 1, 1), 'desc': f'cstream-preload-c/x{x}'})
            yield from emit('bstream-preload-c', {'blockx': x},
                            {'block': (x, 1, 1), 'desc': f'bstream-preload-c/x{x}'})

            for w in P_W:
                w_args = {'dtype': f'{dtype}{w}', 'width': w, 'blockx': x}
                yield from emit('cstream-width', w_args,
                                {'block': (x, 1, 1), 'width': w, 'desc': f'cstream-width/w{w}-x{x}'})
                yield from emit('cstream-width-preload-c', w_args,
                                {'block': (x, 1, 1), 'width': w, 'desc': f'cstream-width-preload-c/w{w}-x{x}'})
                yield from emit('bstream-width', w_args,
                                {'block': (x, 1, 1), 'width': w, 'desc': f'bstream-width/w{w}-x{x}'})
                yield from emit('bstream-width-preload-c', w_args,
                                {'block': (x, 1, 1), 'width': w, 'desc': f'bstream-width-preload-c/w{w}-x{x}'})

        # M-Split Series (bstream)
        for ms in P_MS:
            for bsz in P_BSZ:
                for x in P_BLKX:
                    shared = 2 * bsz * x * dsize
                    base_args = {'msplit': ms, 'bsz': bsz, 'blockx': x}
                    yield from emit('bstream-msplit', base_args,
                                    {'block': (x, ms, 1), 'shared': shared, 'desc': f'bstream-msplit/m{ms}-b{bsz}-x{x}'})
                    yield from emit('bstream-msplit-preload-c', base_args,
                                    {'block': (x, ms, 1), 'shared': shared, 'desc': f'bstream-msplit-preload-c/m{ms}-b{bsz}-x{x}'})

                    for w in P_W:
                        w_args = {**base_args, 'dtype': f'{dtype}{w}', 'width': w}
                        yield from emit('bstream-msplit-width', w_args,
                                        {'block': (x, ms, 1), 'width': w, 'shared': shared * w, 'desc': f'bstream-msplit-width/w{w}-m{ms}-b{bsz}-x{x}'})
                        yield from emit('bstream-msplit-width-preload-c', w_args,
                                        {'block': (x, ms, 1), 'width': w, 'shared': shared * w, 'desc': f'bstream-msplit-width-preload-c/w{w}-m{ms}-b{bsz}-x{x}'})

        # K-Split Series (cstream)
        for ks in P_KS:
            for csz in P_CSZ:
                for x in P_BLKX:
                    shared = (ks - 1) * csz * x * dsize
                    base_args = {'ksplit': ks, 'csz': csz, 'blockx': x}
                    yield from emit('cstream-ksplit', base_args,
                                    {'block': (x, ks, 1), 'shared': shared, 'desc': f'cstream-ksplit/k{ks}-c{csz}-x{x}'})
                    yield from emit('cstream-ksplit-preload-c', base_args,
                                    {'block': (x, ks, 1), 'shared': shared, 'desc': f'cstream-ksplit-preload-c/k{ks}-c{csz}-x{x}'})

                    for w in P_W:
                        w_args = {**base_args, 'dtype': f'{dtype}{w}', 'width': w}
                        yield from emit('cstream-ksplit-width', w_args,
                                        {'block': (x, ks, 1), 'width': w, 'shared': shared * w, 'desc': f'cstream-ksplit-width/w{w}-k{ks}-c{csz}-x{x}'})
                        yield from emit('cstream-ksplit-width-preload-c', w_args,
                                        {'block': (x, ks, 1), 'width': w, 'shared': shared * w, 'desc': f'cstream-ksplit-width-preload-c/w{w}-k{ks}-c{csz}-x{x}'})

        # Row-Tile Series (cstream)
        for ks in P_KS:
            for rt in P_RT:
                tile_size = (self.m + rt - 1) // rt
                for csz in P_CSZ:
                    if csz > tile_size:
                        continue
                    for x in P_BLKX:
                        shared = (ks - 1) * csz * x * dsize
                        base_args = {'ksplit': ks, 'rowtiles': rt, 'csz': csz, 'blockx': x}
                        yield from emit('cstream-ksplit-rowtile', base_args,
                                        {'block': (x, ks, 1), 'grid_y': rt, 'shared': shared, 'desc': f'cstream-ksplit-rowtile/k{ks}-rt{rt}-c{csz}-x{x}'})
                        for w in P_W:
                            w_args = {**base_args, 'dtype': f'{dtype}{w}', 'width': w}
                            yield from emit('cstream-ksplit-rowtile-width', w_args,
                                            {'block': (x, ks, 1), 'grid_y': rt, 'width': w, 'shared': shared * w, 'desc': f'cstream-ksplit-rowtile-width/w{w}-k{ks}-rt{rt}-c{csz}-x{x}'})

        # ---- MI300X / CDNA3 bandwidth-oriented kernels ----------------------
        is_cdna3 = gcn_arch is not None and str(gcn_arch).startswith('gfx94')

        # cstream-wide: vector-width N-blocking to saturate HBM3.
        vw = 4
        wide_ok = (
            (self.aligne is None or self.aligne % vw == 0)
            and (self.ldb is None or self.ldb % vw == 0)
            and (self.ldc is None or self.ldc % vw == 0)
            and (self.n is None or self.n % vw == 0)
        )
        if wide_ok:
            blkx = 256
            args = {'blockx': blkx, 'vw': vw}
            meta = {'block': (blkx, 1, 1), 'width': vw,
                    'desc': f'cstream-wide/vw{vw}-x{blkx}'}
            yield ('cstream-wide', args, meta)

        # bstream-cu-coop: A-stationary in LDS, B streamed through Infinity Cache.
        nnz = int(np.count_nonzero(self.A))
        if is_cdna3 and nnz*dsize <= 16*1024 and self.m <= 256:
            blkx = 256
            args = {'blockx': blkx}
            meta = {'block': (blkx, 1, 1),
                    'shared': nnz*dsize,
                    'desc': f'bstream-cu-coop/x{blkx}'}
            yield ('bstream-cu-coop', args, meta)

        # bstream-msplit-nt: global_load_lds fill + nontemporal C write. gfx940+.
        if is_cdna3:
            ms, bsz, blkx = 4, 24, 64
            args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
            meta = {'block': (blkx, ms, 1), 'shared': 2*bsz*blkx*dsize,
                    'desc': f'bstream-msplit-nt/m{ms}-x{blkx}'}
            yield ('bstream-msplit-nt', args, meta)

        # cstream-ksplit-nt: nontemporal B loads + C store. gfx940+.
        if is_cdna3:
            ks, csz, blkx = 2, 24, 64
            args = {'ksplit': ks, 'csz': csz, 'blockx': blkx}
            meta = {'block': (blkx, ks, 1), 'shared': (ks - 1)*csz*blkx*dsize,
                    'desc': f'cstream-ksplit-nt/k{ks}-x{blkx}'}
            yield ('cstream-ksplit-nt', args, meta)

        # cstream-wide-msplit: wide N-vectors + m-split rows. gfx940+.
        ms = 4
        wms_ok = (
            (self.aligne is None or self.aligne % vw == 0)
            and (self.ldb is None or self.ldb % vw == 0)
            and (self.ldc is None or self.ldc % vw == 0)
            and (self.n is None or self.n % vw == 0)
        )
        if is_cdna3 and wms_ok:
            blkx = 64
            args = {'blockx': blkx, 'msplit': ms, 'vw': vw}
            meta = {'block': (blkx, ms, 1), 'width': vw,
                    'desc': f'cstream-wide-msplit/m{ms}-vw{vw}-x{blkx}'}
            yield ('cstream-wide-msplit', args, meta)

        # cstream-wide-ksplit: wide N-vectors + k-split, nontemporal B loads. gfx940+.
        if is_cdna3 and wide_ok:
            ks, csz, blkx = 2, 24, 64
            args = {'ksplit': ks, 'csz': csz, 'blockx': blkx, 'vw': vw}
            meta = {'block': (blkx, ks, 1), 'width': vw,
                    'shared': (ks - 1)*csz*blkx*vw*dsize,
                    'desc': f'cstream-wide-ksplit/k{ks}-vw{vw}-x{blkx}'}
            yield ('cstream-wide-ksplit', args, meta)

        # mfma-dense: Matrix-Core dense f64 kernel. gfx90a+ and f64 only.
        gfx90a_plus = gcn_arch is not None and str(gcn_arch)[3:6] in (
            '90a', '940', '941', '942')
        if gfx90a_plus and dsize == 8 and self.m <= 256 and self.k <= 256:
            blkx = 64
            args = {'blockx': blkx}
            meta = {'block': (blkx, 1, 1),
                    'desc': f'mfma-dense/x{blkx}'}
            yield ('mfma-dense', args, meta)

    def _process_meta(self, meta):
        if self.n is not None:
            # Safely fetch width, defaulting to 1 if not vectorized
            div = meta['block'][0] * meta.get('width', 1)
            meta['grid'] = (-(-self.n // div), meta.get('grid_y', 1), 1)