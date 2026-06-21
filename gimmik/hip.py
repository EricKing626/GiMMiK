# -*- coding: utf-8 -*-

import numpy as np

from gimmik.base import MatMul


class HIPMatMul(MatMul):
    platform = 'hip'
    basemeta = {'block': (128, 1, 1), 'width': 1, 'shared': 0}

    def _is_cdna3(self, gcn_arch):
        # MI300 series = gfx940/941/942. Real arch strings may carry feature
        # suffixes (e.g. 'gfx942:sramecc+:xnack-'), so match the token only.
        if gcn_arch is None:
            return False
        tok = str(gcn_arch).split(':', 1)[0]
        return tok in ('gfx940', 'gfx941', 'gfx942')

    def _is_cdna(self, gcn_arch):
        # CDNA2 (gfx90a) + CDNA3 (gfx94x/gfx950): the families that provide
        # __builtin_amdgcn_load_to_lds for the global->LDS B fill used by the
        # *-lds kernels. Match the gfxNNN token only (ignore feature suffixes).
        if gcn_arch is None:
            return False
        tok = str(gcn_arch).split(':', 1)[0]
        return tok in ('gfx90a', 'gfx940', 'gfx941', 'gfx942', 'gfx950')

    def _mfma_dense_ok(self, dsize):
        # mfma-dense is a COMPUTE-bound kernel: it densifies A and runs it
        # through the f64 Matrix Core. It only makes sense when A is dense
        # enough that the vector-ALU path is compute-limited (e.g. high-order
        # tet operators). For sparse A it would load zero-padding and lose to
        # the bandwidth kernels, so gate it behind a density threshold.
        nnz = int(np.count_nonzero(self.A))
        density = nnz / (self.A.shape[0] * self.A.shape[1])
        return density >= 0.5 and dsize == 8

    def _kernel_generators(self, dtype, dsize, *, gcn_arch=None, warp_size=64):
        max_block_threads = 1024
        max_shared = 64*1024

        def emit(name, args, meta):
            block = meta.get('block', self.basemeta['block'])
            shared = meta.get('shared', self.basemeta['shared'])
            threads = block[0]*block[1]*block[2]

            if threads <= max_block_threads and shared <= max_shared:
                yield (name, args, meta)

        blkx = self.basemeta['block'][0]

        # B loading, C streaming kernel
        yield from emit('cstream', {'blockx': blkx}, {})

        # B streaming, C accumulation kernel
        yield from emit('bstream', {'blockx': blkx}, {})

        # Four-way m-split B streaming, C accumulation kernel
        ms, bsz, blkx = 4, 24, 64
        args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
        meta = {'block': (blkx, ms, 1), 'shared': 2*bsz*blkx*dsize}
        yield from emit('bstream-msplit', args, meta)

        # Two-way k-split B loading, C streaming kernel
        ks, csz, blkx = 2, 24, 64
        args = {'ksplit': ks, 'csz': csz, 'blockx': blkx}
        meta = {'block': (blkx, ks, 1), 'shared': (ks - 1)*csz*blkx*dsize}
        yield from emit('cstream-ksplit', args, meta)

        # Tuned HIP variants
        msplits, ksplits = [4, 8], [2, 4]
        bsz, csz, blkx = 8, 8, 64
        width = 2 if self.aligne is not None and self.aligne % 2 == 0 else 1

        # The *-lds kernels stage B via a global->LDS DMA builtin that only
        # exists on CDNA (gfx90a / gfx94x); gate their emission on the arch.
        is_cdna = self._is_cdna(gcn_arch)

        # B loading, C streaming kernel
        args = {'blockx': blkx}
        meta = {'block': (blkx, 1, 1), 'desc': f'cstream/x{blkx}'}
        yield from emit('cstream', args, meta)

        # B streaming, C accumulation kernel
        meta = {'block': (blkx, 1, 1), 'desc': f'bstream/x{blkx}'}
        yield from emit('bstream', args, meta)

        for ms in msplits:
            # m-split B streaming, C accumulation kernel
            args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
            shared = 2*bsz*blkx*dsize
            meta = {'block': (blkx, ms, 1), 'shared': shared,
                    'desc': f'bstream-msplit/m{ms}-b{bsz}-x{blkx}'}
            yield from emit('bstream-msplit', args, meta)

        for ks in ksplits:
            # k-split B loading, C streaming kernel
            args = {'ksplit': ks, 'csz': csz, 'blockx': blkx}
            shared = (ks - 1)*csz*blkx*dsize
            meta = {'block': (blkx, ks, 1), 'shared': shared,
                    'desc': f'cstream-ksplit/k{ks}-c{csz}-x{blkx}'}
            yield from emit('cstream-ksplit', args, meta)

        # B loading, C preloading, C streaming kernel
        args = {'blockx': blkx}
        meta = {'block': (blkx, 1, 1), 'desc': f'cstream-preload-c/x{blkx}'}
        yield from emit('cstream-preload-c', args, meta)

        # B streaming, C preloading, C accumulation kernel
        meta = {'block': (blkx, 1, 1), 'desc': f'bstream-preload-c/x{blkx}'}
        yield from emit('bstream-preload-c', args, meta)

        if width > 1:
            args = {'dtype': f'{dtype}{width}', 'width': width,
                    'blockx': blkx}
            meta = {'block': (blkx, 1, 1), 'width': width,
                    'desc': f'cstream-width-preload-c/w{width}-x{blkx}'}
            yield from emit('cstream-width-preload-c', args, meta)

            meta = {'block': (blkx, 1, 1), 'width': width,
                    'desc': f'bstream-width-preload-c/w{width}-x{blkx}'}
            yield from emit('bstream-width-preload-c', args, meta)

        for ms in msplits:
            # m-split B streaming, C preloading, C accumulation kernel
            args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
            shared = 2*bsz*blkx*dsize
            meta = {'block': (blkx, ms, 1), 'shared': shared,
                    'desc': f'bstream-msplit-preload-c/m{ms}-b{bsz}-x{blkx}'}
            yield from emit('bstream-msplit-preload-c', args, meta)

            if width > 1:
                args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx,
                        'dtype': f'{dtype}{width}', 'width': width}
                meta = {
                    'block': (blkx, ms, 1), 'shared': shared*width,
                    'width': width,
                    'desc': (
                        f'bstream-msplit-width-preload-c/w{width}-'
                        f'm{ms}-b{bsz}-x{blkx}'
                    )
                }
                yield from emit('bstream-msplit-width-preload-c', args, meta)

            # LDS B-fill variants: same kernels, but B is staged straight
            # from global memory into LDS (global->LDS DMA), bypassing the
            # VGPR round-trip. CDNA only (gfx90a / gfx94x).
            if is_cdna:
                args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
                meta = {'block': (blkx, ms, 1), 'shared': shared,
                        'desc': f'bstream-msplit-preload-c-lds/m{ms}-b{bsz}-x{blkx}'}
                yield from emit('bstream-msplit-preload-c-lds', args, meta)

                if width > 1:
                    args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx,
                            'dtype': f'{dtype}{width}', 'width': width}
                    meta = {
                        'block': (blkx, ms, 1), 'shared': shared*width,
                        'width': width,
                        'desc': (
                            f'bstream-msplit-width-preload-c-lds/w{width}-'
                            f'm{ms}-b{bsz}-x{blkx}'
                        )
                    }
                    yield from emit('bstream-msplit-width-preload-c-lds', args, meta)

        for ks in ksplits:
            # k-split B loading, C preloading, C streaming kernel
            args = {'ksplit': ks, 'csz': csz, 'blockx': blkx}
            shared = (ks - 1)*csz*blkx*dsize
            meta = {
                'block': (blkx, ks, 1), 'shared': shared,
                'desc': f'cstream-ksplit-preload-c/k{ks}-c{csz}-x{blkx}'
            }
            yield from emit('cstream-ksplit-preload-c', args, meta)

            if width > 1:
                args = {'ksplit': ks, 'csz': csz, 'blockx': blkx,
                        'dtype': f'{dtype}{width}', 'width': width}
                meta = {
                    'block': (blkx, ks, 1), 'shared': shared*width,
                    'width': width,
                    'desc': (
                        f'cstream-ksplit-width-preload-c/w{width}-'
                        f'k{ks}-c{csz}-x{blkx}'
                    )
                }
                yield from emit('cstream-ksplit-width-preload-c', args, meta)

        # mfma-dense: f64 Matrix-Core dense kernel (CDNA3 / gfx94x only).
        # NOT a bandwidth strategy -- it densifies A and uses v_mfma_f64 to win
        # on COMPUTE throughput, which is the right tool only for dense, compute
        # -bound operators (high-order tet). Gated behind _mfma_dense_ok so it
        # never competes on the sparse, bandwidth-bound shapes.
        if self._is_cdna3(gcn_arch) and self._mfma_dense_ok(dsize):
            mblkx = 64                       # one wavefront = 64 lanes
            yield from emit('mfma-dense', {'blockx': mblkx},
                            {'block': (mblkx, 1, 1), 'width': 1,
                             'desc': f'mfma-dense/x{mblkx}'})

    def _process_meta(self, meta):
        if self.n is not None:
            div = meta['block'][0]*meta.get('width', 1)
            meta['grid'] = (-(-self.n // div), 1, 1)
