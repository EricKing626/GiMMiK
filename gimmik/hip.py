# -*- coding: utf-8 -*-

import numpy as np

from gimmik.base import MatMul


class HIPMatMul(MatMul):
    platform = 'hip'
    basemeta = {'block': (128, 1, 1), 'width': 1, 'shared': 0}

    def _kernel_generators(self, dtype, dsize, *, gcn_arch=None, warp_size=64):
        # ---- Portable baseline kernels (all architectures) -----------------
        # B loading, C streaming kernel
        yield ('cstream', {}, {})

        # B streaming, C accumulation kernel
        yield ('bstream', {}, {})

        # Four-way m-split B streaming, C accumulation kernel
        ms, bsz, blkx = 4, 24, 64
        args = {'msplit': ms, 'bsz': bsz, 'blockx': blkx}
        meta = {'block': (blkx, ms, 1), 'shared': 2*bsz*blkx*dsize}
        yield ('bstream-msplit', args, meta)

        # Two-way k-split B loading, C streaming kernel
        ks, csz, blkx = 2, 24, 64
        args = {'ksplit': ks, 'csz': csz, 'blockx': blkx}
        meta = {'block': (blkx, ks, 1), 'shared': (ks - 1)*csz*blkx*dsize}
        yield ('cstream-ksplit', args, meta)

        # ====================================================================
        # MI300X / CDNA3 bandwidth-oriented kernels
        # --------------------------------------------------------------------
        # These target memory-bound shapes on Instinct parts and are emitted in
        # addition to the portable kernels above. PyFR's autotuner benchmarks
        # every candidate and keeps the fastest, so a bad fit for a given shape
        # simply loses the bench -- adding kernels never hurts correctness.
        #
        # They sit on a few orthogonal axes:
        #   * N axis      : double4 wide vectors -> fill whole 128 B cache lines
        #   * M / K axis  : split rows / contraction to cut register pressure
        #                   or shorten dependency chains
        #   * mem qualifier (orthogonal): nontemporal / global_load_lds keep
        #                   single-use streaming data out of L1 / VGPRs
        # ====================================================================
        is_cdna3 = gcn_arch is not None and str(gcn_arch).startswith('gfx94')

        # vw=4 wide kernels need N (and the leading dims) to be a multiple of
        # the vector width. With a compile-time n we also require n % vw == 0 so
        # the bounds math drops out. This single gate covers every wide kernel.
        vw = 4
        wide_ok = (
            (self.aligne is None or self.aligne % vw == 0)
            and (self.ldb is None or self.ldb % vw == 0)
            and (self.ldc is None or self.ldc % vw == 0)
            and (self.n is None or self.n % vw == 0)
        )

        # --- Wide family: cache-line coalescing on the N axis ---------------
        # cstream-wide: plain vector-width N-blocking. double4 = 32 B/thread, a
        # quad of lanes fills whole 128 B cache lines. 256 threads = 4
        # wavefronts/CU for latency hiding. The memory-bound workhorse.
        if wide_ok:
            blkx = 256
            yield ('cstream-wide',
                   {'blockx': blkx, 'vw': vw},
                   {'block': (blkx, 1, 1), 'width': vw,
                    'desc': f'cstream-wide/vw{vw}-x{blkx}'})

        # cstream-wide-msplit: wide N-vectors + m-split over rows. The row split
        # relieves the register pressure the double4 accumulators add, so it
        # wins when m is large. No LDS -> occupancy stays high.
        if is_cdna3 and wide_ok:
            ms, blkx = 4, 64
            yield ('cstream-wide-msplit',
                   {'blockx': blkx, 'msplit': ms, 'vw': vw},
                   {'block': (blkx, ms, 1), 'width': vw,
                    'desc': f'cstream-wide-msplit/m{ms}-vw{vw}-x{blkx}'})

        # cstream-wide-ksplit: wide N-vectors + k-split over the contraction.
        # Shorter dependency chains / more ILP, wins when k is large. B loads
        # are non-temporal; partial sums reduced through an LDS buffer of wide
        # vectors. Supersedes a plain ksplit-nt (this *is* ksplit-nt + vectors).
        if is_cdna3 and wide_ok:
            ks, csz, blkx = 2, 24, 64
            yield ('cstream-wide-ksplit',
                   {'ksplit': ks, 'csz': csz, 'blockx': blkx, 'vw': vw},
                   {'block': (blkx, ks, 1), 'width': vw,
                    'shared': (ks - 1)*csz*blkx*dsize*vw,
                    'desc': f'cstream-wide-ksplit/k{ks}-vw{vw}-x{blkx}'})

        # --- Cache-discipline family: keep streaming data out of L1 ---------
        # bstream-msplit-nt: bandwidth-tuned bstream-msplit. The B double-buffer
        # fill is pure streaming, so fill it global->LDS via global_load_lds
        # (bypasses VGPRs, frees registers, overlaps with VALU) and write C
        # non-temporally to keep streamed data out of L1.
        if is_cdna3:
            ms, bsz, blkx = 4, 24, 64
            yield ('bstream-msplit-nt',
                   {'msplit': ms, 'bsz': bsz, 'blockx': blkx},
                   {'block': (blkx, ms, 1), 'shared': 2*bsz*blkx*dsize,
                    'desc': f'bstream-msplit-nt/m{ms}-x{blkx}'})

        # bstream-cu-coop: A is a constant operator shared by the whole
        # work-group -> stage it into LDS once. B is reused across work-groups
        # -> let it ride in the 256 MB Infinity Cache, and write C
        # non-temporally so the write-once output does not evict the resident
        # B. Best when A's non-zeros fit LDS and m is small enough that csub[m]
        # fits VGPRs.
        nnz = int(np.count_nonzero(self.A))
        if is_cdna3 and nnz*dsize <= 16*1024 and self.m <= 256:
            blkx = 256
            yield ('bstream-cu-coop',
                   {'blockx': blkx},
                   {'block': (blkx, 1, 1), 'shared': nnz*dsize,
                    'desc': f'bstream-cu-coop/x{blkx}'})

        # --- Compute-bound (NOT a bandwidth strategy) -----------------------
        # mfma-dense: Matrix-Core dense kernel (f64). Densifies A and runs it
        # through mfma_f64_16x16x4f64. DISABLED: it trades *more* bandwidth
        # (loading the zero-padded dense A) for compute throughput, which is the
        # opposite of the bandwidth goal here. Re-enable for dense operators
        # once compute, not memory, is the bottleneck.
        #
        # gfx90a_plus = gcn_arch is not None and str(gcn_arch)[3:6] in (
        #     '90a', '940', '941', '942')
        # if gfx90a_plus and dsize == 8 and self.m <= 256 and self.k <= 256:
        #     blkx = 64
        #     yield ('mfma-dense',
        #            {'blockx': blkx},
        #            {'block': (blkx, 1, 1), 'desc': f'mfma-dense/x{blkx}'})

    def _process_meta(self, meta):
        if self.n is not None:
            div = meta['block'][0]*meta['width']
            meta['grid'] = (-(-self.n // div), 1, 1)
