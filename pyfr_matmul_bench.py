#!/usr/bin/env python
import argparse
import csv
import os
import socket
import sys

import numpy as np
from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from scipy.io import mmread


ELES = ['hex', 'pri', 'tet']
ORDERS = [2, 3, 4, 5]
MATS = ['m0', 'm3', 'm6', 'm132', 'm460']
TARGET_GIB = 4


def parse_int_list(s):
    return [int(v) for v in s.split(',')]


def parse_str_list(s):
    return s.split(',')


def mat_path(basedir, p, ele, mat):
    return os.path.join(basedir, f'p{p}', ele, f'{mat}-sp.mtx')


def order_sizes(basedir, orders, eles, mats):
    sizes = {}

    for p in orders:
        rows_cols = []

        for ele in eles:
            for mat in mats:
                m, k = mmread(mat_path(basedir, p, ele, mat)).shape
                rows_cols.append(m + k)

        sizes[p] = max(rows_cols)

    return sizes


def order_ns(backend, basedir, orders, eles, mats, target_gib):
    target_bytes = target_gib*1024**3
    dsize = np.dtype(backend.fpdtype).itemsize

    return {
        p: max(1, int(target_bytes // (rows_cols*dsize)))
        for p, rows_cols in order_sizes(basedir, orders, eles, mats).items()
    }


def benchmark(backend, a_np, n, beta):
    m, k = a_np.shape
    dsize = np.dtype(backend.fpdtype).itemsize

    a_be = backend.const_matrix(a_np)
    b_be = backend.matrix((k, n), tags={'align'})
    c_be = backend.matrix((m, n), tags={'align'})

    kern = backend.kernel('mul', a_be, b_be, c_be, beta=beta)
    prov = kernel_provenance(backend, kern, a_np)

    for _ in range(10):
        backend.run_kernels([kern], wait=False)
    backend.wait()

    fp_dense = 2*m*n*k / kern.dt / 1024**3
    fp_sparse = 2*n*np.count_nonzero(a_np) / kern.dt / 1024**3
    bw = (m + (m if beta else 0) + k)*n*dsize / kern.dt / 1024**3

    return fp_dense, fp_sparse, bw, prov


def kernel_provenance(backend, kern, a_np):
    kcls = type(kern)
    mod = kcls.__module__
    qname = kcls.__qualname__
    token = '{}.{}'.format(mod, qname).lower()

    if 'gimmik' in token:
        source = 'gimmik'
        kind = gimmik_kind(backend, a_np)
    elif 'cublas' in token:
        source = 'blas'
        kind = 'cublaslt'
    elif 'rocblas' in token:
        source = 'blas'
        kind = 'rocblas'
    else:
        source = 'unknown'
        kind = ''

    return {
        'kernel_source': source,
        'kernel_kind': kind,
        'kernel_variant': getattr(kern, 'kernel_variant', ''),
        'kernel_module': mod,
        'kernel_qualname': qname
    }


def gimmik_kind(backend, a_np):
    if backend.name == 'cuda':
        try:
            from gimmik import CUDAMatMul, PTXMatMul
        except ImportError:
            try:
                from gimmik import CUDAMatMul
            except ImportError:
                return 'unknown'
            else:
                if hasattr(CUDAMatMul, 'is_suitable'):
                    if CUDAMatMul.is_suitable(a_np):
                        return 'cuda'
                    else:
                        return 'unknown'
                else:
                    return 'cuda'
        else:
            cc = backend.cuda.compute_capability()
            if PTXMatMul.is_suitable(a_np, cc):
                return 'ptx'
            elif CUDAMatMul.is_suitable(a_np):
                return 'cuda'
            else:
                return 'unknown'
    elif backend.name == 'hip':
        return 'hip'
    else:
        return ''


def gcn_arch(backend):
    # Best-effort discovery of the GCN arch (e.g. 'gfx942') so that the LDS /
    # cu-coop strategies (CDNA-gated) are actually emitted when dumping source.
    # Falls back to None (HIPMatMul then skips the CDNA-only kernels).
    for attr in ('props', 'device_props', 'dev_props'):
        d = getattr(backend, attr, None)
        if isinstance(d, dict):
            for key in ('gcnArchName', 'gcn_arch', 'gcnArch', 'arch'):
                v = d.get(key)
                if v:
                    return str(v)
    return None


def dump_kernel_src(backend, a_np, p, ele, mat, outdir):
    os.makedirs(outdir, exist_ok=True)

    try:
        from gimmik import HIPMatMul
        dtype = np.dtype(backend.fpdtype)
        mm = HIPMatMul(a_np)
        for src, meta in mm.kernels(dtype, gcn_arch=gcn_arch(backend)):
            tplname = meta.get('tplname', 'unknown') if isinstance(meta, dict) else 'unknown'
            fname = os.path.join(outdir, f'p{p}_{ele}_{mat}_{tplname}.hip')
            with open(fname, 'w') as f:
                f.write(src)
            print(f'[dump-kernel-src] wrote {fname}', file=sys.stderr)
    except Exception as ex:
        print(f'[dump-kernel-src] failed for p{p}/{ele}/{mat}: {ex}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('backend')
    parser.add_argument('precision')
    parser.add_argument('n')
    parser.add_argument('basedir')
    parser.add_argument('--out')
    parser.add_argument('--label', default='')
    parser.add_argument('--orders', type=parse_int_list, default=ORDERS)
    parser.add_argument('--eles', type=parse_str_list, default=ELES)
    parser.add_argument('--mats', type=parse_str_list, default=MATS)
    parser.add_argument('--target-gib', type=float, default=TARGET_GIB)
    parser.add_argument('--filter-order', type=parse_int_list, default=None)
    parser.add_argument('--filter-ele', type=parse_str_list, default=None)
    parser.add_argument('--filter-mat', type=parse_str_list, default=None)
    parser.add_argument('--dump-kernel-src', metavar='DIR', default=None)
    parser.add_argument('--disable-rocblas', action='store_true')
    parser.add_argument(
        '--strategy', default=None, metavar='NAME',
        help='Restrict GiMMiK to a single HIP strategy (sets GIMMIK_ONLY). '
             'PyFR still tunes the parameter sweep within it. Match by strategy '
             'name, e.g. bstream-msplit-width-lds, cstream-width, '
             'bstream-cu-coop, bstream-cu-coop-baked; or pin one config with a '
             'full desc like bstream-msplit-width-lds/w2-m4-b16-x128. Note: '
             'CDNA-only strategies (any *-lds, cu-coop*) need a HIP backend on '
             'gfx94x, and width strategies need an even aligne, else the chosen '
             'strategy yields zero kernels.')
    args = parser.parse_args()

    # Pin a single GiMMiK strategy (read by gimmik/hip.py _kernel_generators).
    # Must be set before the backend generates/benchmarks any kernel.
    if args.strategy:
        os.environ['GIMMIK_ONLY'] = args.strategy
        print(f'[strategy] restricting GiMMiK to: {args.strategy}',
              file=sys.stderr)

    #inistr = f'''
    #[backend]
    #precision = {args.precision}
    #'''
    inistr = f'''
    [backend]
    precision = {args.precision}
    autotune-ifac = 1.0
    
    [backend-hip]
    disable-rocblas = {str(args.disable_rocblas).lower()}    
    gimmik-nkerns = 256
    gimmik-nbench = 5
    '''
    ini = Inifile(inistr)
    backend = get_backend(args.backend, ini)

    # Help the CDNA-gated GiMMiK strategies (*-lds, cu-coop*) get emitted
    # even if PyFR does not pass gcn_arch into HIPMatMul.kernels(): expose
    # the device arch via GIMMIK_GCN_ARCH (gimmik/hip.py reads it as a
    # fallback). Does not override a value already set in the environment.
    if backend.name == 'hip' and not os.environ.get('GIMMIK_GCN_ARCH'):
        _arch = gcn_arch(backend)
        if _arch:
            os.environ['GIMMIK_GCN_ARCH'] = _arch
            print(f'[strategy] GIMMIK_GCN_ARCH={_arch}', file=sys.stderr)

    if args.n == 'auto':
        ns = order_ns(backend, args.basedir, args.orders, args.eles,
                      args.mats, args.target_gib)
    else:
        ns = {p: int(args.n) for p in args.orders}

    fieldnames = [
        'label', 'host', 'backend', 'precision', 'N', 'p', 'ele', 'mat',
        'working_set_gib', 'kernel_source', 'kernel_kind',
        'kernel_variant', 'kernel_module', 'kernel_qualname',
        'fp_dense', 'fp_sparse', 'bw'
    ]

    stream = open(args.out, 'w', newline='') if args.out else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        for p in args.orders:
            if args.filter_order and p not in args.filter_order:
                continue
            for e in args.eles:
                if args.filter_ele and e not in args.filter_ele:
                    continue
                for mat in args.mats:
                    if args.filter_mat and mat not in args.filter_mat:
                        continue
                    path = mat_path(args.basedir, p, e, mat)
                    a_np = np.asarray(mmread(path).todense())
                    n = ns[p]
                    beta = 1 if mat in ('m3', 'm6') else 0

                    fp_dense, fp_sparse, bw, prov = benchmark(backend, a_np,
                                                              n, beta)
                    m, k = a_np.shape
                    dsize = np.dtype(backend.fpdtype).itemsize
                    working_set_gib = (m + k)*n*dsize / 1024**3

                    if args.dump_kernel_src:
                        dump_kernel_src(backend, a_np, p, e, mat,
                                        args.dump_kernel_src)

                    writer.writerow({
                        'label': args.label,
                        'host': socket.gethostname(),
                        'backend': args.backend,
                        'precision': args.precision,
                        'N': n,
                        'p': p,
                        'ele': e,
                        'mat': mat,
                        'working_set_gib': working_set_gib,
                        **prov,
                        'fp_dense': fp_dense,
                        'fp_sparse': fp_sparse,
                        'bw': bw
                    })
                    stream.flush()
    finally:
        if args.out:
            stream.close()


if __name__ == '__main__':
    main()
