"""Run one CRC gene-comparison method without a Jupyter kernel.

The named CRC cells in demo.ipynb are the workflow source, not an executed
notebook. Only the selected method's cells are compiled as ordinary Python.
"""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import uuid


METHODS = {'sparseaeh': 'sparse', 'spatialde': 'aeh', 'rsf': 'rsf',
           'mefisto': 'mefisto', 'full-covariance': 'full'}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--output-dir', type=Path, default=Path('/nobackup2/users/jacky/crc_results'))
    parser.add_argument('--rctd-csv', type=Path, default=Path('/nobackup2/users/jacky/RCTD_weights_all_celltypes.csv'))
    parser.add_argument('--no-reference', action='store_true', help='Skip RCTD scoring.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--notebook', type=Path, default=Path(__file__).with_name('demo.ipynb'))
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    return args


def atomic_json(path, value):
    # This bootstrap runs before importing numerical libraries.
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write('\n')
        handle.flush()
    temporary.replace(path)


@contextmanager
def console_log(path):
    """Capture Python and native-library stdout/stderr, flushing Python writes."""
    sys.stdout.flush()
    sys.stderr.flush()
    descriptors = [os.dup(1), os.dup(2)]
    with path.open('a', buffering=1, encoding='utf-8') as handle:
        try:
            os.dup2(handle.fileno(), 1)
            os.dup2(handle.fileno(), 2)
            for stream in (sys.stdout, sys.stderr):
                if hasattr(stream, 'reconfigure'):
                    stream.reconfigure(line_buffering=True, write_through=True)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for target, saved in zip((1, 2), descriptors):
                os.dup2(saved, target)
                os.close(saved)


def load_cells(notebook):
    document = json.loads(notebook.read_text(encoding='utf-8'))
    return {cell['id']: ''.join(cell['source']) for cell in document['cells']
            if cell['cell_type'] == 'code' and cell.get('id', '').startswith('crc-')}


def compile_cell(source, cell_id, overrides=None, initialize_only=False, need_anndata=True):
    tree = ast.parse(source)
    if cell_id == 'crc-config':
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module == 'IPython.display')]
    if overrides:
        found = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in overrides:
                    node.value = ast.Subscript(value=ast.Name(id='_crc_script_overrides', ctx=ast.Load()),
                                               slice=ast.Constant(target.id), ctx=ast.Load())
                    found.add(target.id)
        if found != set(overrides):
            raise ValueError('CRC configuration changed; missing assignments: ' + str(set(overrides) - found))
    if initialize_only:
        # Retain the shared initialization, but never fit SparseAEH as a hidden
        # prerequisite of the independently requested full-covariance method.
        boundaries = [i for i, node in enumerate(tree.body) if isinstance(node, ast.With)
                      and any(isinstance(item.context_expr, ast.Call)
                              and any(k.arg == 'method' and isinstance(k.value, ast.Constant)
                                      and k.value.value == 'SparseAEH' for k in item.context_expr.keywords)
                              for item in node.items)]
        if len(boundaries) != 1:
            raise ValueError('Cannot identify the shared GEM initialization in demo.ipynb.')
        tree.body = tree.body[:boundaries[0]]
    if not need_anndata:
        def target_root(target):
            while isinstance(target, (ast.Attribute, ast.Subscript)):
                target = target.value
            return target.id if isinstance(target, ast.Name) else None
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.Assign) and any(target_root(t) == 'crc_gene_adata' for t in node.targets))]
    ast.fix_missing_locations(tree)
    return compile(tree, 'demo.ipynb:' + cell_id, 'exec')


def load_namespace(method, seed):
    import random
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from SparseAEH import MixedGaussian, plot_clusters
    from SparseAEH.benchmark_logging import sparseaeh_details

    namespace = dict(np=np, pd=pd, plt=plt, SEED=seed, MixedGaussian=MixedGaussian,
                     plot_clusters=plot_clusters, sparseaeh_details=sparseaeh_details)
    if method in ('rsf', 'mefisto'):
        import anndata as ad
        namespace['ad'] = ad
    if method == 'spatialde':
        import SpatialDE
        namespace['SpatialDE'] = SpatialDE
    if method == 'rsf':
        import tensorflow as tf
        from tensorflow_probability import math as tm
        from nsf_paper.models import sf
        from nsf_paper.utils import misc, preprocess, training, postprocess
        tf.config.set_visible_devices([], 'GPU')
        namespace.update(tf=tf, tfk=tm.psd_kernels, sf=sf, misc=misc,
                         preprocess=preprocess, training=training, postprocess=postprocess)
    if method == 'mefisto':
        from mofapy2.run.entry_point import entry_point
        namespace['entry_point'] = entry_point

    def reset_seed():
        random.seed(seed)
        np.random.seed(seed)
        if method == 'rsf':
            tf.random.set_seed(seed)
            sf.rng = np.random.default_rng(seed)

    namespace['reset_seed'] = reset_seed
    reset_seed()
    plt.show = lambda: None
    return namespace


def save_full_result(ns, output):
    """Standalone full GEM has no sparse fit with which to compute agreement."""
    np, pd = ns['np'], ns['pd']
    model = ns['gaussian_crc_full']
    data = ns['crc_gene_standardized']
    pd.DataFrame({'gene': data.columns, 'cluster': model.labels}).to_csv(
        output / 'full_covariance_gem_genes.csv', index=False)
    pd.DataFrame(model.mean, index=data.index).to_csv(output / 'full_covariance_gem_means.csv')
    np.savez_compressed(output / 'full_covariance_gem_fit.npz', mean=model.mean,
                        labels=model.labels, omega=model.omega, pi=model.pi,
                        sigma_sq=model.sigma_sq, delta=model.delta,
                        log_likelihood_history=model.log_likelihood_history)
    np.savez_compressed(output / 'gem_initialization.npz', sampled_gene_indices=ns['crc_gem_sample'],
                        **ns['crc_gem_initial'])
    ns['plot_clusters'](model, s=2, save=str(output / 'full_covariance_gem.png'))


def execute_workflow(args, output, status, update):
    from SparseAEH.benchmark_logging import BenchmarkRecorder
    from threadpoolctl import threadpool_limits

    cells = load_cells(args.notebook)
    prefix = METHODS[args.method]
    selected = ['crc-config', 'crc-input', 'crc-preprocess', 'crc-shared',
                'crc-reference', 'crc-metric-functions']
    if args.method == 'full-covariance':
        selected.append('crc-sparse-fit')
    selected += ['crc-' + prefix + '-fit']
    if args.method != 'full-covariance':
        selected.append('crc-' + prefix + '-output')
    selected.append('crc-' + prefix + '-score')
    atomic_json(output / 'workflow_sources.json', {key: cells[key] for key in selected})
    recorder = BenchmarkRecorder('run_crc.py:' + args.method, output_dir=output / 'logs',
                                 seed=args.seed, persist_peak=True)
    status.update(environment_file=str(recorder.environment_path), run_log=str(recorder.runs_path))
    with recorder.run(args.method + ' whole script', 'CRC_genes',
                      'imports, loading, preprocessing, fitting, epoch output, final output and scoring',
                      parameters={'selected_method': args.method, 'seed': args.seed,
                                  'threads': args.threads}, device='cpu') as workflow:
        status['whole_run_peak_file'] = str(workflow.peak_path)
        update('imports')
        ns = load_namespace(args.method, args.seed)
        ns['display'] = print
        recorder._write_environment()
        overrides = {
            'CRC_RESULTS_ROOT': args.output_dir, 'crc_bench': recorder, 'crc_gene_output': output,
            'CRC_RCTD_CSV': None if args.no_reference else args.rctd_csv,
            'CRC_RUN_SPATIALDE': args.method == 'spatialde', 'CRC_RUN_RSF': args.method == 'rsf',
            'CRC_RUN_MEFISTO': args.method == 'mefisto',
            'CRC_RUN_FULL_COVARIANCE': args.method == 'full-covariance',
        }
        ns['_crc_script_overrides'] = overrides

        def execute(cell_id, initialize_only=False):
            update(cell_id)
            print('Stage:', cell_id, flush=True)
            exec(compile_cell(cells[cell_id], cell_id,
                              overrides=overrides if cell_id == 'crc-config' else None,
                              initialize_only=initialize_only,
                              need_anndata=not (cell_id == 'crc-shared' and args.method not in ('rsf', 'mefisto'))), ns)
            if cell_id == 'crc-config':
                ns['display'] = print
                status['input_shape'] = list(ns['adata_crc_genes'].X.shape)
            if cell_id == 'crc-preprocess':
                status['retained_shape'] = list(ns['crc_gene_input'].shape)
            update(cell_id + ':completed')

        with threadpool_limits(limits=args.threads):
            for cell_id in selected[:6]:
                execute(cell_id)
            # These raw inputs have already been summarized in the saved tables.
            for name in ('adata_crc_genes', 'adata_impt_all', 'crc_gene_expr'):
                ns.pop(name, None)
            gc.collect()
            if args.method == 'full-covariance':
                execute('crc-sparse-fit', initialize_only=True)
            execute('crc-' + prefix + '-fit')
            staging = output / 'final.partial'
            staging.mkdir()
            ns['crc_gene_output'] = staging
            update('saving_final_results')
            if args.method == 'full-covariance':
                save_full_result(ns, staging)
            else:
                execute('crc-' + prefix + '-output')
            atomic_json(staging / 'COMPLETE.json', {'method': args.method,
                        'files': sorted(p.name for p in staging.iterdir())})
            staging.rename(output / 'final')
            ns['crc_gene_output'] = output
            status['final_results'] = str(output / 'final')
            execute('crc-' + prefix + '-score')
            workflow.details.update(selected_method=args.method,
                                    input_shape=status['input_shape'], retained_shape=status['retained_shape'])


def run(args, redirect_console=True):
    args.output_dir = args.output_dir.expanduser().resolve()
    args.notebook = args.notebook.expanduser().resolve()
    args.rctd_csv = args.rctd_csv.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output_dir / ('crc_' + args.method + '_' + stamp + '_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', pid=os.getpid(), method=args.method, arguments=vars(args).copy(),
                  started_utc=datetime.now(timezone.utc).isoformat(), stage='startup',
                  python_executable=sys.executable, command=sys.argv,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  note='If killed, status may remain running. This is not proof of OOM. Ignore .partial directories.')

    def update(stage):
        status.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(output / 'run.json', status)

    update('startup')
    print('CRC results:', output, flush=True)
    print('Console log:', output / 'console.log', flush=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['MPLBACKEND'] = 'Agg'
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
        os.environ[key] = str(args.threads)

    def work():
        try:
            status['notebook_sha256'] = hashlib.sha256(args.notebook.read_bytes()).hexdigest()
            execute_workflow(args, output, status, update)
        except BaseException as exc:
            status.update(status='error', error=repr(exc))
            update(status['stage'])
            traceback.print_exc()
            raise
        else:
            status['status'] = 'ok'
            update('completed')
            atomic_json(output / 'COMPLETE.json', {'method': args.method, 'status': 'ok'})
            print('Completed:', output, flush=True)

    if redirect_console:
        with console_log(output / 'console.log'):
            work()
    else:
        work()
    return output


if __name__ == '__main__':
    run(parse_args())
