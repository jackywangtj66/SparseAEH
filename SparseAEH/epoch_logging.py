"""Per-iteration result snapshots for the CRC comparison notebook.

Some comparison packages expose no epoch callback. Instrument a private copy of
their installed Python loop, leaving updates, early stopping and retries intact.
No installed module or class is patched globally.
"""
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import time
from types import MethodType
import uuid

import numpy as np


class EpochSaveError(RuntimeError):
    """Do not let a training retry handler swallow an output failure."""


def _scalar(value):
    if hasattr(value, 'numpy'):
        value = value.numpy()
    value = float(value)
    return value if np.isfinite(value) else None


class EpochRecorder:
    def __init__(self, output_dir, method):
        self.started = time.perf_counter()
        self.method = method
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.output_dir = Path(output_dir) / method / (stamp + '_' + uuid.uuid4().hex[:8])
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.log_path = self.output_dir / 'epochs.jsonl'
        self.log_path.touch()
        self.events = 0
        self.completed = 0
        self.compute_seconds = 0.0
        self.output_seconds = 0.0

    def begin(self, state):
        epoch = self._epoch(state) + (self.method == 'rsf_gaussian')
        return epoch, time.perf_counter()

    def end(self, token, state, error):
        epoch, started = token
        compute = time.perf_counter() - started
        self.compute_seconds += compute
        self.events += 1
        output_started = time.perf_counter()
        folder = self.output_dir / ('event_{:06d}_epoch_{:06d}'.format(self.events, epoch))
        record = dict(event=self.events, epoch=epoch, method=self.method,
                      recorded_utc=datetime.now(timezone.utc).isoformat(),
                      compute_seconds=compute,
                      cumulative_epoch_compute_seconds=self.compute_seconds,
                      status='failed' if error is not None else 'ok',
                      error=repr(error) if error is not None else None,
                      result_dir=None)
        save_error = None
        try:
            if error is None:
                folder.mkdir()
                record.update(self._save(state, folder))
                record['result_dir'] = str(folder)
                self.completed += 1
        except Exception as exc:
            save_error = exc
            record.update(status='checkpoint_error', error=repr(exc))
        record['checkpoint_seconds'] = time.perf_counter() - output_started
        try:
            with self.log_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, allow_nan=False) + '\n')
        finally:
            self.output_seconds += time.perf_counter() - output_started
        if save_error is not None:
            raise EpochSaveError('Could not save epoch results: ' + str(folder)) from save_error

    def _epoch(self, state):
        if self.method == 'spatialde_aeh':
            return int(state['i']) + 1
        if self.method == 'rsf_gaussian':
            return int(state['self'].epoch.numpy())
        if self.method == 'mefisto':
            return int(state['i'])
        if self.method == 'full_covariance_gem':
            return int(state['_']) + 1
        raise ValueError('Unknown epoch method: ' + self.method)

    def _save(self, state, folder):
        if self.method == 'spatialde_aeh':
            np.savez_compressed(folder / 'result.npz', mean=state['m'], membership=state['r'],
                                labels=np.argmax(state['r'], axis=1), pi=state['pi'],
                                noise=state['s2e'], elbo=state['elbo_1'])
            return {'elbo': _scalar(state['elbo_1'])}
        if self.method == 'rsf_gaussian':
            trainer = state['self']
            checkpoint = trainer.ckpt.write(str(folder / 'model'))
            epoch = int(trainer.epoch.numpy())
            np.savez_compressed(folder / 'result.npz',
                                loadings=trainer.model.get_loadings(),
                                train_loss=state['trl'], epoch=epoch, converged=trainer.converged,
                                train_loss_history=trainer.loss['train'][:epoch + 1],
                                validation_loss_history=trainer.loss['val'][:epoch + 1])
            return {'train_loss': _scalar(state['trl']), 'checkpoint': str(checkpoint),
                    'learning_rate': _scalar(trainer.optimizer.learning_rate)}
        if self.method == 'mefisto':
            model = state['self']
            arrays = dict(factors=model.nodes['Z'].getExpectation(),
                          loadings=model.nodes['W'].getExpectation()[0])
            if 'Sigma' in model.nodes:
                params = model.nodes['Sigma'].getParameters()
                for name in ('l', 'scale', 'Kg'):
                    arrays['gp_' + name] = params[name]
            np.savez_compressed(folder / 'result.npz', **arrays)
            # MOFA trims the current ELBO row on convergence; do not label a stale
            # previous value as the current objective or recompute it with side effects.
            elbo = state['elbo']
            i = int(state['i'])
            objective = _scalar(elbo.loc[i, 'total']) if i in elbo.index else None
            return {'elbo': objective, 'factors': int(arrays['factors'].shape[1])}
        model = state['self']
        # The final responsibility refresh normally happens after the GEM loop.
        # Compute the corresponding snapshot without changing the model's state.
        joint = state['ll'] + state['log_pi']
        weights = np.exp(joint - joint.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        np.savez_compressed(folder / 'result.npz', mean=model.mean, omega=weights,
                            labels=weights.argmax(axis=1), pi=model.pi,
                            sigma_sq=model.sigma_sq, delta=model.delta,
                            log_likelihood_history=model.log_likelihood_history)
        return {'log_likelihood': _scalar(model.log_likelihood_history[-1])}

    def finish(self, run):
        """Keep the benchmark's compute time separate from added snapshot I/O."""
        wall = time.perf_counter() - self.started
        summary = dict(method=self.method, epochs_completed=self.completed,
                       epoch_events=self.events, epoch_log=str(self.log_path),
                       total_wall_seconds_including_epoch_output=wall,
                       epoch_output_seconds=self.output_seconds,
                       total_seconds_excluding_epoch_output=max(0.0, wall - self.output_seconds),
                       epoch_compute_seconds=self.compute_seconds,
                       timing_scope='total includes setup; epoch timing is one original training-loop pass',
                       checkpoint_scope=('model and optimizer checkpoint; final factor maps saved separately'
                                         if self.method == 'rsf_gaussian'
                                         else 'result arrays, not a resumable optimizer checkpoint'))
        run.finish_timing(summary['total_seconds_excluding_epoch_output'])
        run.details.update(summary)
        (self.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


_LOOPS = {
    'spatialde_aeh': 'for i in range(maxiter): pass',
    'rsf_gaussian': 'while (not self.converged) and (self.epoch < num_epochs): pass',
    'mefisto': 'for i in range(1, self.options["maxiter"]): pass',
    'full_covariance_gem': 'for _ in range(iter): pass',
}


def instrument_epochs(function, recorder):
    """Clone one supported loop, wrapping each complete iteration (including break).

    Fail before training if an installed version has a different loop contract.
    The small wrapper executes once per iteration, not once per traced Python line.
    """
    original = function.__func__ if inspect.ismethod(function) else function
    if original.__closure__:
        raise ValueError('Epoch instrumentation requires a module-level function or method.')
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    definition = tree.body[0]
    expected = ast.parse(_LOOPS[recorder.method]).body[0]

    def header(node):
        if isinstance(node, ast.For):
            return ast.dump(node.target), ast.dump(node.iter)
        if isinstance(node, ast.While):
            return ast.dump(node.test)

    loops = [node for node in ast.walk(definition)
             if type(node) is type(expected) and header(node) == header(expected)]
    if len(loops) != 1:
        raise RuntimeError('Unsupported training loop for ' + recorder.method
                           + '; no training was started. Check the installed package version.')
    loop = loops[0]
    # finally executes on early convergence too; failed iterations are logged but
    # never saved as successful results. No frame/array references are retained.
    wrapper = ast.parse('''
_crc_epoch_token = _crc_epoch_recorder.begin(locals())
_crc_epoch_error = None
try:
    pass
except BaseException as _crc_epoch_exception:
    _crc_epoch_error = _crc_epoch_exception
    raise
finally:
    _crc_epoch_recorder.end(_crc_epoch_token, locals(), _crc_epoch_error)
''').body
    wrapper[2].body = loop.body
    loop.body = wrapper
    definition.decorator_list = []
    namespace = dict(original.__globals__, _crc_epoch_recorder=recorder)
    # Avoid evaluating default expressions a second time (e.g. NSF time()).
    definition.args.defaults = [ast.Constant(None) for _ in definition.args.defaults]
    definition.args.kw_defaults = [ast.Constant(None) if value is not None else None
                                   for value in definition.args.kw_defaults]
    ast.fix_missing_locations(tree)
    exec(compile(tree, inspect.getsourcefile(original) or '<epoch loop>', 'exec'), namespace)
    wrapped = namespace[original.__name__]
    wrapped.__defaults__, wrapped.__kwdefaults__ = original.__defaults__, original.__kwdefaults__
    metadata = dict(module=original.__module__, function=original.__qualname__,
                    source_file=inspect.getsourcefile(original),
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    instrumentation='original loop wrapped with per-iteration timing and snapshots')
    (recorder.output_dir / 'implementation.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return MethodType(wrapped, function.__self__) if inspect.ismethod(function) else wrapped


@contextmanager
def epoch_method(instance, name, recorder):
    """Replace only this instance's bound method and restore it even on failure."""
    present = name in instance.__dict__
    previous = instance.__dict__.get(name)
    wrapped = instrument_epochs(getattr(instance, name), recorder)
    setattr(instance, name, wrapped)
    try:
        yield
    finally:
        if present:
            setattr(instance, name, previous)
        else:
            delattr(instance, name)
