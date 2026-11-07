"""Lightweight, persistent NSF stage and epoch diagnostics."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import uuid


class NSFDiagnostics:
    def __init__(self, output):
        self.path = Path(output) / 'nsf_diagnostics.jsonl'
        self.run_id = uuid.uuid4().hex[:8]
        self.started = time.perf_counter()
        self.epoch_started = None
        self.attempt = 0
        self.write('diagnostics_start')
        print('NSF diagnostic log:', self.path, flush=True)

    def write(self, event, **values):
        record = dict(run_id=self.run_id, event=event,
                      utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.perf_counter() - self.started, **values)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        self.write('stage_start', stage=name)
        print('NSF: ' + name + ' started', flush=True)
        try:
            yield
        except BaseException as exc:
            self.write('stage_failed', stage=name, error=repr(exc),
                       seconds=time.perf_counter() - started)
            print('NSF: ' + name + ' interrupted/failed: ' + repr(exc), flush=True)
            raise
        else:
            seconds = time.perf_counter() - started
            self.write('stage_end', stage=name, seconds=seconds)
            print('NSF: {} finished in {:.2f} s'.format(name, seconds), flush=True)

    def epoch(self, event, epoch, loss):
        if event == 'start':
            self.attempt += 1
            self.epoch_started = time.perf_counter()
            self.write('epoch_start', epoch=epoch, attempt=self.attempt)
            print('NSF epoch {} started (attempt {})'.format(epoch, self.attempt), flush=True)
            return
        finite = loss is not None and math.isfinite(loss)
        seconds = time.perf_counter() - self.epoch_started
        self.write('epoch_' + event, epoch=epoch, attempt=self.attempt,
                   loss=loss if finite else None, loss_finite=finite, seconds=seconds)
        if event == 'end':
            print('NSF epoch {}: loss {:.6g}, {:.2f} s'.format(epoch, loss, seconds), flush=True)
