"""Record provenance and resource use for notebook benchmarks.

Each BenchmarkRecorder instance creates a session environment snapshot and a
JSONL file with one record per measured block. Install psutil in the notebook
environment before using this module.
"""

from datetime import datetime, timezone
from importlib import metadata
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
import uuid


_REPO = Path(__file__).resolve().parents[1]
_ENV_NAMES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
    "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "SLURM_JOB_ID",
    "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU",
)


def _command(args, cwd=None):
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                                check=False, timeout=10)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _threadpools():
    try:
        from threadpoolctl import threadpool_info
        return threadpool_info()
    except ImportError:
        return None


def _framework_info():
    result = {}
    tensorflow = sys.modules.get("tensorflow")
    if tensorflow is not None:
        try:
            result["tensorflow"] = {
                "visible_devices": [str(device) for device in tensorflow.config.get_visible_devices()],
                "intra_op_threads": tensorflow.config.threading.get_intra_op_parallelism_threads(),
                "inter_op_threads": tensorflow.config.threading.get_inter_op_parallelism_threads(),
                "build": tensorflow.sysconfig.get_build_info(),
            }
        except Exception as exc:
            result["tensorflow"] = {"query_error": repr(exc)}
    return result


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def sparseaeh_details(model):
    """Summarize the realized partition and line-search work after fitting."""
    sizes = [len(locations) for locations in model.kernel.ss_loc]
    diagnostics = getattr(model, "line_search_diagnostics", None)
    return {
        "fit_defaults": {
            name: parameter.default
            for name, parameter in inspect.signature(model.run_cluster).parameters.items()
            if parameter.default is not inspect.Parameter.empty
        },
        "kernel": model.kernel.kernel,
        "kernel_parameter_l": model.kernel.l,
        "mixture_components": model.K,
        "superspots": model.kernel.M,
        "block_size_min": min(sizes),
        "block_size_max": max(sizes),
        "block_size_mean": sum(sizes) / len(sizes),
        "predecessor_blocks_max": max(map(len, model.kernel.dependency)),
        "iterations": len(diagnostics) if diagnostics is not None else None,
        "line_search_evaluations": sum(
            item.get("evaluations", 0) for iteration in (diagnostics or [])
            for item in iteration
        ) if diagnostics is not None else None,
    }


class BenchmarkRecorder:
    """Create one recorder per notebook execution, on the benchmark machine."""

    def __init__(self, notebook, output_dir="benchmark_logs", sample_interval=0.1,
                 seed=None):
        try:
            import psutil
        except ImportError as exc:
            raise ImportError(
                "Benchmark recording needs psutil in this notebook kernel; "
                "install it with `python -m pip install psutil`."
            ) from exc
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive")
        self.psutil = psutil
        self.notebook = str(notebook)
        self.seed = seed
        self.sample_interval = float(sample_interval)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_id = f"{stamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.environment_path = self.output_dir / f"environment_{self.session_id}.json"
        self.runs_path = self.output_dir / f"runs_{self.session_id}.jsonl"
        self._write_environment()

    def _write_environment(self):
        packages = sorted(
            ({"name": dist.metadata.get("Name", "unknown"), "version": dist.version}
             for dist in metadata.distributions()),
            key=lambda item: item["name"].lower(),
        )
        process = self.psutil.Process()
        try:
            affinity = len(process.cpu_affinity())
        except (AttributeError, OSError, self.psutil.Error):
            affinity = None
        os_release = None
        try:
            os_release = Path("/etc/os-release").read_text(encoding="utf-8")
        except OSError:
            pass
        environment = {
            "session_id": self.session_id,
            "notebook": self.notebook,
            "global_seed": self.seed,
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "os": platform.platform(),
            "os_release": os_release,
            "python": sys.version,
            "python_executable": sys.executable,
            "cpu_model": _cpu_model(),
            "physical_cores": self.psutil.cpu_count(logical=False),
            "logical_cores": self.psutil.cpu_count(logical=True),
            "cores_in_process_affinity": affinity,
            "installed_ram_bytes": self.psutil.virtual_memory().total,
            "gpu_inventory": _command([
                "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]),
            "thread_environment": {key: os.environ.get(key) for key in _ENV_NAMES},
            "loaded_threadpools": _threadpools(),
            "framework_configuration": _framework_info(),
            "packages": packages,
            "git_commit": _command(["git", "rev-parse", "HEAD"], cwd=_REPO),
            "git_status": _command(["git", "status", "--short"], cwd=_REPO),
            "source_sha256": {
                name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                for name in ("base.py", "covariance.py", "benchmark_logging.py")
                if (Path(__file__).parent / name).exists()
            },
            "recorder_source": str(Path(__file__).resolve()),
            "memory_sampling_interval_seconds": self.sample_interval,
        }
        self.environment_path.write_text(
            json.dumps(environment, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )

    def run(self, method, dataset, scope, parameters=None, input_shape=None,
            device="auto"):
        """Measure a code block; use as `with recorder.run(...) as run:`."""
        if device not in ("auto", "cpu", "gpu"):
            raise ValueError("device must be 'auto', 'cpu', or 'gpu'")
        return _Run(self, method, dataset, scope, parameters or {}, input_shape,
                    device)


class _Run:
    def __init__(self, recorder, method, dataset, scope, parameters, input_shape,
                 device):
        self.recorder = recorder
        self.method = method
        self.dataset = dataset
        self.scope = scope
        self.parameters = parameters
        self.input_shape = input_shape
        self.device = device
        self.details = {}
        self._stop = threading.Event()
        self._peak_rss = 0
        self._baseline_rss = None
        self._samples = 0
        self._peak_gpu_mib = None
        self._gpu_process_present = False
        self._gpu_query_available = shutil.which("nvidia-smi") is not None
        self._gpu_query_succeeded = False
        self._pids = {os.getpid()}
        self._gpu_worker = None
        self._gpu_monitor_error = None
        self._monitor_error = None
        self._elapsed = None

    def finish_timing(self, seconds=None):
        """Stop the timer before collecting result metadata or use a cell's timer."""
        self._elapsed = (time.perf_counter() - self._started
                         if seconds is None else float(seconds))

    def _sample(self):
        process = self.recorder.psutil.Process(os.getpid())
        try:
            children = process.children(recursive=True)
        except self.recorder.psutil.Error as exc:
            children = []
            self._monitor_error = f"child-process scan failed: {exc}"
        processes = [process] + children
        rss = 0
        pids = set()
        for current in processes:
            try:
                rss += current.memory_info().rss
                pids.add(current.pid)
            except (self.recorder.psutil.NoSuchProcess,
                    self.recorder.psutil.AccessDenied):
                continue
        self._peak_rss = max(self._peak_rss, rss)
        if self._baseline_rss is None:
            self._baseline_rss = rss
        self._samples += 1
        self._pids = pids

    def _sample_gpu(self):
        if self.device != "cpu" and self._gpu_query_available:
            output = _command([
                "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ])
            if output is not None:
                self._gpu_query_succeeded = True
                used_mib = 0.0
                for line in output.splitlines():
                    fields = [part.strip() for part in line.split(",")]
                    try:
                        if int(fields[0]) in self._pids:
                            used_mib += float(fields[1])
                    except (ValueError, IndexError):
                        continue
                if used_mib > 0:
                    self._gpu_process_present = True
                    self._peak_gpu_mib = max(self._peak_gpu_mib or 0, used_mib)

    def _monitor_gpu(self):
        # NVIDIA queries must not delay the more frequent RAM samples.
        while not self._stop.wait(0.5):
            try:
                self._sample_gpu()
            except Exception as exc:
                self._gpu_monitor_error = repr(exc)
                break

    def _monitor(self):
        while not self._stop.wait(self.recorder.sample_interval):
            try:
                self._sample()
            except Exception as exc:
                self._monitor_error = repr(exc)
                break

    def __enter__(self):
        self._started_utc = datetime.now(timezone.utc).isoformat()
        self._threadpools_before = _threadpools()
        self._frameworks_before = _framework_info()
        self._sample()
        self._sample_gpu()
        self._worker = threading.Thread(target=self._monitor, daemon=True)
        self._worker.start()
        if self.device != "cpu" and self._gpu_query_available:
            self._gpu_worker = threading.Thread(target=self._monitor_gpu, daemon=True)
            self._gpu_worker.start()
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        elapsed = (time.perf_counter() - self._started
                   if self._elapsed is None else self._elapsed)
        self._stop.set()
        self._worker.join(timeout=12)
        if self._gpu_worker is not None:
            self._gpu_worker.join(timeout=12)
            if self._gpu_worker.is_alive():
                self._gpu_monitor_error = "GPU monitor did not stop promptly"
            else:
                try:
                    self._sample_gpu()
                except Exception as sample_exc:
                    self._gpu_monitor_error = repr(sample_exc)
        if self._worker.is_alive():
            self._monitor_error = "memory monitor did not stop promptly"
        else:
            try:
                self._sample()
            except Exception as sample_exc:
                self._monitor_error = repr(sample_exc)
        record = {
            "session_id": self.recorder.session_id,
            "started_utc": self._started_utc,
            "notebook": self.recorder.notebook,
            "global_seed": self.recorder.seed,
            "method": self.method,
            "dataset": self.dataset,
            "scope": self.scope,
            "parameters": self.parameters,
            "input_shape": self.input_shape,
            "device_declared": self.device,
            "gpu_process_present": self._gpu_process_present,
            "gpu_query_available": self._gpu_query_available,
            "gpu_query_succeeded": self._gpu_query_succeeded,
            "gpu_monitor_error": self._gpu_monitor_error,
            "peak_gpu_memory_mib": self._peak_gpu_mib,
            "wall_seconds": elapsed,
            "baseline_process_tree_rss_bytes": self._baseline_rss,
            "peak_process_tree_rss_bytes": self._peak_rss,
            "rss_samples": self._samples,
            "rss_sampling_interval_seconds": self.recorder.sample_interval,
            "rss_definition": "sum of kernel and child-process RSS; shared pages can be counted more than once",
            "threadpools_before": self._threadpools_before,
            "threadpools_after": _threadpools(),
            "frameworks_before": self._frameworks_before,
            "frameworks_after": _framework_info(),
            "details": self.details,
            "status": "error" if exc_type is not None else "ok",
            "error": repr(exc) if exc_type is not None else None,
            "monitor_error": self._monitor_error,
        }
        with self.recorder.runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default) + "\n")
        print(f"Benchmark recorded: {self.method}, {elapsed:.3f} s -> "
              f"{self.recorder.runs_path}")
        return False
