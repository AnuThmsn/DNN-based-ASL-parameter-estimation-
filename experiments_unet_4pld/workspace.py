import os
import sys
import shutil
import json
import hashlib

def get_disk_free_gb(path):
    """Returns free space in gigabytes for the drive containing path."""
    try:
        usage = shutil.disk_usage(os.path.abspath(path))
        return usage.free / (1024 ** 3)
    except Exception:
        return 0.0

class WorkspaceManager:
    """
    Centralized workspace manager ensuring all large artifacts (datasets,
    checkpoints, logs, plots, temporary tensors) reside on a safe, configurable
    drive (default: D:\\ASL_WORKSPACE) rather than exhausting the Windows C: drive.
    """
    def __init__(self, custom_path=None, min_free_gb=5.0):
        # 1. Determine workspace root
        if custom_path:
            self.root = os.path.abspath(custom_path)
        elif "ASL_WORK_DIR" in os.environ:
            self.root = os.path.abspath(os.environ["ASL_WORK_DIR"])
        else:
            # Check if D: exists and has free space, else fallback to repository directory
            if os.path.exists("D:\\"):
                self.root = os.path.abspath("D:\\ASL_WORKSPACE")
            else:
                self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "asl_workspace"))

        self.min_free_gb = min_free_gb

        # 2. Define subdirectories
        self.subdirs = {
            "data": os.path.join(self.root, "data"),
            "checkpoints": os.path.join(self.root, "checkpoints"),
            "results": os.path.join(self.root, "results"),
            "logs": os.path.join(self.root, "logs"),
            "plots": os.path.join(self.root, "plots"),
            "tmp": os.path.join(self.root, "tmp"),
            "final_model": os.path.join(self.root, "final_model_4pld")
        }

        # 3. Initialize directory structures
        self._initialize_workspace()

        # 4. Redirect temporary directories to prevent filling C:
        self._configure_environment()

    def _initialize_workspace(self):
        """Creates necessary workspace directories and verifies write permissions."""
        os.makedirs(self.root, exist_ok=True)
        for d in self.subdirs.values():
            os.makedirs(d, exist_ok=True)

        # Test writability
        test_file = os.path.join(self.subdirs["tmp"], ".write_test")
        try:
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
        except Exception as e:
            raise PermissionError(f"Workspace path {self.root} is not writable: {e}")

    def _configure_environment(self):
        """Redirects OS and library temp directories to workspace tmp directory."""
        tmp_dir = self.subdirs["tmp"]
        os.environ["TMPDIR"] = tmp_dir
        os.environ["TEMP"] = tmp_dir
        os.environ["TMP"] = tmp_dir
        os.environ["TORCH_HOME"] = os.path.join(tmp_dir, "torch_cache")
        os.environ["MPLCONFIGDIR"] = os.path.join(tmp_dir, "mpl_cache")
        os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
        os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    def check_preflight_storage(self, required_gb=2.0):
        """
        Validates available disk space before starting expensive tasks.
        Raises RuntimeError if space is below the safety threshold.
        """
        free_work = get_disk_free_gb(self.root)
        c_drive = "C:\\" if os.name == "nt" else "/"
        free_c = get_disk_free_gb(c_drive)

        print(f"\n[Storage Check] Workspace: {self.root} (Free: {free_work:.2f} GB)")
        print(f"[Storage Check] System Drive: {c_drive} (Free: {free_c:.2f} GB)")

        if free_work < self.min_free_gb:
            raise RuntimeError(
                f"INSUFFICIENT DISK SPACE: Workspace {self.root} has only {free_work:.2f} GB free "
                f"(Minimum threshold: {self.min_free_gb:.1f} GB). Training halted to prevent disk failure."
            )

        if free_work < required_gb:
            raise RuntimeError(
                f"INSUFFICIENT DISK SPACE FOR RUN: Estimated need {required_gb:.2f} GB, "
                f"but only {free_work:.2f} GB available in {self.root}."
            )

        return {
            "workspace_free_gb": free_work,
            "c_free_gb": free_c,
            "status": "PASS"
        }

    def get_path(self, subdir, *filenames):
        """Resolves full path inside the managed workspace."""
        if subdir not in self.subdirs:
            raise KeyError(f"Unknown workspace subdirectory: {subdir}. Available: {list(self.subdirs.keys())}")
        if filenames:
            return os.path.join(self.subdirs[subdir], *filenames)
        return self.subdirs[subdir]

    def get_storage_summary(self):
        """Calculates storage consumed by each workspace category."""
        summary = {}
        total = 0
        for name, path in self.subdirs.items():
            sub_size = 0
            for root, _, files in os.walk(path):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        sub_size += os.path.getsize(fp)
                    except OSError:
                        pass
            summary[name] = round(sub_size / (1024 ** 2), 2) # MB
            total += sub_size
        summary["total_mb"] = round(total / (1024 ** 2), 2)
        summary["workspace_free_gb"] = round(get_disk_free_gb(self.root), 2)
        return summary

# Global singleton instance
_DEFAULT_WORKSPACE = None

def get_workspace(custom_path=None, min_free_gb=5.0):
    global _DEFAULT_WORKSPACE
    if _DEFAULT_WORKSPACE is None or custom_path is not None:
        _DEFAULT_WORKSPACE = WorkspaceManager(custom_path=custom_path, min_free_gb=min_free_gb)
    return _DEFAULT_WORKSPACE

