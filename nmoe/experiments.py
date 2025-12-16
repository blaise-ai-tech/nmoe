"""Experiment tracking with SQLite.

Manages experiment hierarchy and run metadata for NVIZ.

Schema:
  experiments(id, name, project, description, created_at)
  runs(id, experiment_id, config_json, git_hash, git_dirty, started_at, ended_at, status)
"""
import os
import json
import time
import sqlite3
import subprocess
from typing import Optional
from dataclasses import dataclass


@dataclass
class ExperimentInfo:
  id: str
  name: str
  project: str
  description: str
  created_at: str


@dataclass
class RunInfo:
  id: str
  experiment_id: str
  config_json: str
  git_hash: str
  git_dirty: bool
  started_at: str
  ended_at: Optional[str]
  status: str


class ExperimentTracker:
  """SQLite-backed experiment and run tracking."""

  def __init__(self, cfg):
    db_path = getattr(cfg, 'experiments_db', '/data/experiments.db')
    db_dir = os.path.dirname(db_path) or "."
    os.makedirs(db_dir, exist_ok=True)
    try:
      self.conn = sqlite3.connect(db_path, timeout=30.0)
      self.conn.execute("PRAGMA busy_timeout=5000")

      # Prefer WAL for concurrent readers, but tolerate filesystems that can't do it.
      # NOTE: Some network filesystems (or mounts like NFS with locking disabled)
      # error with "locking protocol". In that case, fail loud with a remedy.
      try:
        self.conn.execute("PRAGMA journal_mode=WAL")
      except sqlite3.OperationalError:
        self.conn.execute("PRAGMA journal_mode=DELETE")

      self.cfg = cfg
      self._create_tables()
    except sqlite3.OperationalError as e:
      if "locking protocol" in str(e).lower():
        raise RuntimeError(
          "SQLite experiments DB failed with 'locking protocol'. This typically means the DB path is on a filesystem "
          "without POSIX file locks (e.g. NFS mounted with locking disabled).\n"
          f"experiments_db={db_path}\n"
          "Fix: mount /data with file locking enabled, or set `experiments_db` in your TOML to a local path "
          "(e.g. /workspace/nmoe/tmp/experiments.db)."
        ) from e
      raise

  def _create_tables(self):
    """Create schema if not exists."""
    self.conn.execute("""
      CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        project TEXT NOT NULL,
        description TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      )
    """)

    self.conn.execute("""
      CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        config_json TEXT NOT NULL,
        git_hash TEXT,
        git_dirty INTEGER DEFAULT 0,
        started_at TEXT NOT NULL DEFAULT (datetime('now')),
        ended_at TEXT,
        status TEXT NOT NULL DEFAULT 'running',
        results_json TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id)
      )
    """)

    self.conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id)")
    self.conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
    self.conn.commit()

  def create_experiment(self, name: str, project: str, description: str = "") -> str:
    """Create a new experiment. Returns experiment_id."""
    experiment_id = f"{project}_{name}_{int(time.time())}"

    self.conn.execute(
      "INSERT INTO experiments(id, name, project, description) VALUES (?, ?, ?, ?)",
      (experiment_id, name, project, description)
    )
    self.conn.commit()

    return experiment_id

  def start_run(self, *, run_id: Optional[str] = None) -> str:
    """Start a new run. Returns run_id.

    If run_id is provided, it is used verbatim (callers should ensure uniqueness).
    """
    if run_id is None:
      run_id = f"run_{int(time.time())}_{os.getpid()}"
    experiment_id = getattr(self.cfg, 'experiment_id', 'default')
    config_json = json.dumps(vars(self.cfg) if hasattr(self.cfg, '__dict__') else {}, indent=2)

    # Get git info
    git_hash, git_dirty = self._get_git_info()

    try:
      self.conn.execute(
        """
        INSERT INTO runs(id, experiment_id, config_json, git_hash, git_dirty, status)
        VALUES (?, ?, ?, ?, ?, 'running')
        """,
        (run_id, experiment_id, config_json, git_hash, int(git_dirty))
      )
      self.conn.commit()
    except sqlite3.IntegrityError as e:
      raise RuntimeError(f"run id already exists in experiments db: {run_id}") from e

    return run_id

  def log_results(self, run_id: str, results: dict):
    """Log final results for a run (final_loss, tokens_seen, tok_per_s, etc)."""
    results_json = json.dumps(results, indent=2)
    self.conn.execute(
      "UPDATE runs SET results_json = ? WHERE id = ?",
      (results_json, run_id)
    )
    self.conn.commit()

  def end_run(self, run_id: str, status: str, results: dict = None):
    """Mark run as completed/failed. Status: 'completed', 'failed', 'killed'."""
    if results:
      results_json = json.dumps(results, indent=2)
      self.conn.execute(
        "UPDATE runs SET ended_at = datetime('now'), status = ?, results_json = ? WHERE id = ?",
        (status, results_json, run_id)
      )
    else:
      self.conn.execute(
        "UPDATE runs SET ended_at = datetime('now'), status = ? WHERE id = ?",
        (status, run_id)
      )
    self.conn.commit()

  def get_experiment(self, experiment_id: str) -> Optional[ExperimentInfo]:
    """Get experiment by ID."""
    result = self.conn.execute(
      "SELECT id, name, project, description, created_at FROM experiments WHERE id = ?",
      (experiment_id,)
    ).fetchone()

    if result is None:
      return None

    return ExperimentInfo(*result)

  def get_run(self, run_id: str) -> Optional[RunInfo]:
    """Get run by ID."""
    result = self.conn.execute(
      "SELECT id, experiment_id, config_json, git_hash, git_dirty, started_at, ended_at, status FROM runs WHERE id = ?",
      (run_id,)
    ).fetchone()

    if result is None:
      return None

    return RunInfo(*result)

  def list_runs(self, experiment_id: str, status: Optional[str] = None) -> list[RunInfo]:
    """List all runs for an experiment, optionally filtered by status."""
    if status is None:
      results = self.conn.execute(
        "SELECT id, experiment_id, config_json, git_hash, git_dirty, started_at, ended_at, status FROM runs WHERE experiment_id = ? ORDER BY started_at DESC",
        (experiment_id,)
      ).fetchall()
    else:
      results = self.conn.execute(
        "SELECT id, experiment_id, config_json, git_hash, git_dirty, started_at, ended_at, status FROM runs WHERE experiment_id = ? AND status = ? ORDER BY started_at DESC",
        (experiment_id, status)
      ).fetchall()

    return [RunInfo(*row) for row in results]

  def close(self):
    """Close SQLite connection."""
    try:
      self.conn.close()
    except Exception:
      pass

  def _get_git_info(self) -> tuple[str, bool]:
    """Get current git hash and dirty status."""
    try:
      # Get git hash
      result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=5
      )
      git_hash = result.stdout.strip() if result.returncode == 0 else "unknown"

      # Check if dirty
      result = subprocess.run(
        ['git', 'status', '--porcelain'],
        capture_output=True,
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=5
      )
      git_dirty = len(result.stdout.strip()) > 0 if result.returncode == 0 else False

      return git_hash, git_dirty
    except Exception:
      return "unknown", False
