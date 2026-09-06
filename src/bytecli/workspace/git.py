from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GitState:
    is_repo: bool = False
    current_branch: str | None = None
    remote_url: str | None = None
    commit_hash: str | None = None
    commit_message: str | None = None
    modified_files: list[str] = field(default_factory=list)
    untracked_files: list[str] = field(default_factory=list)
    staged_files: list[str] = field(default_factory=list)
    ahead: int = 0
    behind: int = 0


class GitDetector:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def detect(self) -> GitState:
        git_dir = self.root / ".git"
        if not git_dir.exists():
            return GitState(is_repo=False)
        try:
            import subprocess
            state = GitState(is_repo=True)
            state.current_branch = self._run("rev-parse --abbrev-ref HEAD")
            state.commit_hash = self._run("rev-parse HEAD")
            state.commit_message = self._run("log -1 --format=%s")
            state.remote_url = self._run("config --get remote.origin.url")
            status_output = self._run("status --porcelain") or ""
            for line in status_output.splitlines():
                line = line.strip()
                if not line:
                    continue
                status = line[:2]
                file_path = line[3:]
                if status == "??":
                    state.untracked_files.append(file_path)
                elif "M" in status or "A" in status or "D" in status:
                    if status[0] != " ":
                        state.staged_files.append(file_path)
                    if len(status) > 1 and status[1] != " ":
                        state.modified_files.append(file_path)
            ahead_behind = self._run("rev-list --count --left-right HEAD...@{u}") if state.current_branch else None
            if ahead_behind:
                parts = ahead_behind.strip().split()
                if len(parts) == 2:
                    state.ahead = int(parts[0])
                    state.behind = int(parts[1])
            return state
        except (subprocess.SubprocessError, FileNotFoundError):
            return GitState(is_repo=True)

    def _run(self, args: str) -> str | None:
        try:
            import subprocess
            result = subprocess.run(
                f"git {args}",
                capture_output=True,
                text=True,
                cwd=str(self.root),
                timeout=10,
                shell=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return None
