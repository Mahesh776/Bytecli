from __future__ import annotations

from pathlib import Path
from typing import Any

from bytecli.workspace.detector import WorkspaceDetector


class FileTreeBuilder:
    def __init__(self, root: Path, max_depth: int = 4, ignored_dirs: set[str] | None = None) -> None:
        self.root = root.resolve()
        self.max_depth = max_depth
        self.ignored_dirs = ignored_dirs or {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            ".tox", ".egg-info", "dist", "build", ".idea", ".vscode",
            ".mypy_cache", ".pytest_cache", ".ruff_cache", ".hypothesis",
            ".DS_Store", "target", "bin", "obj", ".next", ".nuxt",
        }
        self._gitignore_specs: list[str] = []
        self._load_gitignore()

    def _load_gitignore(self) -> None:
        gitignore_path = self.root / ".gitignore"
        if gitignore_path.exists():
            try:
                self._gitignore_specs = gitignore_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError:
                pass

    def _is_ignored(self, name: str) -> bool:
        if name in self.ignored_dirs:
            return True
        import fnmatch
        for spec in self._gitignore_specs:
            spec = spec.strip()
            if not spec or spec.startswith("#"):
                continue
            if spec.startswith("/"):
                spec = spec[1:]
            if spec.rstrip("/") == name or fnmatch.fnmatch(name, spec):
                return True
        return False

    def build(self) -> dict[str, Any]:
        return self._build_node(self.root, depth=0)

    def _build_node(self, path: Path, depth: int) -> dict[str, Any]:
        node: dict[str, Any] = {
            "name": path.name,
            "path": str(path),
            "type": "directory",
            "children": [],
        }
        if depth >= self.max_depth:
            return node
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return node
        for entry in entries:
            if self._is_ignored(entry.name):
                continue
            if entry.is_dir():
                child = self._build_node(entry, depth + 1)
                node["children"].append(child)
            elif entry.is_file():
                node["children"].append(self._build_file_node(entry))
        return node

    def _build_file_node(self, path: Path) -> dict[str, Any]:
        try:
            stat = path.stat()
        except OSError:
            stat = None
        return {
            "name": path.name,
            "path": str(path),
            "type": "file",
            "size": stat.st_size if stat else 0,
        }

    def get_source_files(self, extensions: set[str] | None = None) -> list[Path]:
        if extensions is None:
            detector = WorkspaceDetector(self.root)
            langs = detector.detect_languages()
            extensions = set()
            for lang in langs:
                lang_patterns = {
                    "python": {".py"},
                    "javascript": {".js", ".mjs", ".cjs"},
                    "typescript": {".ts", ".tsx", ".mts", ".cts"},
                    "rust": {".rs"},
                    "go": {".go"},
                    "java": {".java"},
                    "ruby": {".rb"},
                    "csharp": {".cs"},
                    "cpp": {".cpp", ".hpp", ".cc", ".h", ".cxx"},
                    "c": {".c", ".h"},
                }
                if lang.name in lang_patterns:
                    extensions.update(lang_patterns[lang.name])

        files: list[Path] = []
        for ext in extensions:
            for p in self.root.rglob(f"*{ext}"):
                rel = p.relative_to(self.root)
                if not any(part.startswith(".") or part in self.ignored_dirs for part in rel.parts[:-1]):
                    files.append(p)
        return sorted(files)
