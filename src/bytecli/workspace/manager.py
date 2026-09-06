from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bytecli.workspace.detector import (
    BuildSystemInfo,
    FrameworkInfo,
    LanguageInfo,
    PackageManagerInfo,
    WorkspaceDetector,
)
from bytecli.workspace.git import GitDetector, GitState
from bytecli.workspace.tree import FileTreeBuilder


@dataclass
class WorkspaceInfo:
    root: Path
    name: str
    languages: list[LanguageInfo] = field(default_factory=list)
    frameworks: list[FrameworkInfo] = field(default_factory=list)
    package_managers: list[PackageManagerInfo] = field(default_factory=list)
    build_systems: list[BuildSystemInfo] = field(default_factory=list)
    git: GitState = field(default_factory=GitState)
    file_count: int = 0
    dir_count: int = 0
    file_tree: dict[str, Any] = field(default_factory=dict)


class WorkspaceManager:
    def __init__(self, root: Path | None = None, max_tree_depth: int = 4) -> None:
        self.root = Path(root).resolve() if root else Path.cwd().resolve()
        self.max_tree_depth = max_tree_depth
        self._info: WorkspaceInfo | None = None

    def analyze(self) -> WorkspaceInfo:
        if self._info is not None:
            return self._info
        detector = WorkspaceDetector(self.root)
        git = GitDetector(self.root)
        tree_builder = FileTreeBuilder(self.root, max_depth=self.max_tree_depth)

        detection = detector.detect_all()
        git_state = git.detect()
        file_tree = tree_builder.build()

        file_count, dir_count = self._count_tree(file_tree)

        info = WorkspaceInfo(
            root=self.root,
            name=self.root.name,
            languages=detection["languages"],
            frameworks=detection["frameworks"],
            package_managers=detection["package_managers"],
            build_systems=detection["build_systems"],
            git=git_state,
            file_count=file_count,
            dir_count=dir_count,
            file_tree=file_tree,
        )
        self._info = info
        return info

    def get_source_files(self, extensions: set[str] | None = None) -> list[Path]:
        tree_builder = FileTreeBuilder(self.root, max_depth=self.max_tree_depth)
        return tree_builder.get_source_files(extensions)

    def _count_tree(self, node: dict[str, Any]) -> tuple[int, int]:
        files = 0
        dirs = 0
        for child in node.get("children", []):
            if child.get("type") == "file":
                files += 1
            elif child.get("type") == "directory":
                dirs += 1
                cf, cd = self._count_tree(child)
                files += cf
                dirs += cd
        return files, dirs

    def summary(self) -> str:
        if self._info is None:
            self.analyze()
        assert self._info is not None
        info = self._info
        parts: list[str] = [f"Workspace: {info.name} ({info.root})"]
        if info.languages:
            langs = ", ".join(lang.name for lang in info.languages)
            parts.append(f"Languages: {langs}")
        if info.frameworks:
            fws = ", ".join(f.name for f in info.frameworks)
            parts.append(f"Frameworks: {fws}")
        if info.build_systems:
            bs = ", ".join(b.name for b in info.build_systems)
            parts.append(f"Build: {bs}")
        if info.package_managers:
            pms = ", ".join(p.name for p in info.package_managers)
            parts.append(f"Package: {pms}")
        if info.git.is_repo:
            parts.append(f"Git: {info.git.current_branch or '(detached)'}")
            if info.git.modified_files:
                parts.append(f"Modified: {len(info.git.modified_files)}")
            if info.git.untracked_files:
                parts.append(f"Untracked: {len(info.git.untracked_files)}")
        parts.append(f"Files: {info.file_count}, Dirs: {info.dir_count}")
        return " | ".join(parts)
