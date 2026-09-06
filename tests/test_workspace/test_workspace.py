import tempfile
from pathlib import Path

import pytest

from bytecli.workspace.detector import (
    LanguageInfo,
    PackageManagerInfo,
    WorkspaceDetector,
)
from bytecli.workspace.git import GitDetector, GitState
from bytecli.workspace.manager import WorkspaceManager
from bytecli.workspace.tree import FileTreeBuilder


class TestDetector:
    @pytest.fixture
    def python_project(self) -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text("[project]\nname = 'test'\nrequires-python = '>=3.12'\n")
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("print('hello')\n")
            (root / "requirements.txt").write_text("fastapi\npydantic\n")
            yield root

    @pytest.fixture
    def js_project(self) -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "package.json").write_text('{"name": "test", "dependencies": {"react": "^18.0.0"}}\n')
            (root / "index.js").write_text("console.log('hello')\n")
            yield root

    @pytest.fixture
    def empty_dir(self) -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_detect_python(self, python_project: Path) -> None:
        detector = WorkspaceDetector(python_project)
        langs = detector.detect_languages()
        assert any(lang.name == "python" for lang in langs)

    def test_detect_python_version(self, python_project: Path) -> None:
        detector = WorkspaceDetector(python_project)
        langs = detector.detect_languages()
        py_lang = next(lang for lang in langs if lang.name == "python")
        assert py_lang.version is not None
        assert "3.12" in py_lang.version

    def test_detect_javascript(self, js_project: Path) -> None:
        detector = WorkspaceDetector(js_project)
        langs = detector.detect_languages()
        assert any(lang.name in ("javascript", "typescript") for lang in langs)

    def test_detect_frameworks_react(self, js_project: Path) -> None:
        detector = WorkspaceDetector(js_project)
        frameworks = detector.detect_frameworks()
        assert any(f.name == "react" for f in frameworks)

    def test_detect_package_managers(self, python_project: Path) -> None:
        detector = WorkspaceDetector(python_project)
        pms = detector.detect_package_managers()
        names = [pm.name for pm in pms]
        assert "pip" in names

    def test_detect_npm(self, js_project: Path) -> None:
        detector = WorkspaceDetector(js_project)
        pms = detector.detect_package_managers()
        names = [pm.name for pm in pms]
        assert "npm" in names

    def test_detect_build_systems(self, python_project: Path) -> None:
        detector = WorkspaceDetector(python_project)
        systems = detector.detect_build_systems()
        assert any(s.name == "setuptools" for s in systems)

    def test_detect_empty(self, empty_dir: Path) -> None:
        detector = WorkspaceDetector(empty_dir)
        result = detector.detect_all()
        assert result["languages"] == []
        assert result["frameworks"] == []
        assert result["package_managers"] == []
        assert result["build_systems"] == []

    def test_detect_all(self, python_project: Path) -> None:
        detector = WorkspaceDetector(python_project)
        result = detector.detect_all()
        assert len(result["languages"]) >= 1
        assert len(result["package_managers"]) >= 1

    def test_language_info(self) -> None:
        li = LanguageInfo(name="python", version=">=3.12", files=["pyproject.toml"])
        assert li.name == "python"
        assert li.version == ">=3.12"

    def test_package_manager_info(self) -> None:
        pm = PackageManagerInfo(name="cargo", lock_files=["Cargo.lock"], config_files=["Cargo.toml"])
        assert pm.name == "cargo"
        assert "Cargo.toml" in pm.config_files


class TestFileTreeBuilder:
    @pytest.fixture
    def project(self) -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("x = 1\n")
            (root / "src" / "utils.py").write_text("y = 2\n")
            (root / "README.md").write_text("# Test\n")
            (root / ".gitignore").write_text("*.md\n")
            yield root

    def test_build_tree(self, project: Path) -> None:
        builder = FileTreeBuilder(project, max_depth=3)
        tree = builder.build()
        assert tree["type"] == "directory"
        assert tree["name"] == project.name
        children = {c["name"]: c for c in tree.get("children", [])}
        assert "src" in children
        assert children["src"]["type"] == "directory"

    def test_gitignore_filtering(self, project: Path) -> None:
        builder = FileTreeBuilder(project, max_depth=3)
        tree = builder.build()
        children_names = [c["name"] for c in tree.get("children", [])]
        assert "README.md" not in children_names

    def test_ignored_dir(self, project: Path) -> None:
        (project / "__pycache__").mkdir()
        builder = FileTreeBuilder(project, max_depth=3)
        tree = builder.build()
        children_names = [c["name"] for c in tree.get("children", [])]
        assert "__pycache__" not in children_names

    def test_max_depth(self, project: Path) -> None:
        (project / "a" / "b" / "c" / "d.py").parent.mkdir(parents=True)
        builder = FileTreeBuilder(project, max_depth=2)
        tree = builder.build()
        a_node = next(c for c in tree.get("children", []) if c["name"] == "a")
        b_node = next(c for c in a_node.get("children", []) if c["name"] == "b")
        assert b_node.get("children") == []

    def test_get_source_files(self, project: Path) -> None:
        builder = FileTreeBuilder(project, max_depth=3)
        files = builder.get_source_files(extensions={".py"})
        assert len(files) == 2
        assert all(f.suffix == ".py" for f in files)


class TestGitDetector:
    @pytest.fixture
    def git_repo(self) -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            import subprocess
            subprocess.run("git init", cwd=str(root), shell=True, capture_output=True)
            subprocess.run("git config user.email test@test.com", cwd=str(root), shell=True, capture_output=True)
            subprocess.run("git config user.name Test", cwd=str(root), shell=True, capture_output=True)
            (root / "README.md").write_text("# Test\n")
            subprocess.run("git add .", cwd=str(root), shell=True, capture_output=True)
            subprocess.run("git commit -m 'initial'", cwd=str(root), shell=True, capture_output=True)
            (root / "untracked.txt").write_text("new\n")
            yield root

    @pytest.fixture
    def non_repo(self) -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_detect_repo(self, git_repo: Path) -> None:
        detector = GitDetector(git_repo)
        state = detector.detect()
        assert state.is_repo is True
        assert state.current_branch is not None
        assert state.commit_hash is not None

    def test_detect_non_repo(self, non_repo: Path) -> None:
        detector = GitDetector(non_repo)
        state = detector.detect()
        assert state.is_repo is False

    def test_detect_untracked(self, git_repo: Path) -> None:
        detector = GitDetector(git_repo)
        state = detector.detect()
        assert "untracked.txt" in state.untracked_files

    def test_git_state_defaults(self) -> None:
        state = GitState()
        assert state.is_repo is False
        assert state.current_branch is None
        assert state.modified_files == []


class TestWorkspaceManager:
    @pytest.fixture
    def project(self) -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text("[project]\nname = 'test'\n")
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("x = 1\n")
            yield root

    def test_analyze(self, project: Path) -> None:
        mgr = WorkspaceManager(root=project)
        info = mgr.analyze()
        assert info.name == project.name
        assert len(info.languages) >= 1
        assert info.file_count >= 1

    def test_summary(self, project: Path) -> None:
        mgr = WorkspaceManager(root=project)
        summary = mgr.summary()
        assert project.name in summary
        assert "python" in summary.lower()

    def test_get_source_files(self, project: Path) -> None:
        mgr = WorkspaceManager(root=project)
        files = mgr.get_source_files(extensions={".py"})
        assert len(files) == 1
        assert files[0].name == "main.py"

    def test_analyze_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = WorkspaceManager(root=Path(tmpdir))
            info = mgr.analyze()
            assert info.languages == []

    def test_analyze_twice_returns_same(self, project: Path) -> None:
        mgr = WorkspaceManager(root=project)
        info1 = mgr.analyze()
        info2 = mgr.analyze()
        assert info1 is info2
