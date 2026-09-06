from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LanguageInfo:
    name: str
    version: str | None = None
    files: list[str] = field(default_factory=list)


@dataclass
class FrameworkInfo:
    name: str
    version: str | None = None
    config_files: list[str] = field(default_factory=list)


@dataclass
class BuildSystemInfo:
    name: str
    config_files: list[str] = field(default_factory=list)


@dataclass
class PackageManagerInfo:
    name: str
    lock_files: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)


_LANGUAGE_PATTERNS: dict[str, dict[str, Any]] = {
    "python": {
        "config_files": {"pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile", "poetry.lock"},
        "extensions": {".py"},
    },
    "javascript": {
        "config_files": {"package.json"},
        "extensions": {".js", ".mjs", ".cjs"},
    },
    "typescript": {
        "config_files": {"package.json", "tsconfig.json"},
        "extensions": {".ts", ".tsx", ".mts", ".cts"},
    },
    "rust": {
        "config_files": {"Cargo.toml"},
        "extensions": {".rs"},
    },
    "go": {
        "config_files": {"go.mod", "go.sum"},
        "extensions": {".go"},
    },
    "java": {
        "config_files": {"pom.xml", "build.gradle", "build.gradle.kts"},
        "extensions": {".java"},
    },
    "ruby": {
        "config_files": {"Gemfile", "Gemfile.lock"},
        "extensions": {".rb"},
    },
    "csharp": {
        "config_files": {"*.csproj", "*.sln"},
        "extensions": {".cs"},
    },
    "cpp": {
        "config_files": {"CMakeLists.txt", "Makefile"},
        "extensions": {".cpp", ".hpp", ".cc", ".h", ".cxx"},
    },
    "c": {
        "config_files": {"CMakeLists.txt", "Makefile"},
        "extensions": {".c", ".h"},
    },
}

_FRAMEWORK_PATTERNS: dict[str, list[str]] = {
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "spring": ["spring-boot"],
    "react": ["react", "react-dom"],
    "nextjs": ["next"],
    "vue": ["vue"],
    "angular": ["@angular/core"],
    "express": ["express"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["torch"],
    "axum": ["axum"],
    "actix": ["actix-web"],
    "rocket": ["rocket"],
    "gin": ["gin"],
    "echo": ["echo"],
}

_PACKAGE_MANAGERS: dict[str, PackageManagerInfo] = {
    "pip": PackageManagerInfo(name="pip", lock_files=["requirements.txt"], config_files=["requirements.txt", "setup.py", "setup.cfg"]),  # noqa: E501
    "poetry": PackageManagerInfo(name="poetry", lock_files=["poetry.lock"], config_files=["pyproject.toml"]),
    "pipenv": PackageManagerInfo(name="pipenv", lock_files=["Pipfile.lock"], config_files=["Pipfile"]),
    "uv": PackageManagerInfo(name="uv", lock_files=["uv.lock"], config_files=["pyproject.toml"]),
    "npm": PackageManagerInfo(name="npm", lock_files=["package-lock.json"], config_files=["package.json"]),
    "yarn": PackageManagerInfo(name="yarn", lock_files=["yarn.lock"], config_files=["package.json"]),
    "pnpm": PackageManagerInfo(name="pnpm", lock_files=["pnpm-lock.yaml"], config_files=["package.json"]),
    "cargo": PackageManagerInfo(name="cargo", lock_files=["Cargo.lock"], config_files=["Cargo.toml"]),
    "go_mod": PackageManagerInfo(name="go modules", lock_files=["go.sum"], config_files=["go.mod"]),
    "bundler": PackageManagerInfo(name="bundler", lock_files=["Gemfile.lock"], config_files=["Gemfile"]),
    "gradle": PackageManagerInfo(name="gradle", lock_files=[], config_files=["build.gradle", "build.gradle.kts"]),
    "maven": PackageManagerInfo(name="maven", lock_files=[], config_files=["pom.xml"]),
}

_BUILD_SYSTEMS: dict[str, BuildSystemInfo] = {
    "setuptools": BuildSystemInfo(name="setuptools", config_files=["setup.py", "setup.cfg", "pyproject.toml"]),
    "poetry": BuildSystemInfo(name="poetry", config_files=["pyproject.toml"]),
    "cargo": BuildSystemInfo(name="cargo", config_files=["Cargo.toml"]),
    "go_build": BuildSystemInfo(name="go build", config_files=["go.mod"]),
    "make": BuildSystemInfo(name="make", config_files=["Makefile", "makefile"]),
    "cmake": BuildSystemInfo(name="cmake", config_files=["CMakeLists.txt"]),
    "gradle": BuildSystemInfo(name="gradle", config_files=["build.gradle", "build.gradle.kts", "settings.gradle"]),
    "maven": BuildSystemInfo(name="maven", config_files=["pom.xml"]),
    "npm": BuildSystemInfo(name="npm", config_files=["package.json"]),
}


class WorkspaceDetector:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def detect_languages(self) -> list[LanguageInfo]:
        detected: list[LanguageInfo] = []
        for lang_name, patterns in _LANGUAGE_PATTERNS.items():
            config_files = patterns["config_files"]
            extensions = patterns["extensions"]
            found_configs: list[str] = []
            for cfg in config_files:
                if "*" in cfg:
                    if list(self.root.glob(cfg)):
                        found_configs.append(cfg)
                elif (self.root / cfg).exists():
                    found_configs.append(cfg)
            if found_configs:
                version = self._detect_lang_version(lang_name, found_configs)
                detected.append(LanguageInfo(name=lang_name, version=version, files=found_configs))
            else:
                has_source = any(
                    p.suffix in extensions
                    for p in self.root.iterdir()
                    if p.is_file()
                )
                if has_source:
                    detected.append(LanguageInfo(name=lang_name))
        return detected

    def detect_frameworks(self) -> list[FrameworkInfo]:
        detected: list[FrameworkInfo] = []
        for fw_name, deps in _FRAMEWORK_PATTERNS.items():
            for dep in deps:
                if self._check_dependency(dep):
                    detected.append(FrameworkInfo(name=fw_name))
                    break
        return detected

    def detect_package_managers(self) -> list[PackageManagerInfo]:
        detected: list[PackageManagerInfo] = []
        for pm in _PACKAGE_MANAGERS.values():
            for cfg in pm.config_files:
                if (self.root / cfg).exists():
                    detected.append(pm)
                    break
            if pm not in detected:
                for lock in pm.lock_files:
                    if (self.root / lock).exists():
                        detected.append(pm)
                        break
        return detected

    def detect_build_systems(self) -> list[BuildSystemInfo]:
        detected: list[BuildSystemInfo] = []
        for bs in _BUILD_SYSTEMS.values():
            for cfg in bs.config_files:
                if (self.root / cfg).exists():
                    detected.append(bs)
                    break
        return detected

    def detect_all(self) -> dict[str, Any]:
        return {
            "languages": self.detect_languages(),
            "frameworks": self.detect_frameworks(),
            "package_managers": self.detect_package_managers(),
            "build_systems": self.detect_build_systems(),
        }

    def _detect_lang_version(self, lang: str, config_files: list[str]) -> str | None:
        if lang == "python":
            for name in ("pyproject.toml", "setup.py", "setup.cfg"):
                path = self.root / name
                if path.exists():
                    try:
                        text = path.read_text(encoding="utf-8", errors="replace")
                        import re
                        m: re.Match[str] | None = re.search(r'requires-python\s*=\s*["\']([^"\']+)["\']', text)
                        if m:
                            return m.group(1)
                        m = re.search(r'python_requires\s*=\s*["\']([^"\']+)["\']', text)
                        if m:
                            return m.group(1)
                    except OSError:
                        pass
        elif lang == "javascript" or lang == "typescript":
            path = self.root / "package.json"
            if path.exists():
                try:
                    import json
                    data: dict[str, object] = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                    dev_deps = data.get("devDependencies", {})
                    deps = data.get("dependencies", {})
                    if isinstance(dev_deps, dict) and isinstance(deps, dict):
                        for dep_name in ("typescript", "@types/node"):
                            ver = dev_deps.get(dep_name) or deps.get(dep_name)
                            if isinstance(ver, str):
                                return ver.lstrip("^~")
                    engines = data.get("engines")
                    if isinstance(engines, dict):
                        engine = engines.get(lang)
                        if isinstance(engine, str):
                            return engine
                except (OSError, json.JSONDecodeError):
                    pass
        return None

    def _check_dependency(self, dep_name: str) -> bool:
        for file_name in ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "package.json"):
            path = self.root / file_name
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8", errors="replace").lower()
                    if dep_name.lower() in text:
                        return True
                except OSError:
                    continue
        return False
