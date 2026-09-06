import importlib.util
import inspect
import sys
from pathlib import Path

from bytecli.plugins.base import Plugin


class PluginLoader:
    def __init__(self, plugin_dirs: list[Path] | None = None) -> None:
        self._plugin_dirs: list[Path] = plugin_dirs or []

    def discover(self) -> list[tuple[str, Path]]:
        found: list[tuple[str, Path]] = []
        seen: set[str] = set()
        for directory in self._plugin_dirs:
            if not directory.is_dir():
                continue
            for entry in sorted(directory.iterdir()):
                if entry.suffix == ".py" and not entry.name.startswith("_"):
                    name = entry.stem
                    if name not in seen:
                        seen.add(name)
                        found.append((name, entry))
        return found

    def load_plugin(self, path: Path) -> Plugin:
        module_name = f"_bytecli_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load plugin from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        plugin_classes: list[type[Plugin]] = []
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Plugin) and obj is not Plugin:
                plugin_classes.append(obj)

        if not plugin_classes:
            raise ValueError(f"No Plugin subclass found in {path}")
        if len(plugin_classes) > 1:
            raise ValueError(f"Multiple Plugin subclasses in {path}: {[c.__name__ for c in plugin_classes]}")

        plugin = plugin_classes[0]()
        return plugin

    def load_plugin_from_source(self, name: str, source_code: str) -> Plugin:
        module_name = f"_bytecli_plugin_{name}"
        spec = importlib.util.spec_from_loader(module_name, None)
        if spec is None:
            raise ImportError(f"Could not create spec for plugin {name}")
        module = importlib.util.module_from_spec(spec)
        code = compile(source_code, f"<plugin:{name}>", "exec")
        exec(code, module.__dict__)
        sys.modules[module_name] = module

        plugin_classes: list[type[Plugin]] = []
        for obj_name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Plugin) and obj is not Plugin:
                plugin_classes.append(obj)

        if not plugin_classes:
            raise ValueError(f"No Plugin subclass found in source for {name}")
        if len(plugin_classes) > 1:
            raise ValueError(f"Multiple Plugin subclasses in source for {name}: {[c.__name__ for c in plugin_classes]}")

        return plugin_classes[0]()
