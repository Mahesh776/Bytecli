import re
from pathlib import Path

from bytecli.skills.base import SkillDef


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text

    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx == -1:
        return {}, text

    frontmatter_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1:]).strip()

    meta: dict[str, object] = {}
    for line in frontmatter_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"(\w+)\s*:\s*(.*)", line)
        if match:
            key = match.group(1)
            value = match.group(2).strip()
            if value.startswith("[") and value.endswith("]"):
                items = [v.strip().strip('"').strip("'") for v in value[1:-1].split(",")]
                meta[key] = items
            elif value.lower() == "true":
                meta[key] = True
            elif value.lower() == "false":
                meta[key] = False
            else:
                try:
                    meta[key] = int(value)
                except ValueError:
                    meta[key] = value

    return meta, body


def load_skill(path: Path) -> SkillDef:
    text = path.read_text(encoding="utf-8")
    meta, instructions = _parse_frontmatter(text)

    patterns_raw = meta.get("patterns", [])
    if isinstance(patterns_raw, list):
        patterns = [str(p) for p in patterns_raw]
    else:
        patterns = []

    name_raw = meta.get("name", path.stem)
    desc_raw = meta.get("description", "")
    priority_raw = meta.get("priority", 50)
    enabled_raw = meta.get("enabled", True)
    version_raw = meta.get("version", "1.0.0")

    return SkillDef(
        name=str(name_raw),
        description=str(desc_raw),
        priority=priority_raw if isinstance(priority_raw, int) else 50,
        patterns=patterns,
        instructions=instructions,
        path=path,
        enabled=bool(enabled_raw),
        version=str(version_raw),
    )


class SkillLoader:
    def __init__(self, skill_dirs: list[Path] | None = None) -> None:
        self._skill_dirs: list[Path] = skill_dirs or []

    def discover(self) -> list[SkillDef]:
        found: list[SkillDef] = []
        seen: set[str] = set()
        for directory in self._skill_dirs:
            if not directory.is_dir():
                continue
            for entry in sorted(directory.iterdir()):
                if entry.suffix.lower() in (".md", ".markdown") and not entry.name.startswith("_"):
                    name = entry.stem
                    if name not in seen:
                        seen.add(name)
                        try:
                            skill = load_skill(entry)
                            found.append(skill)
                        except Exception:
                            pass
                elif entry.is_dir():
                    manifest = entry / "skill.json"
                    if manifest.is_file():
                        import json
                        try:
                            data = json.loads(manifest.read_text(encoding="utf-8"))
                            name = data.get("name", entry.name)
                            if name not in seen:
                                seen.add(name)
                                instructions_file = entry / "instructions.md"
                                instructions = ""
                            if instructions_file.is_file():
                                instructions = instructions_file.read_text(encoding="utf-8")
                                skill = SkillDef(
                                    name=name,
                                    description=str(data.get("description", "")),
                                    priority=int(data.get("priority", 50)),
                                    patterns=list(data.get("patterns", [])),
                                    instructions=instructions,
                                    path=entry,
                                    enabled=bool(data.get("enabled", True)),
                                    version=str(data.get("version", "1.0.0")),
                                )
                                found.append(skill)
                        except Exception:
                            pass
        return found
