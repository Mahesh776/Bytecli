from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillDef:
    name: str
    description: str = ""
    priority: int = 50
    patterns: list[str] = field(default_factory=list)
    instructions: str = ""
    path: Path | None = None
    enabled: bool = True
    version: str = "1.0.0"


@dataclass
class SkillMatch:
    skill: SkillDef
    score: float = 0.0
