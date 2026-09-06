from pathlib import Path

from loguru import logger

from bytecli.core.errors import SkillLoadError
from bytecli.skills.base import SkillDef
from bytecli.skills.loader import SkillLoader, load_skill
from bytecli.skills.registry import SkillRegistry
from bytecli.skills.scorer import SkillScorer


class SkillManager:
    def __init__(
        self,
        skill_dirs: list[Path] | None = None,
        auto_load: bool = True,
        max_instructions_length: int = 20000,
    ) -> None:
        self._loader = SkillLoader(skill_dirs or [])
        self._scorer = SkillScorer()
        self._max_instructions_length = max_instructions_length
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> list[str]:
        if self._initialized:
            return []
        self._initialized = True
        loaded: list[str] = []
        discovered = self._loader.discover()
        for skill in discovered:
            try:
                SkillRegistry.register(skill)
                loaded.append(skill.name)
            except Exception:
                logger.exception("Failed to register skill '{}'", skill.name)
        if loaded:
            logger.debug("Loaded skills: {}", ", ".join(loaded))
        return loaded

    def load_skill_file(self, path: Path) -> SkillDef:
        try:
            skill = load_skill(path)
            SkillRegistry.register(skill)
            return skill
        except Exception as e:
            raise SkillLoadError(skill_name=path.stem, detail=str(e)) from e

    def unload_skill(self, name: str) -> bool:
        skill = SkillRegistry.get(name)
        if skill is None:
            return False
        SkillRegistry.unregister(name)
        return True

    def get_matching_instructions(self, user_input: str) -> str:
        skills = SkillRegistry.list_skills()
        matches = self._scorer.score(user_input, skills)
        if not matches:
            return ""

        parts: list[str] = []
        for match in matches:
            instructions = match.skill.instructions.strip()
            if instructions:
                parts.append(instructions)

        result = "\n\n".join(parts)
        if len(result) > self._max_instructions_length:
            result = result[:self._max_instructions_length] + "\n\n[instructions truncated]"
        return result

    def list_skills(self) -> list[SkillDef]:
        return SkillRegistry.list_skills()

    def get_skill(self, name: str) -> SkillDef | None:
        return SkillRegistry.get(name)

    def clear(self) -> None:
        SkillRegistry.clear()
        self._scorer.clear_cache()
        self._initialized = False
