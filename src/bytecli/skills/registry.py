from bytecli.skills.base import SkillDef


class SkillRegistry:
    _skills: dict[str, SkillDef] = {}  # noqa: RUF012

    @classmethod
    def register(cls, skill: SkillDef) -> None:
        cls._skills[skill.name] = skill

    @classmethod
    def get(cls, name: str) -> SkillDef | None:
        return cls._skills.get(name)

    @classmethod
    def list_skills(cls) -> list[SkillDef]:
        return list(cls._skills.values())

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._skills.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        cls._skills.clear()

    @classmethod
    def is_loaded(cls, name: str) -> bool:
        return name in cls._skills
