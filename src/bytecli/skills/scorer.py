import re
from typing import Any

from bytecli.skills.base import SkillDef, SkillMatch


class SkillScorer:
    def __init__(self) -> None:
        self._compiled: dict[str, re.Pattern[Any]] = {}

    def score(self, user_input: str, skills: list[SkillDef]) -> list[SkillMatch]:
        matches: list[SkillMatch] = []
        for skill in skills:
            if not skill.enabled or not skill.patterns:
                continue
            matched = False
            for pattern in skill.patterns:
                compiled = self._get_pattern(skill.name, pattern)
                if compiled.search(user_input):
                    matched = True
                    break
            if matched:
                score = skill.priority / 100.0
                matches.append(SkillMatch(skill=skill, score=score))

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    def _get_pattern(self, skill_name: str, pattern: str) -> re.Pattern[Any]:
        key = f"{skill_name}:{pattern}"
        if key not in self._compiled:
            self._compiled[key] = re.compile(pattern, re.IGNORECASE)
        return self._compiled[key]

    def clear_cache(self) -> None:
        self._compiled.clear()
