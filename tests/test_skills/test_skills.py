import tempfile
from pathlib import Path

import pytest

from bytecli.core.agent import AgentLoop
from bytecli.core.errors import SkillLoadError
from bytecli.core.turn import AgentConfig
from bytecli.skills.base import SkillDef, SkillMatch
from bytecli.skills.loader import SkillLoader, load_skill
from bytecli.skills.manager import SkillManager
from bytecli.skills.registry import SkillRegistry
from bytecli.skills.scorer import SkillScorer


class TestSkillDef:
    def test_defaults(self) -> None:
        s = SkillDef(name="test")
        assert s.name == "test"
        assert s.description == ""
        assert s.priority == 50
        assert s.patterns == []
        assert s.instructions == ""
        assert s.path is None
        assert s.enabled is True
        assert s.version == "1.0.0"

    def test_full(self) -> None:
        s = SkillDef(
            name="python",
            description="Python dev",
            priority=80,
            patterns=["python", "def "],
            instructions="Use type hints",
            path=Path("/tmp/py.md"),
            enabled=False,
            version="2.0.0",
        )
        assert s.priority == 80
        assert s.patterns == ["python", "def "]
        assert s.instructions == "Use type hints"


class TestSkillMatch:
    def test_defaults(self) -> None:
        s = SkillDef(name="test")
        m = SkillMatch(skill=s)
        assert m.skill is s
        assert m.score == 0.0

    def test_with_score(self) -> None:
        s = SkillDef(name="test", priority=75)
        m = SkillMatch(skill=s, score=0.75)
        assert m.score == 0.75


class TestSkillLoader:
    def test_discover_no_dirs(self) -> None:
        loader = SkillLoader()
        assert loader.discover() == []

    def test_discover_missing_dir(self) -> None:
        loader = SkillLoader([Path("/nonexistent")])
        assert loader.discover() == []

    def test_discover_finds_md_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "python.md").write_text("""---
name: python
description: Python skills
priority: 80
patterns: ["python", "def "]
---

# Python Instructions
Use type hints everywhere.
""")
            loader = SkillLoader([d])
            skills = loader.discover()
            assert len(skills) == 1
            assert skills[0].name == "python"
            assert skills[0].priority == 80
            assert "def " in skills[0].patterns

    def test_discover_skips_underscore_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "_private.md").write_text("no")
            loader = SkillLoader([d])
            assert loader.discover() == []

    def test_load_markdown_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.md"
            path.write_text("""---
name: my_skill
description: My test skill
priority: 90
patterns: ["pattern1", "pattern2"]
---

# Instructions body
Content here
""")
            skill = load_skill(path)
            assert skill.name == "my_skill"
            assert skill.description == "My test skill"
            assert skill.priority == 90
            assert skill.patterns == ["pattern1", "pattern2"]
            assert "Instructions body" in skill.instructions
            assert "Content here" in skill.instructions
            assert skill.path == path

    def test_load_skill_no_frontmatter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "simple.md"
            path.write_text("Just instructions without frontmatter")
            skill = load_skill(path)
            assert skill.name == "simple"
            assert skill.instructions == "Just instructions without frontmatter"
            assert skill.priority == 50

    def test_load_skill_empty_frontmatter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.md"
            path.write_text("""---

---

Body content""")
            skill = load_skill(path)
            assert skill.name == "empty"
            assert skill.instructions == "Body content"

    def test_discover_directory_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            skill_dir = d / "my_skill"
            skill_dir.mkdir()
            (skill_dir / "skill.json").write_text("""{
                "name": "my_skill",
                "description": "Directory skill",
                "priority": 70,
                "patterns": ["test"]
            }""")
            (skill_dir / "instructions.md").write_text("Directory instructions")
            loader = SkillLoader([d])
            skills = loader.discover()
            assert len(skills) == 1
            assert skills[0].name == "my_skill"
            assert skills[0].priority == 70
            assert "Directory instructions" in skills[0].instructions


class TestSkillScorer:
    def test_no_match(self) -> None:
        scorer = SkillScorer()
        skills = [SkillDef(name="s1", patterns=["python"])]
        matches = scorer.score("hello world", skills)
        assert matches == []

    def test_simple_match(self) -> None:
        scorer = SkillScorer()
        skills = [SkillDef(name="python", priority=80, patterns=["python"])]
        matches = scorer.score("write a python function", skills)
        assert len(matches) == 1
        assert matches[0].skill.name == "python"
        assert matches[0].score == 0.8

    def test_case_insensitive(self) -> None:
        scorer = SkillScorer()
        skills = [SkillDef(name="js", patterns=["JavaScript"])]
        matches = scorer.score("I write javascript", skills)
        assert len(matches) == 1

    def test_regex_pattern(self) -> None:
        scorer = SkillScorer()
        skills = [SkillDef(name="functions", patterns=["def \\w+", "async def"])]
        matches = scorer.score("def my_function():", skills)
        assert len(matches) == 1

    def test_multiple_skills_priority_order(self) -> None:
        scorer = SkillScorer()
        skills = [
            SkillDef(name="low", priority=30, patterns=["python"]),
            SkillDef(name="high", priority=90, patterns=["python"]),
            SkillDef(name="mid", priority=60, patterns=["python"]),
        ]
        matches = scorer.score("write python code", skills)
        assert len(matches) == 3
        assert matches[0].skill.name == "high"
        assert matches[1].skill.name == "mid"
        assert matches[2].skill.name == "low"

    def test_disabled_skill_not_matched(self) -> None:
        scorer = SkillScorer()
        skills = [SkillDef(name="s1", patterns=["python"], enabled=False)]
        matches = scorer.score("python code", skills)
        assert matches == []

    def test_skill_no_patterns_not_matched(self) -> None:
        scorer = SkillScorer()
        skills = [SkillDef(name="s1", patterns=[])]
        matches = scorer.score("anything", skills)
        assert matches == []

    def test_cache_persistence(self) -> None:
        scorer = SkillScorer()
        skills = [SkillDef(name="s1", patterns=["cache_test"])]
        matches1 = scorer.score("cache_test here", skills)
        matches2 = scorer.score("cache_test here", skills)
        assert len(matches1) == 1
        assert len(matches2) == 1

    def test_clear_cache(self) -> None:
        scorer = SkillScorer()
        skills = [SkillDef(name="s1", patterns=["python"])]
        scorer.score("python", skills)
        scorer.clear_cache()
        assert len(scorer._compiled) == 0


class TestSkillRegistry:
    def setup_method(self) -> None:
        SkillRegistry.clear()

    def test_register_and_get(self) -> None:
        s = SkillDef(name="test_skill")
        SkillRegistry.register(s)
        assert SkillRegistry.get("test_skill") is s

    def test_get_nonexistent(self) -> None:
        assert SkillRegistry.get("missing") is None

    def test_list_skills(self) -> None:
        SkillRegistry.register(SkillDef(name="a"))
        SkillRegistry.register(SkillDef(name="b"))
        assert len(SkillRegistry.list_skills()) == 2

    def test_unregister(self) -> None:
        SkillRegistry.register(SkillDef(name="tmp"))
        assert SkillRegistry.is_loaded("tmp")
        SkillRegistry.unregister("tmp")
        assert not SkillRegistry.is_loaded("tmp")

    def test_is_loaded(self) -> None:
        assert not SkillRegistry.is_loaded("not_loaded")
        SkillRegistry.register(SkillDef(name="loaded"))
        assert SkillRegistry.is_loaded("loaded")

    def test_clear(self) -> None:
        SkillRegistry.register(SkillDef(name="a"))
        SkillRegistry.register(SkillDef(name="b"))
        SkillRegistry.clear()
        assert SkillRegistry.list_skills() == []


class TestSkillManager:
    def setup_method(self) -> None:
        SkillRegistry.clear()

    async def test_initialize_no_dirs(self) -> None:
        mgr = SkillManager(skill_dirs=[])
        loaded = await mgr.initialize()
        assert loaded == []
        assert mgr.is_initialized

    async def test_initialize_double(self) -> None:
        mgr = SkillManager(skill_dirs=[])
        await mgr.initialize()
        result = await mgr.initialize()
        assert result == []

    async def test_initialize_discovers_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "python.md").write_text("""---
name: python
priority: 80
patterns: ["python"]
---

Python instructions
""")
            mgr = SkillManager(skill_dirs=[d])
            loaded = await mgr.initialize()
            assert "python" in loaded
            assert SkillRegistry.is_loaded("python")

    async def test_bundled_web_frontend_skill_matches_web_requests(self) -> None:
        repo_root = Path(__file__).parents[2]
        skill_dir = repo_root / ".bytecli" / "skills"
        if not skill_dir.is_dir():
            pytest.skip("bundled skills directory not present")
        mgr = SkillManager(skill_dirs=[skill_dir])
        await mgr.initialize()
        web_requests = [
            "make a snake game in html",
            "build a website for my cafe",
            "html landing page with a button",
            "create a css card animation",
            "write a todo web app",
        ]
        non_web = ["fix this python bug", "what is 2+2", "react component?"]
        for req in web_requests:
            assert mgr.get_matching_instructions(req), f"expected match: {req}"
        for req in non_web:
            assert mgr.get_matching_instructions(req) == "", f"expected no match: {req}"
        instructions = mgr.get_matching_instructions("make a snake game")
        assert "DOCTYPE" in instructions
        assert "placeholder" in instructions

    def test_load_skill_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.md"
            path.write_text("""---
name: loaded_skill
priority: 75
patterns: ["test"]
---

Instructions
""")
            mgr = SkillManager()
            skill = mgr.load_skill_file(path)
            assert skill.name == "loaded_skill"
            assert skill.priority == 75

    def test_load_skill_file_graceful(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plain.md"
            path.write_text("Just text, no frontmatter, no errors")
            mgr = SkillManager()
            skill = mgr.load_skill_file(path)
            assert skill.name == "plain"
            assert "Just text" in skill.instructions

    def test_unload_skill(self) -> None:
        SkillRegistry.register(SkillDef(name="to_unload"))
        mgr = SkillManager()
        assert mgr.unload_skill("to_unload") is True
        assert not SkillRegistry.is_loaded("to_unload")

    def test_unload_nonexistent(self) -> None:
        mgr = SkillManager()
        assert mgr.unload_skill("missing") is False

    def test_get_matching_instructions(self) -> None:
        SkillRegistry.register(SkillDef(
            name="python",
            priority=80,
            patterns=["python"],
            instructions="Use type hints\nFollow PEP 8",
        ))
        mgr = SkillManager()
        instructions = mgr.get_matching_instructions("write python code")
        assert "Use type hints" in instructions
        assert "Follow PEP 8" in instructions

    def test_get_matching_instructions_no_match(self) -> None:
        SkillRegistry.register(SkillDef(
            name="python",
            patterns=["python"],
            instructions="Python stuff",
        ))
        mgr = SkillManager()
        instructions = mgr.get_matching_instructions("hello world")
        assert instructions == ""

    def test_get_matching_instructions_empty_registry(self) -> None:
        mgr = SkillManager()
        assert mgr.get_matching_instructions("anything") == ""

    def test_get_matching_instructions_truncates(self) -> None:
        SkillRegistry.register(SkillDef(
            name="long",
            patterns=["test"],
            instructions="X" * 50000,
        ))
        mgr = SkillManager(max_instructions_length=100)
        result = mgr.get_matching_instructions("test input")
        assert len(result) <= 100 + 50  # truncation message

    def test_list_skills(self) -> None:
        SkillRegistry.register(SkillDef(name="s1"))
        SkillRegistry.register(SkillDef(name="s2"))
        mgr = SkillManager()
        skills = mgr.list_skills()
        assert len(skills) == 2

    def test_get_skill(self) -> None:
        s = SkillDef(name="get_me")
        SkillRegistry.register(s)
        mgr = SkillManager()
        assert mgr.get_skill("get_me") is s
        assert mgr.get_skill("missing") is None

    def test_clear(self) -> None:
        SkillRegistry.register(SkillDef(name="a"))
        mgr = SkillManager()
        mgr.clear()
        assert SkillRegistry.list_skills() == []
        assert not mgr.is_initialized


class TestAgentLoopSkillIntegration:
    async def test_skill_injected_into_system_prompt(self) -> None:
        SkillRegistry.clear()
        SkillRegistry.register(SkillDef(
            name="python",
            priority=80,
            patterns=["python"],
            instructions="Use type hints and docstrings.",
        ))

        from bytecli.providers.base import Provider
        from bytecli.providers.types import (
            Choice,
            CompletionRequest,
            CompletionResponse,
            Message,
            ModelInfo,
            Role,
            Usage,
        )
        from collections.abc import AsyncIterator
        from typing import Any

        class FakeProvider(Provider):
            def __init__(self) -> None:
                super().__init__()
                self.last_request: CompletionRequest | None = None

            async def chat(self, request: CompletionRequest) -> CompletionResponse:
                self.last_request = request
                return CompletionResponse(
                    id="r1", model="test",
                    choices=[Choice(index=0, message=Message(role=Role.ASSISTANT, content="ok"), finish_reason="stop")],
                    usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
                )

            def chat_stream(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
                raise NotImplementedError

            async def list_models(self) -> list[ModelInfo]:
                return [ModelInfo(id="test")]

            def _build_headers(self) -> dict[str, str]:
                return {}

            def _build_chat_payload(self, request: CompletionRequest) -> dict[str, Any]:
                return {}

            def _parse_response(self, data: dict[str, Any]) -> CompletionResponse:
                raise NotImplementedError

            def _parse_stream_chunk(self, line: str) -> CompletionResponse | None:
                return None

        from bytecli.skills.manager import SkillManager

        skill_manager = SkillManager()
        await skill_manager.initialize()
        config = AgentConfig(system_prompt="You are helpful.")
        loop = AgentLoop(provider=FakeProvider(), config=config, skill_manager=skill_manager)
        await loop.run("Write python code")
        provider = loop._provider
        assert provider.last_request is not None
        system_msgs = [m for m in provider.last_request.messages if m.role == Role.SYSTEM]
        assert len(system_msgs) >= 1
        assert "Use type hints" in (system_msgs[0].content or "")
        SkillRegistry.clear()

    async def test_skill_without_system_prompt(self) -> None:
        SkillRegistry.clear()
        SkillRegistry.register(SkillDef(
            name="rust",
            patterns=["rust"],
            instructions="Use RAII patterns.",
        ))

        from bytecli.providers.base import Provider
        from bytecli.providers.types import (
            Choice,
            CompletionRequest,
            CompletionResponse,
            Message,
            ModelInfo,
            Role,
            Usage,
        )
        from collections.abc import AsyncIterator
        from typing import Any

        class FakeProvider2(Provider):
            def __init__(self) -> None:
                super().__init__()
                self.last_request: CompletionRequest | None = None

            async def chat(self, request: CompletionRequest) -> CompletionResponse:
                self.last_request = request
                return CompletionResponse(
                    id="r1", model="test",
                    choices=[Choice(index=0, message=Message(role=Role.ASSISTANT, content="ok"), finish_reason="stop")],
                    usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
                )

            def chat_stream(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
                raise NotImplementedError

            async def list_models(self) -> list[ModelInfo]:
                return [ModelInfo(id="test")]

            def _build_headers(self) -> dict[str, str]:
                return {}

            def _build_chat_payload(self, request: CompletionRequest) -> dict[str, Any]:
                return {}

            def _parse_response(self, data: dict[str, Any]) -> CompletionResponse:
                raise NotImplementedError

            def _parse_stream_chunk(self, line: str) -> CompletionResponse | None:
                return None

        from bytecli.skills.manager import SkillManager

        skill_manager = SkillManager()
        await skill_manager.initialize()
        loop = AgentLoop(provider=FakeProvider2(), config=AgentConfig(), skill_manager=skill_manager)
        await loop.run("I love rust")
        provider = loop._provider
        assert provider.last_request is not None
        system_msgs = [m for m in provider.last_request.messages if m.role == Role.SYSTEM]
        assert len(system_msgs) >= 1
        assert "RAII" in (system_msgs[0].content or "")
        SkillRegistry.clear()
