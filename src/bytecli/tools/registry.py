from typing import Any

from bytecli.tools.base import Tool


class ToolRegistry:
    _tools: dict[str, Tool] = {}  # noqa: RUF012

    @classmethod
    def register(cls, tool: Tool) -> None:
        cls._tools[tool.name.lower()] = tool

    @classmethod
    def get(cls, name: str) -> Tool | None:
        normalized = name.strip().lower()
        tool = cls._tools.get(normalized)
        if tool is not None:
            return tool
        for candidate in cls._tools.values():
            if normalized in {alias.lower() for alias in candidate.name_aliases}:
                return candidate
        return None

    @classmethod
    def list_tools(cls) -> list[Tool]:
        return list(cls._tools.values())

    @classmethod
    def get_json_schemas(cls) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.get_json_schema(),
            }
            for tool in cls._tools.values()
        ]

    @classmethod
    async def execute_tool(cls, name: str, **kwargs: Any) -> Any:
        tool = cls.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        return await tool.execute(**kwargs)
