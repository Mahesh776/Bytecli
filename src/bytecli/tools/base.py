from typing import Any

from bytecli.core.errors import ToolExecutionError


class ToolResult:
    def __init__(
        self,
        success: bool = True,
        data: Any = None,
        error: str | None = None,
        tool_name: str = "",
    ) -> None:
        self.success = success
        self.data = data
        self.error = error
        self.tool_name = tool_name


class ToolParameter:
    def __init__(
        self,
        name: str,
        type: str,
        description: str = "",
        required: bool = True,
        default: Any = None,
    ) -> None:
        self.name = name
        self.type = type
        self.description = description
        self.required = required
        self.default = default

    def to_json_schema(self) -> dict[str, Any]:
        type_map: dict[str, dict[str, Any]] = {
            "string": {"type": "string"},
            "integer": {"type": "integer"},
            "number": {"type": "number"},
            "boolean": {"type": "boolean"},
            "array": {"type": "array"},
            "object": {"type": "object"},
        }
        schema = type_map.get(self.type, {"type": "string"}).copy()
        if self.description:
            schema["description"] = self.description
        if not self.required and self.default is not None:
            schema["default"] = self.default
        return schema


class Tool:
    name: str = ""
    description: str = ""
    parameters: list[ToolParameter] = []  # noqa: RUF012
    requires_approval: bool = False
    arg_aliases: dict[str, str] = {}  # noqa: RUF012
    name_aliases: set[str] = set()  # noqa: RUF012

    def get_json_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        normalized = {self.arg_aliases.get(key, key): value for key, value in kwargs.items()}
        param_names = [p.name for p in self.parameters]
        missing = [p.name for p in self.parameters if p.required and p.name not in normalized]
        if missing:
            required = ", ".join(p.name for p in self.parameters if p.required) or "(none)"
            received = ", ".join(normalized) or "none"
            return ToolResult(
                success=False,
                error=(
                    f"Missing required parameter(s): {', '.join(missing)}. "
                    f"Required parameters: {required}. Received: {received}."
                ),
                tool_name=self.name,
            )
        try:
            return await self._execute(**normalized)
        except ToolExecutionError:
            raise
        except Exception as e:
            hint = ""
            if isinstance(e, TypeError):
                hint = f" Valid parameters: {', '.join(param_names)}."
            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {e}.{hint}",
                tool_name=self.name,
            )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

