import json
import re
import textwrap
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from bytecli.config.schema import select_system_prompt
from bytecli.core.errors import AgentLoopError, AgentTerminationError
from bytecli.core.events import EventBus
from bytecli.core.turn import AgentConfig, AgentOutput, TurnRecord, TurnStats, message_entry_to_provider
from bytecli.memory.manager import MemoryManager
from bytecli.memory.types import MessageEntry, estimate_tokens
from bytecli.providers.base import Provider
from bytecli.providers.types import (
    Choice,
    CompletionRequest,
    CompletionResponse,
    Message,
    Role,
    ToolCall,
    Usage,
)
from bytecli.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from bytecli.plugins.manager import PluginManager
    from bytecli.skills.manager import SkillManager


class AgentLoop:
    _LOOP_HINT = (
        "[System] You are repeating the same tool call without making progress. "
        "STOP calling tools now. Give your final answer directly using the information you already have."
    )
    _FORCE_ANSWER_HINT = (
        "[System] You have used all of your available turns. "
        "STOP using tools. Provide your final answer now, based only on the results you already have."
    )
    _CONTENT_TOOLS = frozenset({"write", "append_file"})
    _SAVE_FOLDER_RE = re.compile(
        r"(?:save|put|store|write|keep|copy|create).{0,40}?"
        r"(?:to|in|into|at)\s+(?:this\s+)?(?:folder|directory|location|path|dir)?\s*[:.]?\s*"
        r"(?P<path>(?:[A-Za-z]:[\\/]|/|~/)[^\s\"'`]+)",
        re.IGNORECASE,
    )
    _WIN_ABS_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'`]*")
    _UNIX_ABS_RE = re.compile(r"(?:/|~/)[^\s\"'`]*")
    _SAVE_FOLDER_HINT_TEMPLATE = (
        "[System] The user wants output files saved in this folder: {folder}.\n"
        "When you write a file, ALWAYS use Tool: write with an absolute path INSIDE that folder, "
        "for example path: {folder}\\example.html (append a filename to the folder path).\n"
        "Never just print the code if the user asked you to save it - you MUST call Tool: write and "
        "then tell the user the saved path."
    )
    _PLACEHOLDER_MARKERS = (
        "full code here",
        "your code here",
        "your code goes here",
        "add your",
        "your content",
        "insert your",
        "your implementation",
        "your logic here",
        "game logic here",
        "your game logic",
        "logic here",
        "complete code here",
        "content here",
        "example code",
        "placeholder",
        "some content",
        "lorem ipsum",
        "todo",
        "domcontentloaded",
        "complete working",
        "wrapped in domcontentloaded",
        "if needed",
        "the code",
    )

    def __init__(
        self,
        provider: Provider,
        tool_registry: ToolRegistry | None = None,
        memory: MemoryManager | None = None,
        event_bus: EventBus | None = None,
        config: AgentConfig | None = None,
        plugin_manager: PluginManager | None = None,
        skill_manager: SkillManager | None = None,
    ) -> None:
        self._provider = provider
        self._tool_registry = tool_registry
        self._memory = memory
        self._event_bus = event_bus
        self._config = config or AgentConfig()
        self._plugin_manager = plugin_manager
        self._skill_manager = skill_manager
        self._resolved_model: str | None = None
        self._system_prompt: str | None = None
        self._last_fingerprint: tuple[tuple[str, frozenset[tuple[str, str]]], ...] | None = None
        self._repeat_count = 0

    @property
    def config(self) -> AgentConfig:
        return self._config

    async def _resolve_model(self) -> None:
        if self._resolved_model is not None:
            return
        if self._config.model and self._config.model not in ("local-model", "unknown", ""):
            self._resolved_model = self._config.model
            return
        try:
            resolved = await self._provider.resolve_model_name()
            if resolved:
                self._resolved_model = resolved
            else:
                self._resolved_model = self._config.model
        except Exception:
            self._resolved_model = self._config.model

    async def resolved_model(self) -> str:
        await self._resolve_model()
        return self._resolved_model or self._config.model

    def effective_system_prompt(self) -> str:
        if self._system_prompt is not None:
            return self._system_prompt
        model = self._resolved_model or self._config.model
        return select_system_prompt(model, self._config.system_prompt)

    async def run(self, user_input: str) -> AgentOutput:
        return await self._run_loop(user_input, stream=False)

    async def run_stream(self, user_input: str) -> AsyncIterator[Message]:
        output = await self._run_loop(user_input, stream=True)
        yield output.response

    async def _chat_once(self, request: CompletionRequest) -> CompletionResponse:
        if not self._config.streaming:
            return await self._provider.chat(request)
        try:
            return await self._accumulate_stream(request)
        except NotImplementedError:
            return await self._provider.chat(request)

    async def _accumulate_stream(self, request: CompletionRequest) -> CompletionResponse:
        parts: list[str] = []
        finish_reason: str | None = None
        resp_id = ""
        resp_model = request.model
        usage: Usage | None = None

        async for chunk in self._provider.chat_stream(request):
            if chunk.id:
                resp_id = chunk.id
            if chunk.model:
                resp_model = chunk.model
            if chunk.usage:
                usage = chunk.usage
            for choice in chunk.choices:
                if choice.delta is not None and choice.delta.content:
                    parts.append(choice.delta.content)
                elif choice.message is not None and choice.message.content:
                    parts.append(choice.message.content)
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

        message = Message(role=Role.ASSISTANT, content="".join(parts))
        return CompletionResponse(
            id=resp_id,
            model=resp_model,
            choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
            usage=usage,
        )

    async def _run_loop(self, user_input: str, stream: bool = False) -> AgentOutput:
        await self._resolve_model()
        messages = await self._build_message_list(user_input)
        turns: list[TurnRecord] = []
        truncated = False
        error: str | None = None
        self._last_fingerprint = None
        self._repeat_count = 0

        for turn_num in range(1, self._config.max_turns + 1):
            await self._publish_event("agent.turn.start", {"turn": turn_num})
            await self._fire_hook("agent.before_turn", turn=turn_num, messages=messages)

            request_messages = self._build_request_messages(messages)
            request = CompletionRequest(
                model=self._config.model,
                messages=request_messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                stream=stream,
            )

            await self._fire_hook("agent.before_think", turn=turn_num, request=request)

            try:
                turn_start = time.time()
                response = await self._chat_once(request)
                turn_duration = time.time() - turn_start
            except Exception as e:
                await self._publish_event("agent.error", {"error": str(e), "turn": turn_num})
                await self._fire_hook("agent.on_error", turn=turn_num, error=str(e))
                error = f"Provider error at turn {turn_num}: {e}"
                raise AgentLoopError(error) from e

            await self._fire_hook("agent.after_think", turn=turn_num, response=response)

            if not response.choices:
                error = f"Empty response from provider at turn {turn_num}"
                raise AgentLoopError(error)

            choice = response.choices[0]
            response_msg = choice.message

            if response_msg is None:
                error = f"No message in response at turn {turn_num}"
                raise AgentLoopError(error)

            tool_calls = list(response_msg.tool_calls) if response_msg.tool_calls else []
            tool_results: list[Message] = []

            if not tool_calls and response_msg.content:
                text_tool_calls = self._parse_text_tool_calls(response_msg.content)
                if text_tool_calls:
                    tool_calls = text_tool_calls
                else:
                    cleaned = self._strip_dangling_tool_calls(response_msg.content)
                    if cleaned != response_msg.content:
                        response_msg = replace(response_msg, content=cleaned)

            if tool_calls:
                tool_names = [tc.function.get("name", "") for tc in tool_calls]
                await self._publish_event("agent.action", {"tool_calls": tool_names})

                fingerprint = self._tool_batch_fingerprint(tool_calls)
                if fingerprint and fingerprint == self._last_fingerprint:
                    self._repeat_count += 1
                else:
                    self._repeat_count = 0
                self._last_fingerprint = fingerprint

                for tc in tool_calls:
                    tool_name = tc.function.get("name", "")
                    tool_args = tc.function.get("arguments", "{}")

                    try:
                        args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                    except json.JSONDecodeError:
                        args = {}

                    await self._publish_event("tool.execution.start", {"tool": tool_name, "args": args})
                    await self._fire_hook("tool.before_execute", tool=tool_name, args=args)

                    placeholder = (
                        isinstance(args, dict)
                        and tool_name in self._CONTENT_TOOLS
                        and self._is_placeholder_content(str(args.get("content") or ""))
                    )
                    if placeholder:
                        result_content = (
                            "Error: The content you provided is a placeholder, not real code. "
                            "Write the COMPLETE file content with the full working code for exactly "
                            "what the user asked. Never use placeholders like '(full code here)' or "
                            "'...' or substitute a simpler example. Retry with the real code."
                        )
                    elif self._tool_registry is not None:
                        try:
                            tool_result = await self._tool_registry.execute_tool(tool_name, **args)
                            result_content = (
                                str(tool_result.data) if tool_result.success else f"Error: {tool_result.error}"
                            )
                            external_hint = self._check_external_file_refs(tool_name, args, result_content)
                            if external_hint:
                                result_content = f"{result_content}\n\n{external_hint}"
                        except ValueError as e:
                            available = ", ".join(sorted(t.name for t in self._tool_registry.list_tools()))
                            result_content = f"Error: {e}. Available tools: {available}"
                    else:
                        result_content = f"Tool '{tool_name}' not available (no tool registry)"

                    await self._publish_event("tool.execution.end", {"tool": tool_name})
                    await self._fire_hook("tool.after_execute", tool=tool_name, result=result_content)

                    tool_results.append(
                        Message(
                            role=Role.TOOL,
                            content=result_content,
                            tool_call_id=tc.id,
                            name=tool_name,
                        )
                    )

                messages.append(response_msg)
                messages.extend(tool_results)

                if self._repeat_count >= max(1, self._config.max_repeated_tool_calls):
                    self._repeat_count = 0
                    self._last_fingerprint = None
                    messages.append(Message(role=Role.USER, content=self._LOOP_HINT))
            else:
                messages.append(response_msg)

            turn_record = TurnRecord(
                turn_number=turn_num,
                response_message=response_msg,
                tool_calls=tool_calls,
                tool_results=tool_results,
                usage=response.usage,
                duration=turn_duration,
            )
            turns.append(turn_record)

            await self._publish_event("agent.turn.end", {
                "turn": turn_num,
                "tool_count": len(tool_calls),
                "finished": len(tool_calls) == 0,
            })
            await self._fire_hook("agent.after_turn", turn=turn_num, tool_count=len(tool_calls))

            if not tool_calls:
                break

            if self._check_context_overflow(messages):
                truncated = True
                break
        else:
            error = f"Exceeded maximum turns ({self._config.max_turns})"
            if self._config.max_turns > 0:
                forced = await self._attempt_final_answer(messages)
                if forced is not None:
                    messages.append(forced)
                    turns.append(
                        TurnRecord(
                            turn_number=self._config.max_turns + 1,
                            response_message=forced,
                            tool_calls=[],
                            tool_results=[],
                        )
                    )
                    await self._store_conversation(user_input, messages)
                    stats = self._compute_stats(turns)
                    return AgentOutput(
                        messages=messages,
                        response=forced,
                        turns=turns,
                        stats=stats,
                        truncated=True,
                        error=error,
                    )
                raise AgentTerminationError(error)

        await self._store_conversation(user_input, messages)

        stats = self._compute_stats(turns)

        return AgentOutput(
            messages=messages,
            response=messages[-1],
            turns=turns,
            stats=stats,
            truncated=truncated,
            error=error,
        )

    async def _build_message_list(self, user_input: str) -> list[Message]:
        messages: list[Message] = []

        if self._memory is not None:
            context = await self._memory.get_context()
            for mem_msg in context:
                messages.append(message_entry_to_provider(mem_msg))

        system_content = self._system_prompt or self.effective_system_prompt()
        self._system_prompt = system_content

        available_tools = self._available_tool_names()
        if available_tools:
            system_content = f"{system_content}\n\nAvailable tools: {available_tools}"

        if self._skill_manager is not None:
            skill_instructions = self._skill_manager.get_matching_instructions(user_input)
            if skill_instructions:
                if system_content:
                    system_content += "\n\n" + skill_instructions
                else:
                    system_content = skill_instructions

        system_content = self._append_save_folder_hint(system_content, user_input)

        if system_content:
            system_msg = Message(role=Role.SYSTEM, content=system_content)
            messages.insert(0, system_msg)

        user_msg = Message(role=Role.USER, content=user_input)
        messages.append(user_msg)

        if self._memory is not None:
            user_entry = MessageEntry(role="user", content=user_input, token_count=estimate_tokens(user_input))
            await self._memory.add_message(user_entry)

        return messages

    def _available_tool_names(self) -> str:
        if self._tool_registry is None:
            return ""
        return ", ".join(sorted(tool.name for tool in self._tool_registry.list_tools()))

    def _looks_like_new_arg(self, line: str, open_indent: int | None) -> bool:
        match = re.match(r'^\s*(\w+)\s*:', line)
        if not match:
            return False
        if line.rstrip().endswith(";"):
            return False
        if open_indent is None:
            return True
        line_indent = len(line) - len(line.lstrip())
        return line_indent <= open_indent

    def _check_external_file_refs(self, tool_name: str, args: Any, result: str) -> str | None:
        if tool_name not in self._CONTENT_TOOLS:
            return None
        if not isinstance(args, dict):
            return None
        file_path = str(args.get("file_path") or args.get("path") or "")
        content = str(args.get("content") or "")
        if not file_path.lower().endswith((".html", ".htm")):
            return None
        if "src=" not in content and "href=" not in content:
            return None
        refs = []
        for m in re.finditer(r'(?:src|href)\s*=\s*["\']([^"\']+)["\']', content):
            ref = m.group(1)
            if ref.startswith(("http://", "https://", "#", "data:")):
                continue
            if ref.endswith((".js", ".css")):
                refs.append(ref)
        if not refs:
            return None
        return (
            "WARNING: The HTML file references external files that may not exist yet: "
            f"{', '.join(refs)}. The user asked for a self-contained game, so put ALL CSS and "
            "JavaScript INLINE in the single HTML file (no separate .js/.css files). If you already "
            "wrote those files, re-write the HTML to inline them, or ensure the referenced files exist."
        )

    def _parse_text_tool_calls(self, text: str) -> list[ToolCall]:
        calls: list[ToolCall] = []
        lines = text.strip().split("\n")
        in_tool_block = False
        current_name = ""
        current_args: dict[str, str] = {}
        open_key: str | None = None
        open_value: list[str] = []
        open_indented: bool | None = None
        open_indent: int | None = None
        i = 0
        n = len(lines)

        def append_call(name: str, args: dict[str, str]) -> None:
            calls.append(
                ToolCall(
                    id=f"text_{len(calls)}",
                    type="function",
                    function={"name": name, "arguments": json.dumps(args)},
                )
            )

        def flush_open() -> None:
            nonlocal open_key, open_value, open_indented, open_indent
            if open_key is not None:
                value = textwrap.dedent("\n".join(open_value).strip("\n"))
                if value.strip():
                    current_args[open_key] = value.strip("\n")
                open_key = None
                open_value = []
                open_indented = None
                open_indent = None

        def flush_call() -> None:
            flush_open()
            nonlocal in_tool_block, current_name, current_args
            if in_tool_block and current_name and current_args:
                append_call(current_name, current_args)
            in_tool_block = False
            current_name = ""
            current_args = {}

        while i < n:
            line = lines[i].strip()

            if open_key is not None:
                if self._looks_like_new_arg(lines[i], open_indent):
                    flush_open()
                    continue
                if re.match(r'(?i)^Tool\s*:', line):
                    flush_open()
                    continue
                if not line:
                    open_value.append("")
                    i += 1
                    continue
                if open_indented is None:
                    open_indented = lines[i].startswith((" ", "\t"))
                elif open_indented and not lines[i].startswith((" ", "\t")):
                    flush_open()
                    continue
                open_value.append(lines[i])
                i += 1
                continue

            inline = re.match(r'(?i)^Tool:\s*(\w+)\s*\((.*)\)\s*$', line)
            if inline:
                flush_call()
                name = inline.group(1)
                args_str = inline.group(2).strip()
                kwargs: dict[str, str] = {}
                if args_str:
                    pattern = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s,)]+))')
                    for match in pattern.finditer(args_str):
                        key = match.group(1)
                        value = match.group(2) or match.group(3) or match.group(4) or ""
                        kwargs[key] = value
                if kwargs:
                    append_call(name, kwargs)
                i += 1
                continue

            tl = re.match(r'(?i)^Tool:\s*(\w+)$', line)
            if tl:
                flush_call()
                in_tool_block = True
                current_name = tl.group(1)
                current_args = {}
                i += 1
                continue

            if in_tool_block:
                kv = re.match(r'^\s*(\w+)\s*:\s*(.*)$', lines[i])
                if kv:
                    key = kv.group(1)
                    value = kv.group(2).strip()
                    if value:
                        current_args[key] = value
                        i += 1
                    else:
                        open_key = key
                        open_value = []
                        open_indent = len(lines[i]) - len(lines[i].lstrip())
                        i += 1
                    continue
                flush_call()

            i += 1

        flush_call()

        return calls

    def _strip_dangling_tool_calls(self, text: str) -> str:
        lines = text.strip().split("\n")
        kept: list[str] = []
        for idx, line in enumerate(lines):
            if re.match(r'(?i)^\s*Tool:\s*\w+\s*(\(\))?\s*$', line.strip()):
                next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
                if next_line and re.match(r'^\s+\w+\s*:', next_line):
                    kept.append(line)
                continue
            kept.append(line)
        return "\n".join(kept).strip()

    def _build_request_messages(self, messages: list[Message]) -> list[Message]:
        has_system = any(m.role == Role.SYSTEM for m in messages)
        system_prompt = self._system_prompt or self.effective_system_prompt()
        if system_prompt and not has_system:
            return [Message(role=Role.SYSTEM, content=system_prompt), *messages]
        return messages

    def _tool_batch_fingerprint(
        self,
        tool_calls: list[ToolCall],
    ) -> tuple[tuple[str, frozenset[tuple[str, str]]], ...]:
        fingerprint: list[tuple[str, frozenset[tuple[str, str]]]] = []
        for tc in tool_calls:
            name = tc.function.get("name", "")
            raw_args = tc.function.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            items = frozenset((str(k), str(v)) for k, v in sorted(args.items()))
            fingerprint.append((name, items))
        return tuple(fingerprint)

    def _is_placeholder_content(self, content: str) -> bool:
        lowered = content.strip().lower()
        if not lowered:
            return True
        if len(lowered) <= 60 and "\n" not in lowered:
            return True
        return any(marker in lowered for marker in self._PLACEHOLDER_MARKERS)

    def _extract_target_folder(self, user_input: str) -> str | None:
        text = user_input.strip()
        match = self._SAVE_FOLDER_RE.search(text)
        if match:
            return match.group("path").rstrip("\\/")
        match = self._WIN_ABS_RE.search(text)
        if match:
            return match.group(0).rstrip("\\/")
        match = self._UNIX_ABS_RE.search(text)
        if match:
            return match.group(0).rstrip("\\/")
        return None

    def _append_save_folder_hint(self, system_content: str, user_input: str) -> str:
        folder = self._extract_target_folder(user_input)
        if not folder:
            return system_content
        hint = self._SAVE_FOLDER_HINT_TEMPLATE.format(folder=folder)
        return f"{system_content}\n\n{hint}" if system_content else hint

    async def _attempt_final_answer(self, messages: list[Message]) -> Message | None:
        request_messages = self._build_request_messages(messages)
        prompt = Message(role=Role.USER, content=self._FORCE_ANSWER_HINT)
        request = CompletionRequest(
            model=self._config.model,
            messages=[*request_messages, prompt],
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            stream=False,
        )
        try:
            response = await self._provider.chat(request)
        except Exception:
            return None
        if not response.choices:
            return None
        msg = response.choices[0].message
        if msg is None or msg.tool_calls or not msg.content:
            return None
        return msg

    def _check_context_overflow(self, messages: list[Message]) -> bool:
        if self._memory is None:
            return False
        total = sum(len(m.content or "") for m in messages)
        return total > self._config.max_tokens * 4

    def _compute_stats(self, turns: list[TurnRecord]) -> TurnStats:
        stats = TurnStats()
        stats.total_turns = len(turns)
        for t in turns:
            stats.total_tool_calls += len(t.tool_calls)
            if t.usage:
                stats.total_prompt_tokens += t.usage.prompt_tokens
                stats.total_completion_tokens += t.usage.completion_tokens
            stats.total_duration += t.duration
        return stats

    async def _store_conversation(self, user_input: str, messages: list[Message]) -> None:
        if self._memory is None:
            return
        for msg in messages:
            if msg.role == Role.USER and msg.content == user_input:
                continue
            entry = MessageEntry(
                role=str(msg.role.value),
                content=msg.content or "",
                token_count=estimate_tokens(msg.content or ""),
            )
            await self._memory.add_message(entry)

    async def _publish_event(self, event_type: str, data: dict[str, object]) -> None:
        if self._event_bus is not None:
            from bytecli.core.events import Event

            event = Event(type=event_type, timestamp=time.time(), data=data)
            await self._event_bus.publish(event)

    async def _fire_hook(self, hook: str, **kwargs: object) -> None:
        if self._plugin_manager is not None:
            await self._plugin_manager.fire_hook(hook, **kwargs)
