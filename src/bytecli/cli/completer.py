from pathlib import Path

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

SLASH_COMMANDS = [
    "/help", "/exit", "/quit", "/clear", "/history",
    "/model", "/provider", "/temperature", "/system",
    "/max_tokens", "/top_p", "/config",
    "/memory", "/tokens", "/compact", "/forget", "/context",
    "/tools", "/tool",
    "/workspace", "/tree", "/git", "/language", "/frameworks",
    "/status", "/retry", "/undo",
    "/log", "/debug", "/version", "/env",
    "/plugins", "/plugin", "/skills", "/skill",
    "/export", "/import",
]


class ByteCliCompleter(Completer):
    def __init__(self, tool_names: list[str] | None = None) -> None:
        self._tool_names = tool_names or []

    def update_tool_names(self, names: list[str]) -> None:
        self._tool_names = names

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> list[Completion]:
        text = document.text_before_cursor
        word = self._get_current_word(text)

        if not word:
            return []

        results: list[Completion] = []

        if text.startswith("/"):
            results.extend(self._complete_command(word))
        else:
            results.extend(self._complete_tool(word))
            results.extend(self._complete_filepath(word))

        return results

    def _get_current_word(self, text: str) -> str:
        text = text.rstrip()
        if not text:
            return ""
        idx = len(text) - 1
        while idx >= 0 and not text[idx].isspace():
            idx -= 1
        return text[idx + 1 :]

    def _complete_command(self, word: str) -> list[Completion]:
        results: list[Completion] = []
        for cmd in SLASH_COMMANDS:
            if cmd.startswith(word) and word != cmd:
                results.append(Completion(cmd, -len(word)))
        return results

    def _complete_tool(self, word: str) -> list[Completion]:
        results: list[Completion] = []
        for name in self._tool_names:
            if name.startswith(word) and word != name:
                results.append(Completion(name, -len(word)))
        return results

    def _complete_filepath(self, word: str) -> list[Completion]:
        results: list[Completion] = []
        try:
            p = Path(word)
            parent = p.parent if p.parent else Path(".")
            if parent.exists():
                for child in parent.iterdir():
                    candidate = str(child)
                    if candidate.startswith(word):
                        display = candidate + "/" if child.is_dir() else candidate
                        results.append(Completion(display, -len(word)))
        except (OSError, ValueError):
            pass
        return results
