from typing import Any

from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


class RichRenderer:
    def __init__(self, theme: str = "monokai") -> None:
        self._console = Console()
        self._theme = theme

    @property
    def console(self) -> Console:
        return self._console

    def welcome(self, version: str, provider: str, model: str) -> None:
        from rich.panel import Panel

        text = Text()
        text.append("\nByteCli ", style="bold cyan")
        text.append(f"v{version}", style="yellow")
        text.append("\nAI Coding Agent Framework", style="dim")
        text.append("\nProvider: ", style="dim")
        text.append(provider, style="green")
        text.append(" | Model: ", style="dim")
        text.append(model, style="green")
        text.append("\nType ", style="dim")
        text.append("/help", style="bold yellow")
        text.append(" for commands, ", style="dim")
        text.append("/exit", style="bold yellow")
        text.append(" to quit", style="dim")
        self._console.print(Panel(text, border_style="cyan"))

    def user_message(self, content: str) -> None:
        self._console.print(f"\n[bold green]You:[/] {content}")

    def assistant_message(self, content: str) -> None:
        markdown = RichMarkdown(content)
        self._console.print("\n[bold blue]Assistant:[/]")
        self._console.print(markdown)

    def assistant_streaming(self, content: str) -> None:
        self._console.print(content, end="")

    def tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        panel = Panel(
            f"[bold yellow]{tool_name}[/]",
            title="[yellow]Tool Call[/]",
            border_style="yellow",
            subtitle=f"Args: {args}",
        )
        self._console.print(panel)

    def tool_result(self, tool_name: str, success: bool, data: Any) -> None:
        style = "green" if success else "red"
        label = "Success" if success else "Error"
        content = str(data)[:500]
        panel = Panel(content, title=f"[{style}]{label}[/] — {tool_name}", border_style=style)
        self._console.print(panel)

    def error(self, message: str) -> None:
        self._console.print(f"\n[bold red]Error:[/] {message}")

    def info(self, message: str) -> None:
        self._console.print(f"\n[bold cyan]Info:[/] {message}")

    def help_table(self, commands: list[tuple[str, str]], title: str = "Commands") -> None:
        table = Table(title=title, border_style="cyan")
        table.add_column("Command", style="bold yellow")
        table.add_column("Description", style="white")
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        self._console.print(table)

    def memory_stats_table(self, stats: dict[str, Any]) -> None:
        table = Table(title="Memory Stats", border_style="magenta")
        table.add_column("Store", style="bold")
        table.add_column("Entries", style="cyan")
        table.add_column("Tokens", style="yellow")
        for name, s in stats.items():
            table.add_row(name, str(getattr(s, "total_entries", "?")), str(getattr(s, "total_tokens", "?")))
        self._console.print(table)

    def code_block(self, code: str, language: str = "python") -> None:
        syntax = Syntax(code, language, theme=self._theme, line_numbers=True)
        self._console.print(syntax)

    def separator(self) -> None:
        self._console.print(Rule(style="dim"))

    def status(self, text: str = "Working...") -> None:
        self._console.print(f"[dim]{text}[/]")
