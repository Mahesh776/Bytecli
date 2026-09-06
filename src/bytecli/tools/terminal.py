# ruff: noqa: RUF012
# mypy: disable-error-code="override"
import asyncio
import platform
from pathlib import Path

from bytecli.tools.base import Tool, ToolParameter, ToolResult


class BashTool(Tool):
    name = "bash"
    description = "Execute a shell command with optional timeout. Returns stdout, stderr, and exit code."
    name_aliases = {
        "run_command",
        "run",
        "shell",
        "execute",
        "execute_command",
        "terminal",
        "command",
        "run_terminal",
        "shell_command",
        "exec_command",
    }
    arg_aliases = {
        "cmd": "command",
        "shell_command": "command",
        "exec": "command",
        "cwd": "workdir",
        "working_dir": "workdir",
        "dir": "workdir",
    }
    parameters = [
        ToolParameter(name="command", type="string", description="The command to execute"),
        ToolParameter(name="description", type="string", description="Brief description (5-10 words)", required=False),
        ToolParameter(name="timeout", type="integer", description="Timeout in milliseconds", required=False, default=120000),  # noqa: E501
        ToolParameter(name="workdir", type="string", description="Working directory for the command", required=False),
    ]

    async def _execute(self, command: str, description: str = "", timeout: int = 120000, workdir: str | None = None) -> ToolResult:  # noqa: E501
        shell = "powershell" if platform.system() == "Windows" else "bash"
        shell_flag = "-c" if shell == "bash" else "-Command"
        cwd = workdir or str(Path.cwd())
        try:
            proc = await asyncio.create_subprocess_exec(
                shell, shell_flag, command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout / 1000)
            except TimeoutError:
                proc.kill()
                return ToolResult(success=False, error=f"Command timed out after {timeout}ms", tool_name=self.name)
        except FileNotFoundError as e:
            return ToolResult(success=False, error=f"Shell not found: {e}", tool_name=self.name)
        except Exception as e:
            return ToolResult(success=False, error=f"Execution failed: {e}", tool_name=self.name)
        stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""
        retcode = proc.returncode or 0
        output = {"stdout": stdout_str, "stderr": stderr_str, "exit_code": retcode, "command": command}
        return ToolResult(
            success=retcode == 0,
            data=output,
            error=stderr_str if retcode != 0 else None,
            tool_name=self.name,
        )
