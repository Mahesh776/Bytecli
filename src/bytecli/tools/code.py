# ruff: noqa: RUF012
# mypy: disable-error-code="override"
import asyncio
import platform
import sys
from pathlib import Path
from typing import Any

from bytecli.tools.base import Tool, ToolParameter, ToolResult


async def _run_subprocess(
    args: list[str],
    timeout_ms: int,
    workdir: str | None,
    tool_name: str,
) -> ToolResult:
    cwd = workdir or str(Path.cwd())
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_ms / 1000)
        except TimeoutError:
            proc.kill()
            return ToolResult(success=False, error=f"Command timed out after {timeout_ms}ms", tool_name=tool_name)
    except FileNotFoundError as e:
        return ToolResult(success=False, error=f"Executable not found: {e}", tool_name=tool_name)
    except Exception as e:
        return ToolResult(success=False, error=f"Execution failed: {e}", tool_name=tool_name)
    stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
    stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""
    retcode = proc.returncode or 0
    return ToolResult(
        success=retcode == 0,
        data={"stdout": stdout_str, "stderr": stderr_str, "exit_code": retcode},
        error=stderr_str if retcode != 0 else None,
        tool_name=tool_name,
    )


class RunPythonTool(Tool):
    name = "run_python"
    description = "Execute Python code and capture output. Pass 'code' for a snippet or 'file' for a script."
    name_aliases = {"python", "execute_python", "py", "run_python_code", "exec_python"}
    arg_aliases = {
        "script": "file",
        "script_path": "file",
        "file_path": "file",
        "path": "file",
        "snippet": "code",
        "python_code": "code",
        "cwd": "workdir",
        "dir": "workdir",
    }
    parameters = [
        ToolParameter(name="code", type="string", description="Python code to execute", required=False),
        ToolParameter(name="file", type="string", description="Path to a Python script to run", required=False),
        ToolParameter(name="args", type="array", description="Script arguments, or a single path to run", required=False),  # noqa: E501
        ToolParameter(name="workdir", type="string", description="Working directory for the process", required=False),
        ToolParameter(name="timeout", type="integer", description="Timeout in milliseconds", required=False, default=120000),  # noqa: E501
    ]

    async def _execute(
        self,
        code: str | None = None,
        file: str | None = None,
        args: Any = None,
        workdir: str | None = None,
        timeout: int = 120000,
    ) -> ToolResult:
        script_args = self._parse_args(args)
        if script_args and not file and not code:
            file = script_args[0]
            script_args = script_args[1:]
        if not code and not file:
            return ToolResult(success=False, error="Provide either 'code' or 'file'", tool_name=self.name)
        if code and file:
            return ToolResult(success=False, error="Provide only one of 'code' or 'file'", tool_name=self.name)
        if file:
            script = Path(file)
            if not script.exists():
                return ToolResult(success=False, error=f"File not found: {file}", tool_name=self.name)
            command_args = [sys.executable, str(script), *script_args]
        else:
            assert code is not None
            command_args = [sys.executable, "-c", code]
        return await _run_subprocess(command_args, timeout, workdir, self.name)

    @staticmethod
    def _parse_args(args: Any) -> list[str]:
        if args is None:
            return []
        if isinstance(args, str):
            value = args.strip()
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]
            return [part.strip().strip("'\"") for part in value.split(",") if part.strip()]
        if isinstance(args, list):
            return [str(item).strip().strip("'\"") for item in args if str(item).strip()]
        if isinstance(args, tuple):
            return [str(item).strip().strip("'\"") for item in args if str(item).strip()]
        return [str(args)]


class RunTestsTool(Tool):
    name = "run_tests"
    description = "Run the test suite with pytest and capture the results."
    name_aliases = {"test", "pytest", "run_test", "run_pytest", "test_suite"}
    arg_aliases = {
        "test_path": "path",
        "target": "path",
        "dir": "path",
        "cwd": "workdir",
    }
    parameters = [
        ToolParameter(name="path", type="string", description="Test path or expression (defaults to 'tests')", required=False),  # noqa: E501
        ToolParameter(name="args", type="string", description="Extra pytest arguments", required=False),
        ToolParameter(name="workdir", type="string", description="Working directory for the process", required=False),
        ToolParameter(name="timeout", type="integer", description="Timeout in milliseconds", required=False, default=300000),  # noqa: E501
    ]

    async def _execute(
        self,
        path: str | None = None,
        args: str | None = None,
        workdir: str | None = None,
        timeout: int = 300000,
    ) -> ToolResult:
        cmd = [sys.executable, "-m", "pytest", "-q", "--no-header"]
        if path:
            cmd.append(path)
        if args:
            cmd.extend(args.split())
        return await _run_subprocess(cmd, timeout, workdir, self.name)


class DebugTool(Tool):
    name = "debug"
    description = "Run a script or command and capture the full output including errors and tracebacks. Use to debug failing code."  # noqa: E501
    name_aliases = {"debug_code", "run_and_debug", "run_script", "run_debug"}
    arg_aliases = {
        "cmd": "command",
        "shell_command": "command",
        "exec": "command",
        "script": "command",
        "cwd": "workdir",
        "dir": "workdir",
        "working_dir": "workdir",
    }
    parameters = [
        ToolParameter(name="command", type="string", description="The command to run (e.g. 'python main.py' or 'pytest')"),  # noqa: E501
        ToolParameter(name="workdir", type="string", description="Working directory for the command", required=False),
        ToolParameter(name="timeout", type="integer", description="Timeout in milliseconds", required=False, default=120000),  # noqa: E501
    ]

    async def _execute(self, command: str, workdir: str | None = None, timeout: int = 120000) -> ToolResult:
        shell = "powershell" if platform.system() == "Windows" else "bash"
        shell_flag = "-Command" if shell == "powershell" else "-c"
        return await _run_subprocess([shell, shell_flag, command], timeout, workdir, self.name)
