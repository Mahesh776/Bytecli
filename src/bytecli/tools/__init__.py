from bytecli.tools.base import Tool
from bytecli.tools.code import DebugTool, RunPythonTool, RunTestsTool
from bytecli.tools.filesystem import (
    AppendFileTool,
    CopyFileTool,
    DeleteFileTool,
    EditTool,
    FileInfoTool,
    FileTreeTool,
    GlobTool,
    GrepTool,
    ListDirectoryTool,
    MkdirTool,
    MoveFileTool,
    OpenFileTool,
    ReadTool,
    WriteTool,
)
from bytecli.tools.git import GitTool
from bytecli.tools.registry import ToolRegistry
from bytecli.tools.terminal import BashTool
from bytecli.tools.web import DownloadTool, WebFetchTool, WebSearchTool

_registered = False


def _register_all() -> None:
    global _registered
    if _registered:
        return
    for tool in [
        ReadTool(),
        WriteTool(),
        EditTool(),
        AppendFileTool(),
        DeleteFileTool(),
        MoveFileTool(),
        ListDirectoryTool(),
        FileInfoTool(),
        FileTreeTool(),
        GlobTool(),
        GrepTool(),
        MkdirTool(),
        CopyFileTool(),
        OpenFileTool(),
        BashTool(),
        GitTool(),
        DebugTool(),
        RunPythonTool(),
        RunTestsTool(),
        WebFetchTool(),
        WebSearchTool(),
        DownloadTool(),
    ]:
        ToolRegistry.register(tool)
    _registered = True


_register_all()


__all__ = [
    "AppendFileTool",
    "BashTool",
    "CopyFileTool",
    "DebugTool",
    "DeleteFileTool",
    "DownloadTool",
    "EditTool",
    "FileInfoTool",
    "FileTreeTool",
    "GitTool",
    "GlobTool",
    "GrepTool",
    "ListDirectoryTool",
    "MkdirTool",
    "MoveFileTool",
    "OpenFileTool",
    "ReadTool",
    "RunPythonTool",
    "RunTestsTool",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "WebFetchTool",
    "WebSearchTool",
    "WriteTool",
]
