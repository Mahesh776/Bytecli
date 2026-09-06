# ruff: noqa: RUF012
# mypy: disable-error-code="override"
import json
import re
from pathlib import Path

import httpx

from bytecli.tools.base import Tool, ToolParameter, ToolResult


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch content from a URL. Returns the content as text."
    name_aliases = {"fetch", "fetch_url", "get_url", "http_get", "download_url", "read_url", "fetch_web"}
    parameters = [
        ToolParameter(name="url", type="string", description="The URL to fetch content from"),
        ToolParameter(name="timeout", type="integer", description="Timeout in seconds", required=False, default=30),
    ]

    async def _execute(self, url: str, timeout: int = 30) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    data = response.json()
                    return ToolResult(
                        success=True,
                        data={"url": url, "content": json.dumps(data, indent=2), "content_type": content_type},
                        tool_name=self.name,
                    )
                return ToolResult(
                    success=True,
                    data={"url": url, "content": response.text, "content_type": content_type},
                    tool_name=self.name,
                )
        except httpx.TimeoutException:
            return ToolResult(success=False, error=f"Request to {url} timed out after {timeout}s", tool_name=self.name)
        except httpx.HTTPStatusError as e:
            err = e.response.text[:500]
            return ToolResult(success=False, error=f"HTTP {e.response.status_code}: {err}", tool_name=self.name)
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to fetch {url}: {e}", tool_name=self.name)


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for information. Returns top results with snippets."
    name_aliases = {"search_web", "websearch", "internet_search", "search_internet", "search_online", "web_query"}
    parameters = [
        ToolParameter(name="query", type="string", description="The search query"),
        ToolParameter(name="num_results", type="integer", description="Number of results to return", required=False, default=8),  # noqa: E501
    ]

    async def _execute(self, query: str, num_results: int = 8) -> ToolResult:
        search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(
                    search_url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; ByteCli/0.1)"},
                )
                response.raise_for_status()
            html = response.text
            results: list[dict[str, str]] = []
            for match in re.finditer(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                html,
            ):
                if len(results) >= num_results:
                    break
                url = match.group(1)
                title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
                snippet_match = re.search(
                    r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                    html[match.end():],
                )
                snippet = ""
                if snippet_match:
                    snippet = re.sub(r"<[^>]+>", "", snippet_match.group(1)).strip()
                results.append({"title": title, "url": url, "snippet": snippet})
            return ToolResult(
                success=True,
                data={"query": query, "results": results, "count": len(results)},
                tool_name=self.name,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Search failed: {e}", tool_name=self.name)


class DownloadTool(Tool):
    name = "download"
    description = "Download content from a URL and save it to a local file. Returns the saved path."
    name_aliases = {"download_file", "download_url", "save_url", "save_url_to_file", "fetch_to_file"}
    arg_aliases = {
        "url": "url",
        "path": "file_path",
        "file": "file_path",
        "filepath": "file_path",
        "output": "file_path",
        "save_to": "file_path",
        "destination": "file_path",
        "dest": "file_path",
    }
    parameters = [
        ToolParameter(name="url", type="string", description="The URL to download"),
        ToolParameter(name="file_path", type="string", description="Absolute path to save the content to"),
        ToolParameter(name="timeout", type="integer", description="Timeout in seconds", required=False, default=60),
    ]

    async def _execute(self, url: str, file_path: str, timeout: int = 60) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                content = response.content
        except httpx.TimeoutException:
            return ToolResult(success=False, error=f"Download of {url} timed out after {timeout}s", tool_name=self.name)
        except httpx.HTTPStatusError as e:
            err = e.response.text[:500]
            return ToolResult(success=False, error=f"HTTP {e.response.status_code}: {err}", tool_name=self.name)
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to download {url}: {e}", tool_name=self.name)
        target = Path(file_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to save file: {e}", tool_name=self.name)
        return ToolResult(
            success=True,
            data={"url": url, "file_path": str(target), "bytes": len(content)},
            tool_name=self.name,
        )
