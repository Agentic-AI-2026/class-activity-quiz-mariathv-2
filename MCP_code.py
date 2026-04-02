"""
Helper utilities for MCP tool loading.

This file is intentionally *not* tied to any specific LLM provider.
It exists to provide a valid, importable MCP tools helper (no Ollama).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
from langchain_mcp_adapters.client import MultiServerMCPClient


def _start_weather_server(python_exe: str, weather_script: Path) -> subprocess.Popen[str]:
    # Starts the weather MCP server locally over HTTP (FastMCP default port is commonly 8000).
    return subprocess.Popen(
        [python_exe, str(weather_script)],
        cwd=str(weather_script.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_weather(url: str, timeout_s: int = 10) -> None:
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            requests.get(url, timeout=2)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"Weather MCP server did not become ready. Last error: {last_err!r}")


async def create_tools_map(
    *,
    python_exe: str,
    project_root: Path,
    start_weather: bool = True,
) -> Tuple[Dict[str, Any], Optional[subprocess.Popen[str]]]:
    """
    Returns (tools_map, weather_process_handle).
    The caller is responsible for terminating the weather process (if started).
    """

    math_script = project_root / "Tools" / "math_server.py"
    search_script = project_root / "Tools" / "search_server.py"
    weather_script = project_root / "Tools" / "weather_server.py"

    weather_proc: Optional[subprocess.Popen[str]] = None
    if start_weather:
        weather_proc = _start_weather_server(python_exe, weather_script)
        _wait_for_weather("http://localhost:8000/mcp", timeout_s=12)

    mcp = MultiServerMCPClient(
        {
            "math": {"command": python_exe, "args": [str(math_script)], "transport": "stdio"},
            "search": {"command": python_exe, "args": [str(search_script)], "transport": "stdio"},
            "weather": {"url": "http://localhost:8000/mcp", "transport": "streamable_http"},
        }
    )

    tools_map: Dict[str, Any] = {}
    for server_name in ["math", "search", "weather"]:
        try:
            tools = await mcp.get_tools(server_name=server_name)
            for t in tools:
                tools_map[t.name] = t
        except Exception:
            continue

    return tools_map, weather_proc