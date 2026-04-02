from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict

from colorama import Fore, Style, init
from langchain_groq import ChatGroq

from graph import build_planner_executor_graph
from MCP_code import create_tools_map

init(autoreset=True)

SAMPLE_GOALS = {
    "1": "Plan an outdoor event for 150 people: calculate tables/chairs, find average ticket price, check weather, and summarize.",
    "2": "Calculate 250 * 18, then search for average catering cost per person, and summarize a rough event budget.",
    "3": "Fetch Q3 sales data and summarize it.",
}


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Missing environment variable {name!r}. Set it before running (e.g. in PowerShell: "
            f"`$env:{name} = '...')."
        )
    return val


def _project_root() -> Path:
    return Path(__file__).resolve().parent


async def run(goal: str) -> Dict[str, Any]:
    python_exe = sys.executable
    project_root = _project_root()

    # LLM (Groq only; no Ollama).
    _require_env("GROQ_API_KEY")
    # Use a currently supported Groq default model. Can still be overridden via GROQ_MODEL.
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm = ChatGroq(model=model, temperature=0)

    tools_map, weather_proc = await create_tools_map(
        python_exe=python_exe, project_root=project_root, start_weather=True
    )
    if not tools_map:
        raise RuntimeError("No MCP tools loaded. Check that MCP servers are reachable.")

    app = build_planner_executor_graph(llm=llm, tools_map=tools_map)
    try:
        final_state = await app.ainvoke({"goal": goal})
        return final_state
    finally:
        if weather_proc is not None:
            weather_proc.terminate()


def _print_banner() -> None:
    print(Fore.CYAN + Style.BRIGHT + "\nLangGraph Planner-Executor (Groq)")
    print(Fore.CYAN + "-" * 40)


def _print_results(final_state: Dict[str, Any]) -> None:
    plan = final_state.get("plan", [])
    results = final_state.get("results", [])

    print(Fore.YELLOW + Style.BRIGHT + "\n=== PLAN ===")
    for s in plan:
        print(
            Fore.YELLOW
            + f"Step {s.get('step')}: {s.get('description')} | tool={s.get('tool')}"
        )

    print(Fore.GREEN + Style.BRIGHT + "\n=== RESULTS ===")
    for r in results:
        print(Fore.GREEN + f"[Step {r['step']}] {r['description']}")
        print(Fore.WHITE + f"{r['result']}\n")


def _prompt_menu_choice() -> str:
    print(Fore.MAGENTA + Style.BRIGHT + "\nMenu")
    print(Fore.MAGENTA + "1) Run sample goal: outdoor event planning")
    print(Fore.MAGENTA + "2) Run sample goal: budget estimation")
    print(Fore.MAGENTA + "3) Run sample goal: Q3 sales summary")
    print(Fore.MAGENTA + "4) Enter custom goal")
    print(Fore.MAGENTA + "5) Exit")
    return input(Fore.CYAN + "\nSelect option (1-5): ").strip()


def _menu_loop() -> None:
    _print_banner()
    while True:
        choice = _prompt_menu_choice()

        if choice == "5":
            print(Fore.CYAN + "Exiting. Goodbye.")
            return

        if choice in SAMPLE_GOALS:
            goal = SAMPLE_GOALS[choice]
        elif choice == "4":
            goal = input(Fore.CYAN + "Enter your goal: ").strip()
            if not goal:
                print(Fore.RED + "Goal cannot be empty.")
                continue
        else:
            print(Fore.RED + "Invalid choice. Please pick 1-5.")
            continue

        print(Fore.BLUE + Style.BRIGHT + f"\nRunning goal:\n{goal}\n")
        try:
            final_state = asyncio.run(run(goal))
            _print_results(final_state)
        except Exception as exc:  # noqa: BLE001
            print(Fore.RED + Style.BRIGHT + f"\nError: {exc}")
            print(
                Fore.RED
                + "Tip: ensure GROQ_API_KEY (and optionally TAVILY_API_KEY) are set."
            )


if __name__ == "__main__":
    # Keep direct CLI argument mode for quick one-off execution:
    #   python main.py "your goal"
    if len(sys.argv) > 1:
        try:
            _print_banner()
            goal = " ".join(sys.argv[1:])
            print(Fore.BLUE + Style.BRIGHT + f"\nRunning goal:\n{goal}\n")
            state = asyncio.run(run(goal))
            _print_results(state)
        except Exception as exc:  # noqa: BLE001
            print(Fore.RED + Style.BRIGHT + f"\nError: {exc}")
            sys.exit(1)
    else:
        _menu_loop()
