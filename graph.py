"""
LangGraph Planner–Executor implementation.

This file contains the state definition and the graph nodes:
START -> planner_node -> executor_node -> (loop) -> END
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TypedDict, Union, cast

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage


class Step(TypedDict):
    step: int
    description: str
    tool: Optional[str]
    args: Optional[Dict[str, Any]]


class ExecutedResult(TypedDict):
    step: int
    description: str
    result: str


class AgentState(TypedDict, total=False):
    # User input
    goal: str

    # Planner output
    plan: List[Step]

    # Execution cursor
    current_step: int

    # Executor output
    results: List[ExecutedResult]


PLAN_SYSTEM = """You are a planner for a tool-using agent.

Break the user's goal into an ordered JSON array of steps that will be executed sequentially.

Each step MUST follow this EXACT schema:
  {"step": int, "description": str, "tool": str or null, "args": object or null}

Allowed tool names and their EXACT argument names:
  - calculator(expression: str)             # safe calculator
  - search_web(query: str)                # web search (short factual snippets)
  - search_news(query: str)              # news search (latest articles)
  - get_current_weather(city: str)       # current weather
  - get_weather_forecast(city: str, days: int)  # forecast for next N days (1-7)

Rules:
  - Use tool ONLY when executing a factual/tool-based step.
  - For synthesis/writing steps, set "tool": null and "args": null.
  - Return steps as a JSON array only. No markdown. No extra keys. No commentary.
  - Step numbers must be increasing starting at 1.
"""


SYNTHESIS_TEMPLATE = """You are executing a single step of a larger plan.

Step description:
{description}

Previous step results (may be empty):
{context}

Task:
Produce the result for this step as plain text.
If a previous step provides relevant details, use it.
Do not mention JSON or planning.
"""


TOOL_ARG_MAP: Dict[str, Union[str, List[str]]] = {
    "calculator": "expression",
    "search_web": "query",
    "search_news": "query",
    "get_current_weather": "city",
    "get_weather_forecast": ["city", "days"],
}


def _strip_json_wrappers(text: str) -> str:
    # Remove common markdown fences if the model used them anyway.
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    return text.strip()


def _parse_plan_json(text: str) -> List[Step]:
    cleaned = _strip_json_wrappers(text)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON array in planner output: {text!r}")

    arr_text = cleaned[start : end + 1]
    data = json.loads(arr_text)
    if not isinstance(data, list):
        raise ValueError("Planner output JSON must be a list")

    # Normalize shape a bit (but do not hardcode content).
    normalized: List[Step] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        tool = cast(Optional[str], item.get("tool"))
        args = item.get("args")
        normalized.append(
            Step(
                step=int(item.get("step") or i + 1),
                description=str(item.get("description") or ""),
                tool=tool if tool is not None else None,
                args=cast(Optional[Dict[str, Any]], args) if isinstance(args, dict) else None,
            )
        )
    return normalized


def _safe_args(tool_name: str, raw_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not raw_args:
        # Tool args are missing; return empty dict and let the tool error (preferred to guessing).
        return {}

    expected = TOOL_ARG_MAP.get(tool_name)
    if expected is None:
        return raw_args

    if isinstance(expected, str):
        if expected in raw_args:
            return raw_args
        # Remap hallucinated arg name -> expected parameter
        first_val = next(iter(raw_args.values()), None)
        if first_val is None:
            return {}
        return {expected: first_val}

    # expected is ["city", "days"]
    city = raw_args.get("city")
    days = raw_args.get("days")
    if city is None:
        # Best-effort remap from the first provided value
        first_val = next(iter(raw_args.values()), None)
        city = first_val
    if days is None:
        days = 3
    return {"city": str(city), "days": int(days)}


def build_planner_executor_graph(llm: Any, tools_map: Dict[str, Any]) -> Any:
    """
    Build a compiled LangGraph app that:
    - plans once
    - executes exactly one step per executor invocation
    - loops until all steps are completed
    """

    async def planner_node(state: AgentState) -> AgentState:
        goal = cast(str, state["goal"])

        resp = await llm.ainvoke([SystemMessage(content=PLAN_SYSTEM), HumanMessage(content=goal)])
        content = resp.content if hasattr(resp, "content") else str(resp)
        plan = _parse_plan_json(str(content))

        return {
            "plan": plan,
            "current_step": 0,
            "results": [],
        }

    async def executor_node(state: AgentState) -> AgentState:
        plan = cast(List[Step], state["plan"])
        idx = cast(int, state["current_step"])
        step = plan[idx]

        tool_name = step.get("tool")
        raw_args = step.get("args")
        result_text: str

        if tool_name and tool_name in tools_map:
            args = _safe_args(tool_name, raw_args)
            # Tools from MCP adapters expose ainvoke for async execution.
            tool = tools_map[tool_name]
            result = await tool.ainvoke(args)
            result_text = str(result).strip()
        else:
            # Synthesis step: use the LLM to produce the step result.
            prior_context = "\n".join(
                [f"Step {r['step']}: {r['result']}" for r in cast(List[ExecutedResult], state.get("results", []))]
            )
            synth_prompt = SYNTHESIS_TEMPLATE.format(
                description=step.get("description", ""),
                context=prior_context if prior_context else "(none)",
            )
            resp = await llm.ainvoke([HumanMessage(content=synth_prompt)])
            result_text = (resp.content if hasattr(resp, "content") else str(resp)).strip()

        prior_results = cast(List[ExecutedResult], state.get("results", []))
        new_results: List[ExecutedResult] = prior_results + [
            {
                "step": cast(int, step["step"]),
                "description": cast(str, step["description"]),
                "result": result_text,
            }
        ]

        return {
            "current_step": idx + 1,
            "results": new_results,
        }

    def route_after_executor(state: AgentState) -> str:
        plan = cast(List[Step], state.get("plan", []))
        idx = cast(int, state.get("current_step", 0))
        if idx >= len(plan):
            return "end"
        return "continue"

    graph = StateGraph(AgentState)
    graph.add_node("planner_node", planner_node)
    graph.add_node("executor_node", executor_node)

    graph.set_entry_point("planner_node")
    graph.add_edge("planner_node", "executor_node")
    graph.add_conditional_edges(
        "executor_node",
        route_after_executor,
        {"continue": "executor_node", "end": END},
    )

    return graph.compile()

