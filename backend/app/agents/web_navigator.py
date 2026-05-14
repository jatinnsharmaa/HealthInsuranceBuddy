"""
Web Navigator Agent: navigates websites and returns raw content.
Does not evaluate results. Does not generate answers.
"""
from typing import Any
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import AgentWorkflow
from llama_index.llms.anthropic import Anthropic

from app.prompts.web_navigator_prompt import WEB_NAVIGATOR_SYSTEM_PROMPT
from app.agents.web_tools import web_navigate_impl
from app import timing


def build_web_navigator(anthropic_api_key: str) -> AgentWorkflow:
    import json

    def web_navigate(start_url: str, goal: str) -> str:
        """
        Navigate a website to find content matching the goal.
        Tries start_url first, then crawls links, then falls back to root domain.
        Returns content or full failure trace.
        """
        timing.t(f"WEB_NAVIGATOR: web_navigate START — {start_url[:60]}")
        result = web_navigate_impl(start_url, goal)
        timing.t(f"WEB_NAVIGATOR: web_navigate DONE — status: {result['status']}, steps: {len(result.get('trace', []))}")
        return json.dumps(result)

    llm = Anthropic(
        model="claude-sonnet-4-6",
        api_key=anthropic_api_key,
        max_tokens=1024,
        cache_idx=-1,  # cache system prompt for repeated navigation tasks
    )

    return AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[
            FunctionTool.from_defaults(fn=web_navigate, name="web_navigate"),
        ],
        llm=llm,
        system_prompt=WEB_NAVIGATOR_SYSTEM_PROMPT,
        verbose=False,
    )


async def run_web_navigator(agent: AgentWorkflow, start_url: str, goal: str) -> str:
    timing.t(f"WEB_NAVIGATOR LLM START — goal: {goal[:60]}")
    handler = agent.run(user_msg=f"Navigate to {start_url} to find: {goal}")
    response = await handler
    timing.t("WEB_NAVIGATOR LLM DONE")
    return str(response)
