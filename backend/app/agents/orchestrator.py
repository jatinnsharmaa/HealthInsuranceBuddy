"""
Orchestrator/Evaluator Agent.

Tools:
  1. retrieve_from_policy(query) — deterministic Python function, no LLM
  2. search_web(intent, start_url, goal) — delegates to Web Navigator Agent (LLM)

The orchestrator independently evaluates web results and generates the final answer.
"""
import re
import threading
from typing import Any, AsyncGenerator
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import AgentWorkflow
from llama_index.llms.anthropic import Anthropic

from app.prompts.orchestrator_prompt import ORCHESTRATOR_SYSTEM_PROMPT
from app.agents.web_navigator import build_web_navigator, run_web_navigator
from app.retrieval.retriever import retrieve
from app import timing


class OrchestratorAgent:
    def __init__(
        self,
        policy_id: str,
        pinecone_index: Any,
        cohere_api_key: str,
        anthropic_api_key: str,
        data_dir: str,
        retrieval_mode: str = "hybrid_rerank",
        top_k: int = 5,
        candidate_k: int = 20,
        model: str = "claude-haiku-4-5-20251001",
    ):
        self.retrieval_log: list[dict] = []
        self.web_fetch_status: str = "not_triggered"
        self._anthropic_api_key = anthropic_api_key
        self._model = model

        self._web_navigator = build_web_navigator(anthropic_api_key=anthropic_api_key)

        self._workflow = self._build_workflow(
            policy_id=policy_id,
            pinecone_index=pinecone_index,
            cohere_api_key=cohere_api_key,
            data_dir=data_dir,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            candidate_k=candidate_k,
        )

    def _build_workflow(
        self,
        policy_id: str,
        pinecone_index: Any,
        cohere_api_key: str,
        data_dir: str,
        retrieval_mode: str,
        top_k: int,
        candidate_k: int,
    ) -> AgentWorkflow:
        log = self.retrieval_log
        log_lock = threading.Lock()  # makes len(log)+append atomic for parallel tool calls
        state = {"web_fetch_status": "not_triggered", "web_calls": 0}
        web_navigator = self._web_navigator

        def retrieve_from_policy(query: str) -> str:
            """
            Retrieve policy clauses using hybrid search. Returns top 5 chunks with
            clause reference, page number, section, and type. URL_DEFERRAL tags flag
            clauses that defer to live websites. Call this ONCE with the verbatim user question.
            """
            timing.t(f"ORCHESTRATOR: retrieve_from_policy START — {query[:60]}")
            with log_lock:  # atomic: capture index + append (prevents parallel-call race)
                entry_idx = len(log)
                log.append({"tool": "retrieve_from_policy", "query": query})

            try:
                nodes = retrieve(
                    query=query,
                    policy_id=policy_id,
                    pinecone_index=pinecone_index,
                    cohere_api_key=cohere_api_key,
                    data_dir=data_dir,
                    mode=retrieval_mode,
                    top_k=top_k,
                    candidate_k=candidate_k,
                )

                timing.t(f"ORCHESTRATOR: retrieve_from_policy DONE — {len(nodes) if nodes else 0} nodes")

                if not nodes:
                    log[entry_idx]["nodes_returned"] = 0
                    log[entry_idx]["result_full"] = ""
                    return "No relevant clauses found for this query."

                parts = []
                for nws in nodes:
                    meta = nws.node.metadata
                    text = nws.node.get_content()
                    urls = re.findall(r"https?://\S+|www\.\S+", text)
                    url_flag = f"\nURL_DEFERRAL: {urls[0]}" if urls else ""
                    parts.append(
                        f"Clause: {meta.get('sub_clause', 'Unknown')}\n"
                        f"Section: {meta.get('section', '')}\n"
                        f"Page: {meta.get('page_number', '?')}\n"
                        f"Type: {meta.get('chunk_type', 'clause')}\n"
                        f"Text: {text}{url_flag}"
                    )

                result_text = "\n---\n".join(parts)
                log[entry_idx]["nodes_returned"] = len(nodes)
                log[entry_idx]["result_chunks"] = parts   # list form — durable for eval scoring
                log[entry_idx]["result_full"] = result_text  # joined form — sent to LLM
                return result_text

            except Exception as e:
                # Always set result_full even on failure so eval context extraction works
                log[entry_idx]["nodes_returned"] = 0
                log[entry_idx]["result_full"] = ""
                log[entry_idx]["error"] = str(e)[:200]
                timing.t(f"ORCHESTRATOR: retrieve_from_policy ERROR — {str(e)[:80]}")
                raise

        async def search_web(intent: str, start_url: str, goal: str) -> str:
            """
            Delegate web navigation to the Web Navigator Agent.
            The Navigator navigates start_url to find content matching goal.
            It returns raw content or a full failure trace — you evaluate the results.
            Maximum 2 calls total. On a second call, include the failure reason in intent.
            """
            if state["web_calls"] >= 2:
                timing.t("ORCHESTRATOR: search_web BLOCKED — budget exhausted (2/2)")
                return "WEB_BUDGET_EXHAUSTED: Already called search_web twice. Generate answer from evidence collected so far."

            state["web_calls"] += 1
            call_num = state["web_calls"]
            timing.t(f"ORCHESTRATOR: search_web call {call_num}/2 START — {start_url[:60]}")
            log.append({"tool": "search_web", "call_number": call_num, "intent": intent, "start_url": start_url, "goal": goal})
            state["web_fetch_status"] = "triggered"

            result = await run_web_navigator(web_navigator, start_url, goal)

            if '"status": "failed"' in result or "status: failed" in result.lower():
                state["web_fetch_status"] = "failed"
            else:
                state["web_fetch_status"] = "succeeded"

            timing.t(f"ORCHESTRATOR: search_web call {call_num}/2 DONE — {state['web_fetch_status']}")
            log[-1]["web_fetch_status"] = state["web_fetch_status"]
            log[-1]["result_preview"] = result[:200]
            return result

        self._state = state

        llm = Anthropic(
            model=self._model,
            api_key=self._anthropic_api_key,
            max_tokens=4096,
            cache_idx=-1,  # cache system prompt + all messages — 90% off repeated input tokens
        )

        return AgentWorkflow.from_tools_or_functions(
            tools_or_functions=[
                FunctionTool.from_defaults(fn=retrieve_from_policy, name="retrieve_from_policy"),
                FunctionTool.from_defaults(async_fn=search_web, name="search_web"),
            ],
            llm=llm,
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            verbose=True,
        )

    async def stream_chat(
        self,
        message: str,
        conversation_history: list[dict] | None = None,
    ) -> AsyncGenerator[dict, None]:
        self.retrieval_log.clear()
        self._state["web_fetch_status"] = "not_triggered"
        self._state["web_calls"] = 0

        full_message = message
        if conversation_history:
            lines = []
            for turn in conversation_history[-6:]:
                lines.append(f"{turn.get('role','user').upper()}: {turn.get('content','')}")
            if lines:
                full_message = "Previous conversation:\n" + "\n".join(lines) + f"\n\nCurrent question: {message}"

        timing.reset()
        timing.t("ORCHESTRATOR START")
        handler = self._workflow.run(user_msg=full_message)

        full_response = ""
        input_tokens = 0
        output_tokens = 0
        try:
            async for event in handler.stream_events():
                event_type = type(event).__name__

                if hasattr(event, "tool_name") and not hasattr(event, "tool_output"):
                    tool_name = getattr(event, "tool_name", "")
                    step_msg = _tool_step_label(tool_name)
                    if step_msg:
                        yield {"type": "agent_step", "content": step_msg}

                elif hasattr(event, "delta"):
                    delta = getattr(event, "delta", "")
                    if delta:
                        full_response += delta
                        yield {"type": "chunk", "content": delta}

                # Capture token usage from AgentOutput
                elif event_type == "AgentOutput" or (hasattr(event, "response") and hasattr(getattr(event, "response", None), "additional_kwargs")):
                    resp = getattr(event, "response", None)
                    if resp:
                        usage = getattr(resp, "additional_kwargs", {}).get("usage", {})
                        input_tokens += usage.get("input_tokens", 0)
                        output_tokens += usage.get("output_tokens", 0)

        except Exception as e:
            yield {"type": "error", "content": str(e)}
            return

        if not full_response:
            try:
                response = await handler
                full_response = str(response)
                # Try to get token usage from final response
                usage = getattr(response, "additional_kwargs", {}).get("usage", {}) if hasattr(response, "additional_kwargs") else {}
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
                yield {"type": "chunk", "content": full_response}
            except Exception as e:
                yield {"type": "error", "content": str(e)}
                return

        citations = []
        for m in re.finditer(
            r"\[(?:Clause|Section|Annexure)\s*([\d.IVXivx]+),?\s*Page\s*(\d+)\]",
            full_response,
            re.IGNORECASE,
        ):
            citations.append({"clause": f"Clause {m.group(1)}", "page": int(m.group(2))})

        timing.t(f"ORCHESTRATOR DONE — tokens in={input_tokens} out={output_tokens}")
        yield {
            "type": "done",
            "citations": citations,
            "retrieval_log": self.retrieval_log,
            "web_fetch_status": self._state["web_fetch_status"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


def _tool_step_label(tool_name: str) -> str:
    return {
        "retrieve_from_policy": "Searching policy document...",
        "search_web": "Navigating live website...",
        "web_navigate": "Crawling website...",
    }.get(tool_name, "")
