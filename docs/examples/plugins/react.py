# pytest: ollama, e2e, qualitative
#
# ReACT agent with tool allow-list enforcement via plugins.
#
# This interactive demo shows how a TOOL_PRE_INVOKE plugin can act as a
# security boundary for a ReACT agent loop — permitting only approved tools
# and blocking everything else before invocation.
#
# The agent has access to three tools:
#   - lookup_employee   (ALLOWED)
#   - check_pto_balance (ALLOWED)
#   - delete_employee   (BLOCKED by allow-list policy)
#
# Run:
#   uv run python docs/examples/plugins/react.py
#
# Try these inputs:
#   Happy path  → "How many PTO days does Alice have left?"
#   Blocked     → "Delete the employee record for Bob"

import json

import pydantic

from mellea import start_session
from mellea.backends import ModelOption, tool
from mellea.plugins import (
    HookType,
    PluginMode,
    PluginSet,
    PluginViolationError,
    block,
    hook,
)
from mellea.stdlib.functional import _call_tools

# ---------------------------------------------------------------------------
# Terminal formatting helpers
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"


def _header(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}\n")


def _step(label: str, content: str) -> None:
    print(f"  {BOLD}{MAGENTA}[{label}]{RESET} {content}")


def _ok(text: str) -> None:
    print(f"  {GREEN}✓ {text}{RESET}")


def _blocked(text: str) -> None:
    print(f"  {RED}✗ {text}{RESET}")


def _info(text: str) -> None:
    print(f"  {DIM}{text}{RESET}")


def _tool_output(name: str, result: str) -> None:
    print(f"  {YELLOW}⚙ {name}{RESET} → {result}")


# ---------------------------------------------------------------------------
# Mock tools — an HR system
# ---------------------------------------------------------------------------


@tool
def lookup_employee(name: str) -> str:
    """Look up basic information about an employee by name.

    Args:
        name: The employee's first name.
    """
    employees = {
        "alice": {"name": "Alice Chen", "department": "Engineering", "id": "E-1042"},
        "bob": {"name": "Bob Martinez", "department": "Sales", "id": "E-2091"},
        "carol": {"name": "Carol Park", "department": "Design", "id": "E-3017"},
    }
    key = name.strip().lower()
    if key in employees:
        return json.dumps(employees[key])
    return json.dumps({"error": f"No employee named '{name}' found."})


@tool
def check_pto_balance(employee_id: str) -> str:
    """Check the remaining PTO (paid time off) balance for an employee.

    Args:
        employee_id: The employee ID (e.g. 'E-1042').
    """
    balances = {
        "E-1042": {
            "employee_id": "E-1042",
            "pto_remaining_days": 12,
            "pto_used_days": 8,
        },
        "E-2091": {
            "employee_id": "E-2091",
            "pto_remaining_days": 3,
            "pto_used_days": 17,
        },
        "E-3017": {
            "employee_id": "E-3017",
            "pto_remaining_days": 18,
            "pto_used_days": 2,
        },
    }
    if employee_id in balances:
        return json.dumps(balances[employee_id])
    return json.dumps({"error": f"No PTO record for employee '{employee_id}'."})


@tool
def delete_employee(employee_id: str) -> str:
    """Permanently delete an employee record from the HR system.

    Args:
        employee_id: The employee ID to delete.
    """
    return json.dumps({"status": "deleted", "employee_id": employee_id})


# ---------------------------------------------------------------------------
# Plugin — tool allow-list policy
# ---------------------------------------------------------------------------

ALLOWED_TOOLS: frozenset[str] = frozenset({"lookup_employee", "check_pto_balance"})


@hook(HookType.TOOL_PRE_INVOKE, mode=PluginMode.CONCURRENT, priority=5)
async def enforce_tool_allowlist(payload, _):
    """Block any tool not on the approved list."""
    if payload.is_control_flow:
        return
    tool_name = payload.model_tool_call.name
    if tool_name not in ALLOWED_TOOLS:
        _blocked(
            f"POLICY VIOLATION: tool '{tool_name}' is not in the allow list "
            f"{sorted(ALLOWED_TOOLS)}"
        )
        return block(
            f"Tool '{tool_name}' is not permitted by the security policy",
            code="TOOL_NOT_ALLOWED",
            details={"tool": tool_name, "allowed": sorted(ALLOWED_TOOLS)},
        )
    _ok(f"Policy check passed for tool '{tool_name}'")


@hook(HookType.TOOL_POST_INVOKE, mode=PluginMode.FIRE_AND_FORGET)
async def audit_tool_call(payload, _):
    """Log successful tool invocations."""
    if payload.success:
        _tool_output(payload.model_tool_call.name, str(payload.tool_output)[:120])


react_security = PluginSet("react-security", [enforce_tool_allowlist, audit_tool_call])


# ---------------------------------------------------------------------------
# ReACT loop (simplified, tool-calling style)
# ---------------------------------------------------------------------------

ALL_TOOLS = [lookup_employee, check_pto_balance, delete_employee]

MAX_TURNS = 5


class IsDone(pydantic.BaseModel):
    is_done: bool
    final_answer: str | None = None


def react_loop(goal: str) -> str | None:
    """Run a ReACT agent loop with tool allow-list enforcement."""
    _header(f"ReACT Agent — Goal: {goal}")

    _info(f"Allowed tools: {sorted(ALLOWED_TOOLS)}")
    _info(f"Available tools: {[t.name for t in ALL_TOOLS]}")
    _info(f"Max turns: {MAX_TURNS}")
    print()

    observations: list[str] = []

    with start_session(plugins=[react_security]) as m:
        for turn in range(1, MAX_TURNS + 1):
            print(f"  {BOLD}{BLUE}── Turn {turn} ──{RESET}")

            obs_summary = ""
            if observations:
                obs_summary = " Previous results: " + " | ".join(observations) + "."

            # Thought: ask the model what to do next
            thought = m.chat(
                f"You are an HR assistant answering: '{goal}'.{obs_summary} "
                f"What is the next tool you need to call? Think step by step. "
                f"Tools: lookup_employee(name), check_pto_balance(employee_id), "
                f"delete_employee(employee_id). Do not repeat a previous call."
            )
            _step("Thought", thought.content[:200])

            # Action: instruct the model to call a tool
            try:
                result = m.instruct(
                    description=(
                        f"Call the next tool to help answer: '{goal}'.{obs_summary}"
                    ),
                    model_options={ModelOption.TOOLS: ALL_TOOLS},
                    tool_calls=True,
                )

                tool_outputs = _call_tools(result, m.backend)

                if not tool_outputs:
                    _info("No tool was called this turn.")
                else:
                    for to in tool_outputs:
                        _step("Observation", to.content[:200])
                        observations.append(to.content[:300])

            except PluginViolationError as e:
                _blocked(f"Agent blocked: [{e.code}] {e.reason}")
                print()
                _header("Result: BLOCKED BY POLICY")
                print(
                    f"  {RED}The agent attempted to use a tool that violates the\n"
                    f"  security policy. The operation was prevented.{RESET}\n"
                )
                print(f"  {DIM}Plugin: {e.plugin_name}{RESET}")
                print(f"  {DIM}Hook: {e.hook_type}{RESET}")
                print(f"  {DIM}Code: {e.code}{RESET}")
                return None

            # Check if done
            obs_for_check = "; ".join(observations) if observations else "nothing yet"
            done_check = IsDone.model_validate_json(
                m.chat(
                    f"Gathered so far: {obs_for_check}. "
                    f"Can you fully answer: '{goal}'? "
                    f"If yes, set is_done=true and provide final_answer. Otherwise is_done=false.",
                    format=IsDone,
                ).content
            )

            if done_check.is_done and done_check.final_answer:
                print()
                _header("Result: SUCCESS")
                print(f"  {GREEN}{BOLD}{done_check.final_answer}{RESET}\n")
                return done_check.final_answer

        _info("Max turns reached without a final answer.")
        return None


# ---------------------------------------------------------------------------
# Interactive main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _header("ReACT Agent with Tool Allow-List Policy")
    print(f"  {GREEN}Allowed tools:{RESET}  lookup_employee, check_pto_balance")
    print(f"  {RED}Blocked tools:{RESET}  delete_employee\n")
    print(f"  {DIM}Try these queries:{RESET}")
    print(f'  {DIM}  • "How many PTO days does Alice have left?"  (happy path){RESET}')
    print(f'  {DIM}  • "Delete the employee record for Bob"       (blocked){RESET}')
    print()

    while True:
        try:
            query = input(f"  {BOLD}Enter your query{RESET} (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        react_loop(query)
        print()
