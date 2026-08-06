"""
Virtual AI Health Assistant — Agent Brain
==========================================
A single-agent ReAct-style loop built with LangGraph.

Flow: extract symptoms -> check red flags -> ask follow-up (if info missing)
      -> generate triage summary (once enough info is gathered or an
      emergency is detected).

Run standalone for a CLI demo:
    python health_assistant_brain.py
"""

import os
from typing import List, Optional, TypedDict

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

GEMMA_MODEL = "gemma-4-26b-a4b-it"


# ==========================================
# 1. STATE & DATA MODELS
# ==========================================

class HealthState(TypedDict):
    messages: List[dict]         # Chat history: [{"role": ..., "content": ...}]
    symptoms: List[str]          # Extracted symptoms
    duration: Optional[str]      # e.g. "3 days"
    severity: Optional[int]      # 1-10
    is_emergency: bool           # Red-flag state
    triage_level: Optional[str]  # Final urgency category
    turn_count: int              # Guards against infinite follow-up loops


class ExtractedSymptoms(BaseModel):
    symptoms: List[str] = Field(description="Physical or mental symptoms mentioned.")
    duration: Optional[str] = Field(default=None, description="How long symptoms have lasted, e.g. '2 days'.")
    severity: Optional[int] = Field(default=None, description="Severity rating 1-10, if mentioned.")


# ==========================================
# 2. TOOLS
# ==========================================

CRITICAL_FLAGS = {
    "chest pain", "shortness of breath", "slurred speech",
    "sudden numbness", "coughing blood", "stiff neck with fever",
    "difficulty breathing", "loss of consciousness",
}


@tool
def evaluate_red_flags(symptoms: List[str]) -> bool:
    """Checks input symptoms against critical emergency red flags."""
    return any(
        flag in symptom.lower()
        for symptom in symptoms
        for flag in CRITICAL_FLAGS
    )


def _get_llm() -> ChatGoogleGenerativeAI:
    """Lazily builds the LLM client so the module can be imported without a key set."""
    return ChatGoogleGenerativeAI(model=GEMMA_MODEL, temperature=0.2)


# ==========================================
# 3. AGENT NODES
# ==========================================

def symptom_extractor_node(state: HealthState) -> HealthState:
    """Analyzes the latest user message and merges it into the structured state."""
    user_msgs = [m["content"] for m in state["messages"] if m["role"] == "user"]
    latest_user_input = user_msgs[-1] if user_msgs else ""

    llm = _get_llm()
    extractor_llm = llm.with_structured_output(ExtractedSymptoms)

    prompt = f"""
    Extract all medical symptoms, duration, and severity mentioned in this input:
    "{latest_user_input}"

    Existing tracked symptoms: {state['symptoms']}
    Existing duration: {state['duration']}
    Existing severity: {state['severity']}
    """
    result: ExtractedSymptoms = extractor_llm.invoke(prompt)

    updated_symptoms = list(dict.fromkeys(state["symptoms"] + result.symptoms))  # dedupe, keep order
    updated_duration = result.duration or state["duration"]
    updated_severity = result.severity or state["severity"]

    is_emergency = evaluate_red_flags.invoke({"symptoms": updated_symptoms})

    return {
        **state,
        "symptoms": updated_symptoms,
        "duration": updated_duration,
        "severity": updated_severity,
        "is_emergency": is_emergency,
        "turn_count": state["turn_count"] + 1,
    }


def follow_up_node(state: HealthState) -> HealthState:
    """Asks 1-2 targeted follow-up questions to close missing information gaps."""
    missing_items = []
    if not state["duration"]:
        missing_items.append("how long symptoms have been present")
    if not state["severity"]:
        missing_items.append("severity rating on a scale from 1 to 10")

    llm = _get_llm()
    prompt = f"""
    You are an empathetic Clinical Assistant.
    Symptoms identified so far: {', '.join(state['symptoms']) or 'none yet'}.
    Missing information: {', '.join(missing_items)}.

    Ask 1 or 2 concise, polite follow-up questions to gather this missing detail.
    Do NOT offer diagnosis or medical advice yet.
    """
    response = llm.invoke(prompt)

    new_messages = state["messages"] + [{"role": "assistant", "content": response.content}]
    return {**state, "messages": new_messages}


def triage_summary_node(state: HealthState) -> HealthState:
    """Generates the final case summary, triage level, and next-step recommendations."""
    if state["is_emergency"]:
        triage_level = "EMERGENCY (Call 911 / Go to ER immediately)"
    elif state["severity"] and state["severity"] >= 7:
        triage_level = "Urgent Care (Visit within 24 hours)"
    elif state["duration"] and any(ch.isdigit() for ch in state["duration"]):
        triage_level = "Primary Care (Routine)"
    else:
        triage_level = "Self-Care / Telehealth"

    llm = _get_llm()
    prompt = f"""
    You are a Virtual Health Assistant. Provide a structured Case Summary and Triage Assessment.

    PATIENT PROFILE:
    - Reported Symptoms: {', '.join(state['symptoms']) or 'None reported'}
    - Duration: {state['duration'] or 'Not specified'}
    - Severity (1-10): {state['severity'] or 'Not specified'}
    - Triage Urgency Category: {triage_level}

    INSTRUCTIONS:
    1. Summarize the intake clearly, in plain language.
    2. State the recommended Triage Urgency Level.
    3. Suggest 2-3 practical next steps.
    4. End with this exact disclaimer on its own line:
       "I am an AI, not a doctor. Seek immediate emergency services for severe or escalating symptoms."
    """
    response = llm.invoke(prompt)

    new_messages = state["messages"] + [{"role": "assistant", "content": response.content}]
    return {**state, "triage_level": triage_level, "messages": new_messages}


# ==========================================
# 4. CONDITIONAL ROUTING
# ==========================================

def decide_next_step(state: HealthState) -> str:
    """Decides whether to escalate, ask a follow-up, or wrap up with a triage summary."""
    if state["is_emergency"]:
        return "generate_triage"
    if (not state["duration"] or not state["severity"]) and state["turn_count"] < 3:
        return "ask_followup"
    return "generate_triage"


# ==========================================
# 5. BUILD & COMPILE THE GRAPH
# ==========================================

builder = StateGraph(HealthState)
builder.add_node("symptom_extractor", symptom_extractor_node)
builder.add_node("ask_followup", follow_up_node)
builder.add_node("generate_triage", triage_summary_node)

builder.set_entry_point("symptom_extractor")
builder.add_conditional_edges(
    "symptom_extractor",
    decide_next_step,
    {"ask_followup": "ask_followup", "generate_triage": "generate_triage"},
)
builder.add_edge("ask_followup", END)
builder.add_edge("generate_triage", END)

health_agent = builder.compile()


def new_session_state() -> HealthState:
    """Returns a fresh, empty session state."""
    return {
        "messages": [],
        "symptoms": [],
        "duration": None,
        "severity": None,
        "is_emergency": False,
        "triage_level": None,
        "turn_count": 0,
    }


# ==========================================
# 6. INTERACTIVE CLI RUNNER
# ==========================================

if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY is not set. Set it before running, e.g.:")
        print('  export GOOGLE_API_KEY="your-api-key-here"\n')

    print("=== Virtual AI Health Assistant Engine Initialized ===")
    state = new_session_state()

    while True:
        user_input = input("\nPatient Input (or 'exit'): ")
        if user_input.lower() in ("exit", "quit"):
            break

        state["messages"].append({"role": "user", "content": user_input})
        state = health_agent.invoke(state)

        last_response = state["messages"][-1]["content"]
        print(f"\nAI Assistant:\n{last_response}")

        if state["triage_level"]:
            print("\n--- Evaluation Completed ---")
            break
