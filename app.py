"""
Virtual AI Health Assistant — Streamlit Frontend
=================================================
Chat UI on top of the LangGraph agent brain, with:
  - Custom CSS styling
  - A live "Clinical State" sidebar with symptom chips
  - A manual severity slider the patient can set/confirm directly
  - Color-coded urgency cards (Emergency / Urgent / Primary / Self-care)

Run:
    streamlit run app.py
"""

import streamlit as st
from health_assistant_brain import health_agent, new_session_state

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Virtual AI Health Assistant",
    page_icon="🩺",
    layout="centered",
)

# ------------------------------------------------------------------
# Custom styling
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f4f9fb 0%, #ffffff 40%);
        }
        .main-title {
            font-size: 2rem;
            font-weight: 700;
            color: #0f4c5c;
            margin-bottom: 0;
        }
        .subtitle {
            color: #5c7a85;
            font-size: 0.95rem;
            margin-top: 0.2rem;
        }
        .symptom-chip {
            display: inline-block;
            background: #e6f2f5;
            color: #0f4c5c;
            border: 1px solid #b6dbe3;
            border-radius: 999px;
            padding: 4px 12px;
            margin: 3px 4px 3px 0;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .stat-card {
            background: #ffffff;
            border: 1px solid #e5edf0;
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(15, 76, 92, 0.06);
        }
        .stat-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #7c97a0;
            margin-bottom: 2px;
        }
        .stat-value {
            font-size: 1.1rem;
            font-weight: 600;
            color: #0f4c5c;
        }
        .urgency-card {
            border-radius: 14px;
            padding: 18px 20px;
            margin: 14px 0;
            font-weight: 600;
            font-size: 1.05rem;
        }
        .urgency-emergency { background: #fdeaea; color: #a02323; border: 1px solid #f3b6b6; }
        .urgency-urgent    { background: #fff4e0; color: #97600b; border: 1px solid #f5d18c; }
        .urgency-primary   { background: #eaf3ff; color: #1b5fa8; border: 1px solid #b7d6f7; }
        .urgency-selfcare  { background: #eafaf0; color: #1f7a44; border: 1px solid #b7e8c9; }
        .disclaimer {
            font-size: 0.8rem;
            color: #8a8a8a;
            margin-top: 14px;
            border-top: 1px dashed #ddd;
            padding-top: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🩺 Virtual AI Health Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Describe your symptoms below. The assistant will ask clarifying '
    'questions if needed, then assess urgency and recommend next steps.</div>',
    unsafe_allow_html=True,
)
st.divider()

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "health_state" not in st.session_state:
    st.session_state.health_state = new_session_state()

if "manual_severity" not in st.session_state:
    st.session_state.manual_severity = None


def urgency_css_class(level: str) -> str:
    level = (level or "").lower()
    if "emergency" in level:
        return "urgency-emergency"
    if "urgent" in level:
        return "urgency-urgent"
    if "primary" in level:
        return "urgency-primary"
    return "urgency-selfcare"


def urgency_icon(level: str) -> str:
    level = (level or "").lower()
    if "emergency" in level:
        return "🚨"
    if "urgent" in level:
        return "⚠️"
    if "primary" in level:
        return "🩺"
    return "🌿"


# ------------------------------------------------------------------
# Sidebar: live clinical state panel
# ------------------------------------------------------------------
with st.sidebar:
    st.header("📊 Clinical State Tracking")
    state = st.session_state.health_state

    st.markdown('<div class="stat-label">Extracted Symptoms</div>', unsafe_allow_html=True)
    if state["symptoms"]:
        chips = "".join(f'<span class="symptom-chip">{s.capitalize()}</span>' for s in state["symptoms"])
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.caption("No symptoms recorded yet.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Duration</div>
            <div class="stat-value">{state['duration'] or 'Unspecified'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Severity (auto-extracted)</div>
            <div class="stat-value">{state['severity'] or 'Unspecified'} / 10</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Manual severity slider — lets the patient set/override severity directly
    st.markdown('<div class="stat-label">Set severity yourself</div>', unsafe_allow_html=True)
    slider_val = st.slider(
        "Severity",
        min_value=1,
        max_value=10,
        value=state["severity"] or 5,
        label_visibility="collapsed",
    )
    if st.button("Confirm severity", use_container_width=True):
        st.session_state.health_state["severity"] = slider_val
        st.rerun()

    if state["triage_level"]:
        st.divider()
        st.markdown(
            f"""
            <div class="urgency-card {urgency_css_class(state['triage_level'])}">
                {urgency_icon(state['triage_level'])} {state['triage_level']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("🔄 Start new session", use_container_width=True):
        st.session_state.health_state = new_session_state()
        st.rerun()

# ------------------------------------------------------------------
# Chat history
# ------------------------------------------------------------------
for msg in st.session_state.health_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------
# Chat input -> run the agent
# ------------------------------------------------------------------
if user_input := st.chat_input("How are you feeling today? (e.g., 'I have a sore throat and fever')"):
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.health_state["messages"].append({"role": "user", "content": user_input})

    with st.spinner("Analyzing symptoms and checking red flags..."):
        try:
            updated_state = health_agent.invoke(st.session_state.health_state)
            st.session_state.health_state = updated_state
        except Exception as e:
            st.session_state.health_state["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "⚠️ I couldn't reach the AI service right now. Double-check that "
                        "GOOGLE_API_KEY is set correctly and that your Google AI Studio key has "
                        "access to the Gemma models, then try again.\n\n"
                        f"*Technical detail: {type(e).__name__}: {e}*"
                    ),
                }
            )

    st.rerun()

# ------------------------------------------------------------------
# Final case summary card
# ------------------------------------------------------------------
final_state = st.session_state.health_state
if final_state["triage_level"]:
    st.divider()
    st.markdown(
        f"""
        <div class="urgency-card {urgency_css_class(final_state['triage_level'])}">
            {urgency_icon(final_state['triage_level'])} Assessed Urgency: {final_state['triage_level']}
        </div>
        <div class="disclaimer">
            I am an AI, not a doctor. Seek immediate emergency services for severe or escalating symptoms.
        </div>
        """,
        unsafe_allow_html=True,
    )
