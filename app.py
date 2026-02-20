import json
from typing import List, TypedDict

import google.generativeai as genai
import pandas as pd
import plotly.express as px
import streamlit as st


# --- 1. Define Schemas for Structured Output (The Silver Bullet) ---
class Phase1Schema(TypedDict):
    summary_assessment: str
    key_warnings: List[str]
    suggested_prompts: List[str]


class Phase2Schema(TypedDict):
    thought_process: str
    python_code: str
    explanation_for_user: str
    suggested_next_steps: List[str]


class Phase3Schema(TypedDict):
    thought_process: str
    python_code: str
    explanation_for_user: str
    suggested_tweaks: List[str]


# --- 2. API Configuration ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- 3. Session State Initialization ---
if "df" not in st.session_state:
    st.session_state.df = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "current_phase" not in st.session_state:
    st.session_state.current_phase = 1
if "cleaning_chat" not in st.session_state:
    st.session_state.cleaning_chat = []
if "viz_chat" not in st.session_state:
    st.session_state.viz_chat = []
if "current_suggestions" not in st.session_state:
    st.session_state.current_suggestions = []
if "current_viz_suggestions" not in st.session_state:
    st.session_state.current_viz_suggestions = [
        "Show me a correlation heatmap of all numeric columns.",
        "Create a histogram of the most interesting feature.",
    ]


# --- 4. Helper Function: Call Gemini ---
def get_ai_response(prompt_text, phase):
    from google.generativeai.types import HarmBlockThreshold, HarmCategory

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    if phase == 1:
        sys_instruct = """Role: You are the Initial Profiling Agent for a data analysis web app.
Task: Analyze metadata, highlight critical data quality issues, and provide 3 actionable next steps.
CRITICAL RULE: Your suggested next steps MUST be data manipulation tasks (e.g., dropping nulls, changing data types, imputing missing values, renaming columns, encoding categories). DO NOT suggest observational tasks."""
        schema = Phase1Schema

    elif phase == 2:
        sys_instruct = """Role: You are the Data Transformation Agent.
Task: Translate natural language requests into executable Python code using the pandas library.
Rules:
1. Assume the dataframe is loaded into a variable named df.
2. Modify df directly or create necessary intermediate variables.
3. If the user asks for a summary, overwrite the 'df' variable with that summary dataframe (e.g., df = df.describe()).
4. Do not use print() or pd.read_csv()."""
        schema = Phase2Schema

    else:
        sys_instruct = """Role: You are the Visualization Agent.
Task: Translate requests into executable Python code using plotly.express or plotly.graph_objects.
Rules:
1. Assume the dataframe is loaded into a variable named df.
2. Always import necessary Plotly modules.
3. Generate a Plotly figure and assign it to a variable named fig.
4. Do not include commands to show the plot."""
        schema = Phase3Schema

    phase_model = genai.GenerativeModel(
        "gemini-2.5-flash", system_instruction=sys_instruct
    )

    # Using response_schema forces the API to perfectly format and escape the JSON
    response = phase_model.generate_content(
        prompt_text,
        safety_settings=safety_settings,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json", response_schema=schema
        ),
    )

    try:
        # The output is now guaranteed to be clean JSON, so we can parse it directly
        return json.loads(response.text)
    except Exception as e:
        reason = (
            response.candidates[0].finish_reason.name
            if response.candidates
            else "UNKNOWN"
        )
        raise ValueError(
            f"The AI returned an invalid response (Finish Reason: {reason}). Error: {e}"
        )


# --- 5. Main Application UI ---
st.title("AI Data Analyst")

# --- DEBUG MODE TOGGLE ---
with st.sidebar:
    st.header("⚙️ Settings")
    debug_mode = st.checkbox("Enable Debug Mode", value=False)
    if debug_mode:
        st.info(
            "Debug mode is ON. Raw AI JSON outputs will be displayed below the chat."
        )

# ==========================================
# PHASE 1: UPLOAD & PROFILE
# ==========================================
if st.session_state.current_phase == 1:
    st.header("Step 1: Upload Your Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.raw_df = df.copy()

        st.success("File uploaded successfully!")
        st.dataframe(df.head())

        import io

        buffer = io.StringIO()
        df.info(buf=buffer)
        metadata = (
            f"Info:\n{buffer.getvalue()}\n\nDescribe:\n{df.describe().to_string()}"
        )

        if st.button("Analyze Data Profile"):
            with st.spinner("AI is analyzing the dataset..."):
                try:
                    profile_json = get_ai_response(
                        f"Here is the metadata: {metadata}", phase=1
                    )

                    if debug_mode:
                        with st.expander(
                            "🔍 DEBUG: View Raw JSON Response", expanded=True
                        ):
                            st.json(profile_json)

                    st.session_state.current_suggestions = profile_json.get(
                        "suggested_prompts", []
                    )
                    st.subheader("AI Assessment")
                    st.write(
                        profile_json.get("summary_assessment", "Analysis complete.")
                    )

                    st.markdown("**Key Warnings:**")
                    for warning in profile_json.get("key_warnings", []):
                        st.markdown(f"- ⚠️ {warning}")

                except Exception as e:
                    st.error(f"Error generating profile: {e}")

        if st.button("Move to Data Cleaning (Phase 2)"):
            st.session_state.current_phase = 2
            st.rerun()

# ==========================================
# PHASE 2: CLEANING & EXPLORATION LOOP
# ==========================================
elif st.session_state.current_phase == 2:
    st.header("Step 2: Clean & Explore")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back to Step 1"):
            st.session_state.current_phase = 1
            st.rerun()
    with col2:
        if st.button("🔴 Revert to Original", use_container_width=True):
            st.session_state.df = st.session_state.raw_df.copy()
            st.session_state.cleaning_chat.append(
                {"role": "ai", "content": "🔄 Reverted to original data."}
            )
            st.session_state.current_suggestions = []
            st.rerun()
    with col3:
        if st.button(
            "🔵 Move to Visualization", type="primary", use_container_width=True
        ):
            st.session_state.current_phase = 3
            st.rerun()

    st.divider()

    with st.expander("View Current Dataframe", expanded=True):
        st.dataframe(st.session_state.df.head())

    for message in st.session_state.cleaning_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "code" in message:
                with st.expander("Show Executed Code"):
                    st.code(message["code"], language="python")

    user_prompt = None
    if st.session_state.current_suggestions:
        st.caption("✨ **Suggested Next Steps:**")
        sugg_cols = st.columns(len(st.session_state.current_suggestions))
        for i, suggestion in enumerate(st.session_state.current_suggestions):
            if sugg_cols[i].button(suggestion, key=f"sugg_{i}"):
                user_prompt = suggestion

    if chat_input := st.chat_input(
        "Ask the AI to clean, filter, or manipulate the data..."
    ):
        user_prompt = chat_input

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.cleaning_chat.append({"role": "user", "content": user_prompt})

        schema_context = f"Columns: {st.session_state.df.columns.tolist()}\nTypes: {st.session_state.df.dtypes.to_dict()}"
        full_prompt = f"Context: {schema_context}\nUser Request: {user_prompt}"

        with st.spinner("Writing and executing Pandas code..."):
            try:
                ai_response = get_ai_response(full_prompt, phase=2)

                if debug_mode:
                    with st.expander("🔍 DEBUG: View Raw JSON Response", expanded=True):
                        st.json(ai_response)

                code_to_run = ai_response["python_code"]
                explanation = ai_response["explanation_for_user"]
                st.session_state.current_suggestions = ai_response.get(
                    "suggested_next_steps", []
                )

                local_vars = {"df": st.session_state.df.copy(), "pd": pd}
                exec(code_to_run, {}, local_vars)
                st.session_state.df = local_vars["df"]

                st.session_state.cleaning_chat.append(
                    {"role": "ai", "content": explanation, "code": code_to_run}
                )
                st.rerun()

            except Exception as e:
                st.error(f"Execution failed: {e}")

# ==========================================
# PHASE 3: VISUALIZATION LOOP
# ==========================================
elif st.session_state.current_phase == 3:
    st.header("Step 3: Visualize Data")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back to Step 2"):
            st.session_state.current_phase = 2
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Visualizations", use_container_width=True):
            st.session_state.viz_chat = []
            st.session_state.current_viz_suggestions = [
                "Show me a correlation heatmap of all numeric columns.",
                "Create a histogram of the most interesting feature.",
            ]
            st.rerun()

    st.divider()

    for message in st.session_state.viz_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "fig" in message:
                st.plotly_chart(message["fig"], use_container_width=True)
            if "code" in message:
                with st.expander("Show Plotly Code"):
                    st.code(message["code"], language="python")

    user_prompt = None
    if st.session_state.current_viz_suggestions:
        st.caption("🎨 **Suggested Visualizations & Tweaks:**")
        sugg_cols = st.columns(len(st.session_state.current_viz_suggestions))
        for i, suggestion in enumerate(st.session_state.current_viz_suggestions):
            if sugg_cols[i].button(suggestion, key=f"viz_sugg_{i}"):
                user_prompt = suggestion

    if chat_input := st.chat_input(
        "Ask the AI to generate or modify a Plotly chart..."
    ):
        user_prompt = chat_input

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.viz_chat.append({"role": "user", "content": user_prompt})

        schema_context = f"Columns: {st.session_state.df.columns.tolist()}\nTypes: {st.session_state.df.dtypes.to_dict()}"
        full_prompt = f"Context: {schema_context}\nUser Request: {user_prompt}"

        with st.spinner("Generating chart..."):
            try:
                ai_response = get_ai_response(full_prompt, phase=3)

                if debug_mode:
                    with st.expander("🔍 DEBUG: View Raw JSON Response", expanded=True):
                        st.json(ai_response)

                code_to_run = ai_response["python_code"]
                explanation = ai_response["explanation_for_user"]
                st.session_state.current_viz_suggestions = ai_response.get(
                    "suggested_tweaks", []
                )

                local_vars = {"df": st.session_state.df, "px": px}
                exec(code_to_run, {}, local_vars)

                if "fig" in local_vars:
                    fig = local_vars["fig"]
                    st.session_state.viz_chat.append(
                        {
                            "role": "ai",
                            "content": explanation,
                            "code": code_to_run,
                            "fig": fig,
                        }
                    )
                    st.rerun()
                else:
                    st.error("AI code did not generate a 'fig' variable.")

            except Exception as e:
                st.error(f"Visualization failed: {e}")
