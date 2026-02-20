import json

import google.generativeai as genai
import pandas as pd
import plotly.express as px
import streamlit as st

# --- 1. API Configuration ---
# Pulls the API key securely from .streamlit/secrets.toml
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Initialize the Gemini model (2.5 Flash is fast and cheap for this)
# You will update the system_instruction in the actual helper function below based on the phase.
model = genai.GenerativeModel("gemini-2.5-flash")

# --- 2. Session State Initialization ---
if "df" not in st.session_state:
    st.session_state.df = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "current_phase" not in st.session_state:
    st.session_state.current_phase = 1  # 1: Profile, 2: Clean, 3: Visualize
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


# --- 3. Helper Function: Call Gemini ---
def get_ai_response(prompt_text, phase):
    """Handles the API call to Gemini with the appropriate system instructions."""

    if phase == 1:
        sys_instruct = """Role: You are the "Initial Profiling Agent" for an interactive data analysis web application. You are an expert data scientist and Python developer.
        
Task: The user will provide you with the metadata of a freshly uploaded CSV file. Your job is to analyze this metadata, highlight critical data quality issues, and provide exactly three actionable next steps. 

CRITICAL RULE: Your suggested next steps MUST be data manipulation tasks (e.g., dropping nulls, changing data types, imputing missing values, renaming columns, encoding categories). DO NOT suggest observational tasks.

Tone: Professional, concise, and highly analytical.
Output Constraints: Output STRICTLY as a valid JSON object.

Required JSON Structure:
{
  "summary_assessment": "A 2-3 sentence overview of the dataset's health.",
  "key_warnings": ["A list of 1-3 critical issues found in the metadata"],
  "suggested_prompts": ["Actionable manipulation prompt 1", "Actionable manipulation prompt 2", "Actionable manipulation prompt 3"]
}"""

    elif phase == 2:
        sys_instruct = """Role: You are the "Data Transformation Agent" for an interactive data analysis web application. Your job is to translate the user's natural language requests into precise, executable Python code using the pandas library.

Context: You will receive the user's prompt along with the current columns and data types of the active dataframe.

Rules for Code Generation:
1. Assume the active dataframe is already loaded into a variable named df.
2. Your code must modify df directly or create necessary intermediate variables.
3. If the user asks for a summary or to "examine" something, overwrite the 'df' variable with that summary dataframe (e.g., df = df.describe()) so it can be viewed.
4. Do not include commands to read or load data.
5. Do not include print() statements. 
6. CRITICAL STRING RULE: You MUST use single quotes ('') for all strings inside your Python code (e.g., df['column_name']). Using double quotes will corrupt the JSON output.

Output Constraints: Output STRICTLY as a valid JSON object.

Required JSON Structure:
{
  "thought_process": "A 1-sentence internal reasoning of what pandas operations are needed.",
  "python_code": "The raw, executable Python code string. Use \\n for line breaks.",
  "explanation_for_user": "A brief, friendly explanation of what you just did to the data.",
  "suggested_next_steps": ["Logical next step 1", "Logical next step 2"]
}"""

    else:
        sys_instruct = """Role: You are the "Visualization Agent" for an interactive data analysis web application. Your job is to translate the user's data visualization requests into executable Python code using the plotly.express or plotly.graph_objects libraries.

Rules for Code Generation:
1. Assume the active dataframe is already loaded into a variable named df.
2. Always import the necessary Plotly modules within the code block.
3. Your code must generate a Plotly figure and assign it to a variable named fig.
4. Do not include commands to show the plot (like fig.show()).
5. CRITICAL STRING RULE: You MUST use single quotes ('') for all strings inside your Python code. Using double quotes will corrupt the JSON output.

Output Constraints: Output STRICTLY as a valid JSON object.

Required JSON Structure:
{
  "thought_process": "A 1-sentence internal reasoning of why this chart type is best.",
  "python_code": "The raw, executable Python code string. Remember to assign to 'fig'.",
  "explanation_for_user": "A brief explanation of what the chart illustrates.",
  "suggested_tweaks": ["Clickable prompt to modify", "Clickable prompt to drill down"]
}"""

    phase_model = genai.GenerativeModel(
        "gemini-2.5-flash", system_instruction=sys_instruct
    )

    # We remove the rigid JSON generation_config to allow our custom parser to handle errors safely
    response = phase_model.generate_content(prompt_text)

    # --- SAFELY PARSE JSON ---
    try:
        raw_text = response.text

        # --- TERMINAL DEBUGGER ---
        print(f"\n--- RAW AI RESPONSE (PHASE {phase}) ---")
        print(raw_text)
        print("--------------------------------------\n")

        # 1. Clean up any markdown hallucinated by the AI
        cleaned_text = raw_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]

        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]

        # 2. Parse the clean string into a Python dictionary
        return json.loads(cleaned_text.strip())

    except json.JSONDecodeError as e:
        # This MUST come before ValueError so it catches formatting bugs correctly
        raise ValueError(
            f"The AI generated corrupted JSON (Error: {e}). Please try again."
        )
    except ValueError:
        # This catches actual blank responses from safety filters or API limits
        reason = (
            response.candidates[0].finish_reason.name
            if response.candidates
            else "UNKNOWN"
        )
        raise ValueError(
            f"The AI returned a blank response (Finish Reason: {reason}). Try clicking the suggestion again."
        )


# --- 4. Main Application UI ---
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
        # Load and save data
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.raw_df = df.copy()

        st.success("File uploaded successfully!")
        st.dataframe(df.head())

        # Generate Metadata for the AI
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

                    st.markdown("**Suggested Starting Points:**")
                    for suggestion in profile_json.get("suggested_prompts", []):
                        st.markdown(f"- 💡 {suggestion}")

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

    # 1. Top Navigation & Actions
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back to Step 1 (Upload)"):
            st.session_state.current_phase = 1
            st.rerun()
    with col2:
        if st.button("🔴 Revert to Original", use_container_width=True):
            st.session_state.df = st.session_state.raw_df.copy()
            st.session_state.cleaning_chat.append(
                {"role": "ai", "content": "🔄 Reverted to original data."}
            )
            st.session_state.current_suggestions = []  # Clear suggestions on revert
            st.rerun()
    with col3:
        if st.button(
            "🔵 Move to Visualization", type="primary", use_container_width=True
        ):
            st.session_state.current_phase = 3
            st.rerun()

    st.divider()

    # Show current data state
    with st.expander("View Current Dataframe", expanded=True):
        st.dataframe(st.session_state.df.head())

    # Render chat history
    for message in st.session_state.cleaning_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "code" in message:
                with st.expander("Show Executed Code"):
                    st.code(message["code"], language="python")

    # 2. Render Suggestions as Buttons
    user_prompt = None
    if st.session_state.current_suggestions:
        st.caption("✨ **Suggested Next Steps:**")
        sugg_cols = st.columns(len(st.session_state.current_suggestions))
        for i, suggestion in enumerate(st.session_state.current_suggestions):
            # If a suggestion button is clicked, assign it to user_prompt
            if sugg_cols[i].button(suggestion, key=f"sugg_{i}"):
                user_prompt = suggestion

    # 3. Handle Chat Input
    # If the user types in the bar, it overrides the button clicks
    if chat_input := st.chat_input(
        "Ask the AI to clean, filter, or manipulate the data..."
    ):
        user_prompt = chat_input

    # 4. Execute AI Logic (Triggers if either a button was clicked or text was typed)
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.cleaning_chat.append({"role": "user", "content": user_prompt})

        schema_context = f"Columns: {st.session_state.df.columns.tolist()}\nTypes: {st.session_state.df.dtypes.to_dict()}"
        full_prompt = f"Context: {schema_context}\nUser Request: {user_prompt}"

        with st.spinner("Writing and executing Pandas code..."):
            try:
                ai_response = get_ai_response(full_prompt, phase=2)
                code_to_run = ai_response["python_code"]
                explanation = ai_response["explanation_for_user"]

                # --- INJECT DEBUGGER HERE ---
                if debug_mode:
                    with st.expander("🔍 DEBUG: View Raw JSON Response", expanded=True):
                        st.json(ai_response)

                # Update suggestions with the new ones from the AI
                st.session_state.current_suggestions = ai_response.get(
                    "suggested_next_steps", []
                )

                # Execute the code safely
                local_vars = {"df": st.session_state.df.copy(), "pd": pd}
                exec(code_to_run, {}, local_vars)

                # Update state
                st.session_state.df = local_vars["df"]

                # Save to history
                st.session_state.cleaning_chat.append(
                    {"role": "ai", "content": explanation, "code": code_to_run}
                )
                st.rerun()  # Refresh to show new dataframe, new chat, and new suggestions

            except Exception as e:
                st.error(f"Execution failed: {e}")


# ==========================================
# PHASE 3: VISUALIZATION LOOP
# ==========================================
elif st.session_state.current_phase == 3:
    st.header("Step 3: Visualize Data")

    # 1. Top Navigation & Actions
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back to Step 2 (Cleaning)"):
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

    # Render visualization chat history and plots
    for message in st.session_state.viz_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "fig" in message:
                st.plotly_chart(message["fig"], use_container_width=True)
            if "code" in message:
                with st.expander("Show Plotly Code"):
                    st.code(message["code"], language="python")

    # 2. Render Suggestions as Buttons
    user_prompt = None
    if st.session_state.current_viz_suggestions:
        st.caption("🎨 **Suggested Visualizations & Tweaks:**")
        # Handle dynamic column sizing based on the number of suggestions
        sugg_cols = st.columns(len(st.session_state.current_viz_suggestions))
        for i, suggestion in enumerate(st.session_state.current_viz_suggestions):
            if sugg_cols[i].button(suggestion, key=f"viz_sugg_{i}"):
                user_prompt = suggestion

    # 3. Handle Chat Input
    # If the user types in the bar, it overrides the button clicks
    if chat_input := st.chat_input(
        "Ask the AI to generate or modify a Plotly chart..."
    ):
        user_prompt = chat_input

    # 4. Execute AI Logic (Triggers if either a button was clicked or text was typed)
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.viz_chat.append({"role": "user", "content": user_prompt})

        schema_context = f"Columns: {st.session_state.df.columns.tolist()}\nTypes: {st.session_state.df.dtypes.to_dict()}"
        full_prompt = f"Context: {schema_context}\nUser Request: {user_prompt}"

        with st.spinner("Generating chart..."):
            try:
                ai_response = get_ai_response(full_prompt, phase=3)
                code_to_run = ai_response["python_code"]
                explanation = ai_response["explanation_for_user"]

                # --- INJECT DEBUGGER HERE ---
                if debug_mode:
                    with st.expander("🔍 DEBUG: View Raw JSON Response", expanded=True):
                        st.json(ai_response)

                # Update suggestions with the new "tweaks" from the AI
                st.session_state.current_viz_suggestions = ai_response.get(
                    "suggested_tweaks", []
                )

                # Execute the code to extract 'fig'
                local_vars = {"df": st.session_state.df, "px": px}
                exec(code_to_run, {}, local_vars)

                if "fig" in local_vars:
                    fig = local_vars["fig"]

                    # Save everything to history and rerun
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
