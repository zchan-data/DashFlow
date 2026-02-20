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


# --- 3. Helper Function: Call Gemini ---
def get_ai_response(prompt_text, phase):
    """Handles the API call to Gemini with the appropriate system instructions."""

    if phase == 1:
        sys_instruct = """Role: You are the "Initial Profiling Agent" for an interactive data analysis web application. You are an expert data scientist and Python developer.
        
Task: The user will provide you with the metadata of a freshly uploaded CSV file. This metadata will include the output of Pandas functions like df.info(), df.describe(), and df.isnull().sum(). You will never receive the raw row data. Your job is to analyze this metadata, highlight critical data quality issues, and provide exactly three actionable next steps for data cleaning or exploratory data analysis (EDA).

Tone: Professional, concise, and highly analytical.

Output Constraints: You must output your response STRICTLY as a valid JSON object. Do not include markdown formatting like ```json or any conversational filler outside of the JSON structure.

Required JSON Structure:
{
  "summary_assessment": "A 2-3 sentence overview of the dataset's shape, primary data types, and general health.",
  "key_warnings": ["A list of 1-3 critical issues found in the metadata"],
  "suggested_prompts": ["Actionable prompt 1", "Actionable prompt 2", "Actionable prompt 3"]
}"""

    elif phase == 2:
        sys_instruct = """Role: You are the "Data Transformation Agent" for an interactive data analysis web application. Your job is to translate the user's natural language requests into precise, executable Python code using the pandas library.

Context: You will receive the user's prompt along with the current columns and data types of the active dataframe.

Rules for Code Generation:
1. Assume the active dataframe is already loaded into a variable named df.
2. Your code must modify df directly or create necessary intermediate variables.
3. Do not include commands to read or load data (like pd.read_csv()).
4. Do not include print() statements. The application backend will handle displaying the updated df.

Output Constraints: You must output your response STRICTLY as a valid JSON object. Do not include markdown formatting like ```json or any conversational filler outside of the JSON structure.

Required JSON Structure:
{
  "thought_process": "A 1-sentence internal reasoning of what pandas operations are needed.",
  "python_code": "The raw, executable Python code string. Use \\n for line breaks.",
  "explanation_for_user": "A brief, friendly explanation of what you just did to the data.",
  "suggested_next_steps": ["Logical next step 1", "Logical next step 2"]
}"""

    else:
        sys_instruct = """Role: You are the "Visualization Agent" for an interactive data analysis web application. Your job is to translate the user's data visualization requests into executable Python code using the plotly.express or plotly.graph_objects libraries.

Context: You will receive the user's prompt along with the current columns and data types of the active dataframe.

Rules for Code Generation:
1. Assume the active dataframe is already loaded into a variable named df.
2. Always import the necessary Plotly modules within the code block (e.g., import plotly.express as px).
3. Your code must generate a Plotly figure and assign it to a variable named fig.
4. Do not include commands to show the plot (like fig.show()).
5. Ensure the charts are aesthetically pleasing.

Output Constraints: You must output your response STRICTLY as a valid JSON object. Do not include markdown formatting like ```json or any conversational filler outside of the JSON structure.

Required JSON Structure:
{
  "thought_process": "A 1-sentence internal reasoning of why this specific chart type is best.",
  "python_code": "The raw, executable Python code string. Remember to assign the final chart to a variable named 'fig'.",
  "explanation_for_user": "A brief explanation of what the chart illustrates.",
  "suggested_tweaks": ["Clickable prompt to modify the chart", "Clickable prompt to drill down further"]
}"""

    # We use gemini-2.5-flash as updated previously
    phase_model = genai.GenerativeModel(
        "gemini-2.5-flash", system_instruction=sys_instruct
    )

    # We ask for JSON specifically to ensure clean parsing
    response = phase_model.generate_content(
        prompt_text, generation_config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)


# --- 4. Main Application UI ---
st.title("AI Data Analyst")

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

    # Show current data state
    with st.expander("View Current Dataframe"):
        st.dataframe(st.session_state.df.head())

    # Render chat history
    for message in st.session_state.cleaning_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "code" in message:
                with st.expander("Show Executed Code"):
                    st.code(message["code"], language="python")

    # Handle new chat input
    if prompt := st.chat_input(
        "Ask the AI to clean, filter, or manipulate the data..."
    ):
        st.chat_message("user").markdown(prompt)
        st.session_state.cleaning_chat.append({"role": "user", "content": prompt})

        # Prepare context for AI
        schema_context = f"Columns: {st.session_state.df.columns.tolist()}\nTypes: {st.session_state.df.dtypes.to_dict()}"
        full_prompt = f"Context: {schema_context}\nUser Request: {prompt}"

        with st.spinner("Writing and executing Pandas code..."):
            try:
                ai_response = get_ai_response(full_prompt, phase=2)
                code_to_run = ai_response["python_code"]
                explanation = ai_response["explanation_for_user"]

                # Execute the code safely
                local_vars = {"df": st.session_state.df.copy(), "pd": pd}
                exec(code_to_run, {}, local_vars)

                # Update state
                st.session_state.df = local_vars["df"]

                # Display AI reply
                with st.chat_message("ai"):
                    st.markdown(explanation)
                    with st.expander("Show Executed Code"):
                        st.code(code_to_run, language="python")

                # Save to history
                st.session_state.cleaning_chat.append(
                    {"role": "ai", "content": explanation, "code": code_to_run}
                )
                st.rerun()

            except Exception as e:
                st.error(f"Execution failed: {e}")

    # Navigation
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Revert to Original Upload"):
            st.session_state.df = st.session_state.raw_df.copy()
            st.session_state.cleaning_chat.append(
                {"role": "ai", "content": "🔄 Reverted to original data."}
            )
            st.rerun()
    with col2:
        if st.button("Move to Visualization (Phase 3)"):
            st.session_state.current_phase = 3
            st.rerun()

# ==========================================
# PHASE 3: VISUALIZATION LOOP
# ==========================================
elif st.session_state.current_phase == 3:
    st.header("Step 3: Visualize Data")

    # Render visualization chat history
    for message in st.session_state.viz_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "fig" in message:
                st.plotly_chart(message["fig"], use_container_width=True)
            if "code" in message:
                with st.expander("Show Plotly Code"):
                    st.code(message["code"], language="python")

    # Handle new chat input
    if prompt := st.chat_input("Ask the AI to generate a Plotly chart..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.viz_chat.append({"role": "user", "content": prompt})

        schema_context = f"Columns: {st.session_state.df.columns.tolist()}\nTypes: {st.session_state.df.dtypes.to_dict()}"
        full_prompt = f"Context: {schema_context}\nUser Request: {prompt}"

        with st.spinner("Generating chart..."):
            try:
                ai_response = get_ai_response(full_prompt, phase=3)
                code_to_run = ai_response["python_code"]
                explanation = ai_response["explanation_for_user"]

                # Execute the code to extract 'fig'
                local_vars = {"df": st.session_state.df, "px": px}
                exec(code_to_run, {}, local_vars)

                if "fig" in local_vars:
                    fig = local_vars["fig"]

                    with st.chat_message("ai"):
                        st.markdown(explanation)
                        st.plotly_chart(fig, use_container_width=True)
                        with st.expander("Show Plotly Code"):
                            st.code(code_to_run, language="python")

                    # Save to history
                    st.session_state.viz_chat.append(
                        {
                            "role": "ai",
                            "content": explanation,
                            "code": code_to_run,
                            "fig": fig,
                        }
                    )
                else:
                    st.error("AI code did not generate a 'fig' variable.")

            except Exception as e:
                st.error(f"Visualization failed: {e}")

    # Navigation
    st.divider()
    if st.button("Back to Data Cleaning (Phase 2)"):
        st.session_state.current_phase = 2
        st.rerun()
