AI Data Analyst (DashFlow)
A fully interactive, AI-powered data analysis web application built with Python, Streamlit, and Google's Gemini 2.5 Flash model.

This application acts as an autonomous data analyst. It allows users to upload raw datasets, instantly generates profiling statistics, and provides an iterative, chat-based interface to clean data, perform exploratory data analysis (EDA), and generate interactive visualizations—all by translating natural language into executable Pandas and Plotly code.

[View Live App on Streamlit Community Cloud] https://dashflow-5psusywa4cvxqhzwjs4dmt.streamlit.app/

Key Features
The application is structured into a guided, 3-phase workflow:

1. Automated Data Profiling
Upload any raw CSV.

The app calculates metadata behind the scenes and feeds it to the LLM context.

Instantly generates a health assessment, flags critical data warnings (e.g., high null counts, improper data types), and suggests clickable starting points.

2. Interactive Cleaning & Exploration Loop
A conversational UI where users can type natural language requests (e.g., "Impute missing ages with the median" or "Drop the ID column").

The AI dynamically writes and executes pandas code directly against the active dataframe.

Features a Revert button to instantly undo mistakes and return to the raw upload.

3. Dynamic Visualization Generation
Transitions the cleaned data into a visualization environment.

Generates interactive plotly charts based on user prompts.

Offers dynamic UI buttons to continuously tweak the charts (e.g., "Change to a scatter plot," "Add a trendline").

Under the Hood: LLM Engineering & Architecture
Building reliable code-generation agents requires strict guardrails. This project implements several advanced LLM engineering techniques to ensure stability:

Structured Outputs (TypedDicts): The Gemini API is strictly constrained using response_schema. This forces the AI to output perfect, escaped JSON dictionaries, completely eliminating the common issue of corrupted JSON parsing and quote-escaping crashes.

Complex State Management: Utilizes Streamlit's st.session_state extensively to maintain a continuous memory of the modified dataframe, conversation logs, and dynamic UI button states across app reruns.

Isolated Execution: Translates LLM output strings into raw Python code and executes it within a safely scoped local dictionary using exec().

Built-in Debugger: Includes a toggleable Developer Debug Mode in the sidebar that intercepts and renders the raw JSON payloads from the Gemini API for easy observability.

Tech Stack
Frontend/UI: Streamlit

Data Manipulation: Pandas

Visualizations: Plotly Express

AI/LLM: Google GenAI SDK (Gemini 2.5 Flash)

Local Installation & Setup
Follow these step to run locally:

1. Clone the repository:

Bash
git clone https://github.com/your-username/DashFlow.git
cd DashFlow
2. Install dependencies: It is recommended to use a virtual environment.

Bash
pip install -r requirements.txt
3. Set up your Google API Key: This app requires a free Google Gemini API key.

Create a folder named .streamlit in the root directory.

Create a file inside it named secrets.toml.

Add your API key to the file:

Ini, TOML
GOOGLE_API_KEY = "your_api_key_here"
4. Run the app:

Bash
streamlit run app.py
