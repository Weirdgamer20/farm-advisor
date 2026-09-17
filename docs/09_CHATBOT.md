# 09 - Chatbot & Interactive Advisory Interface

## 1. Advisory Workflow
The system features an interactive agricultural advisory workflow integrated into the Streamlit presentation layer:
1. **Interactive Prompting**: Guides the farmer through image submission and sensor/soil parameters entry.
2. **Context-Aware Responses**: Translates neural network suitability scores into natural language explanations.
3. **Condition-Specific Diagnostics**: Details exact nutrient shortfalls or excesses (e.g., *"Nitrogen is below the dataset's typical range for Tomato"*).

## 2. Future Conversational Extension
The architecture is structured to support seamless LLM / Chatbot extension:
- `src/analysis.py` exports `generate_advisory_summary()`, outputting a clean JSON dictionary.
- This dictionary serves as structured system context for an agronomic RAG or conversational agent (e.g., using LangChain or LlamaIndex) to answer farmer follow-up questions regarding chemical brand dosages, application timings, and organic alternatives.
