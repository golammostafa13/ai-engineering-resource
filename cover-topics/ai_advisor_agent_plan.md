# AI Model Advisor Agent: Implementation Plan

## 🎯 Goal
Build an intelligent agent that acts as a consultant. When a user describes a task or problem they want to solve, the agent analyzes the intent, maps it to a strict taxonomy, searches the web for current benchmarks, and provides a ranked list of the best AI models for that specific job, each with a one-line justification.

---

## 🏗️ System Architecture

### 1. The Taxonomy Classifier (System Prompt)
The agent will first classify the user's prompt into one or more of our defined taxonomy categories to guide its search.

*   **Complexity:** Small (Formatting/Chat), Medium (Summarization), Large (Complex Logic).
*   **Capability:** General, Reasoning (RL models), Coding, Creative/Writing, Vision.
*   **Security/Privacy:** Public API acceptable, or Private/Local required (sensitive data).

### 2. The Tool Set
The agent needs only one external tool to ensure its knowledge is never outdated:
*   `search_web(query: str)`: Searches for live leaderboards (e.g., LMSYS Chatbot Arena, Artificial Analysis, current coding benchmarks).

### 3. The Execution Loop
1.  **Analyze & Classify:** Read user request, identify the taxonomy tags.
2.  **Dynamic Search:** Generate a search query based on the tags (e.g., `"best local open source coding LLM benchmarks 2026"`).
3.  **Synthesize:** Read search results and rank the top 3 models.
4.  **Format Output:** Present the ranked list with a concise, one-line remark for each.

---

## 🛠️ Implementation Details

### Step 1: Tool Definition
We will define a Python function `search_web(query)` that hooks into a search API (like DuckDuckGo, Tavily, or SerpAPI). We expose this to the agent using standard JSON schema.

### Step 2: The Agent's System Prompt
```text
You are an expert AI Architect and Model Advisor.
Your job is to recommend the best AI models for a user's specific use case.

First, silently classify the user's request based on:
1. Complexity (Small, Medium, Large)
2. Capability (Reasoning, Coding, Creative, Vision, General)
3. Security Requirements (Public API vs Local/Private)

Second, use the 'search_web' tool to find the most current benchmark winners and community favorites for that specific classification in the current year.

Third, output a ranked list of the top 3 recommended models.
For each model, provide exactly ONE line explaining why it is ranked there based on the classification.

Format:
1. [Model Name] (Provider/Local): [One line remark on why it fits the taxonomy]
2. [Model Name] (Provider/Local): [One line remark on why it fits the taxonomy]
3. [Model Name] (Provider/Local): [One line remark on why it fits the taxonomy]
```

### Step 3: Example Expected Output

**User Input:**
> "I need to build an automated pipeline that reviews our proprietary internal codebase for security vulnerabilities."

**Agent Internal Process:**
*   *Taxonomy detected:* Complexity (Large), Capability (Coding), Security (Private/Local required).
*   *Tool called:* `search_web("best local open weight coding llms for security review")`

**Agent Final Output:**
1. **Qwen2.5-Coder 32B (Local):** Best-in-class open-weight coding performance that can run securely on your private servers to protect proprietary code.
2. **DeepSeek-Coder V2 (Local):** Highly capable reasoning and coding model that offers excellent performance without sending data to a cloud API.
3. **Llama-3-70B-Instruct (Local):** A heavy but versatile local model that provides strong generalized logic needed for complex vulnerability analysis.

---

## 🚀 Next Steps to Build
1.  **Set up the environment:** Install necessary libraries (e.g., `openai` for LLM routing, `duckduckgo-search` for free web searching).
2.  **Write the Python script:** Implement the prompt, the tool, and the execution loop.
3.  **Test edge cases:** Ask it for a math problem, a creative writing task, and a data-sensitive task to verify it searches and ranks correctly.
