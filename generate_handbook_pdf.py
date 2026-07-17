import os
from fpdf import FPDF

class AgentHandbookPDF(FPDF):
    def header(self):
        # Set font
        self.set_font('Helvetica', 'B', 10)
        # Title of the page
        self.set_text_color(100, 110, 120)
        self.cell(0, 10, 'AI Agent Engineering Reference Handbook', 0, 0, 'R')
        self.ln(12)
        # Draw a line below the header
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        # Go to 1.5 cm from bottom
        self.set_y(-15)
        # Set font
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_handbook():
    pdf = AgentHandbookPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(26, 54, 93) # Deep Blue
    pdf.cell(0, 15, 'AI Agent Engineering', 0, 1, 'L')
    
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(43, 108, 176) # Secondary Blue
    pdf.cell(0, 10, 'Core Concepts & Architectural Principles', 0, 1, 'L')
    pdf.ln(5)
    
    # Introduction
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    intro_text = (
        "Building AI agents requires transitioning from a linear model interaction paradigm (prompt -> response) "
        "to a dynamic closed-loop execution pattern (sense -> plan -> act). This handbook outlines the critical "
        "knowledge, patterns, and design patterns required to build reliable, production-ready AI agents."
    )
    pdf.multi_cell(0, 5, intro_text)
    pdf.ln(8)
    
    # Section 1: The Core Agent Architecture
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(44, 82, 130)
    pdf.cell(0, 8, '1. Core Agentic Loop: Sense-Plan-Act', 0, 1, 'L')
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    section1_text = (
        "Unlike standard pipelines, an agent operates within an execution loop:\n"
        "  - Sense (Input): The agent perceives the environment through user input or callback events.\n"
        "  - Plan (Reasoning): The LLM analyzes the inputs, sets goals, decomposes tasks, and selects tools.\n"
        "  - Act (Execution): The system runs the chosen tools (databases, external APIs, code execution) "
        "and feeds the results back into the system to restart the loop.\n\n"
        "Key architecture styles include:\n"
        "  1. ReAct (Reasoning + Acting): Alternating between 'Thought' (explaining what to do), 'Action' "
        "(specifying tool to run), and 'Observation' (result from the tool). This is the foundation of tool use.\n"
        "  2. Plan-and-Execute: Decomposing a complex user request into a step-by-step list of tasks, then "
        "executing each sequentially, checking progress and replanning if a task fails.\n"
        "  3. State Machines (Cyclic Graphs): Mapping agent flows as state nodes and transitions (conditional edges). "
        "This allows building loops (e.g., code writer node -> tester node -> loop back if test fails) and is "
        "the standard for robust agentic systems (e.g., LangGraph)."
    )
    pdf.multi_cell(0, 5, section1_text)
    pdf.ln(8)
    
    # Section 2: Tool Use / Function Calling
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(44, 82, 130)
    pdf.cell(0, 8, '2. Function Calling & Tool Execution', 0, 1, 'L')
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    section2_text = (
        "Tools allow LLMs to interact with the outside world. The framework flow is as follows:\n"
        "  - Schemas: You define tools using structured schemas (typically JSON Schema) detailing descriptions, "
        "argument names, and types. Excellent, clear descriptions are critical; the model uses them to decide "
        "when and how to call the tool.\n"
        "  - Model Output: The model outputs a structured payload (often a tool call containing tool name and arguments) "
        "instead of conversational text.\n"
        "  - System Interception: The application intercepts this output, executes the function locally, retrieves "
        "the outcome, formats it as a tool response message, and feeds it back to the LLM.\n\n"
        "Key Engineering Guardrails:\n"
        "  - Validation: Always validate JSON outputs and argument types before executing code to prevent system failures.\n"
        "  - Refusal & Recovery: If the LLM generates bad tool parameters, catch the exception and pass the error "
        "message back to the LLM as the observation, prompting it to self-correct."
    )
    pdf.multi_cell(0, 5, section2_text)
    pdf.ln(8)
    
    # Section 3: Memory Architectures
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(44, 82, 130)
    pdf.cell(0, 8, '3. Memory Systems for Agents', 0, 1, 'L')
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    section3_text = (
        "Memory is what allows agents to build context and personalizations across time. It is split into:\n"
        "  - Short-Term Memory: The chat history. Contains system prompts, previous user messages, tool calls, "
        "and tool observations. Because of context window limits, short-term memory is often compressed using "
        "sliding windows (discarding oldest messages) or summarization (LLM summarizing historical turns).\n"
        "  - Long-Term Memory: Stored external knowledge retrieved on demand. Powered by embedding vectors and "
        "Vector Databases (ChromaDB, Pinecone). The agent embeds the query, searches the DB, and injects "
        "the most similar documents as context (Retrieval-Augmented Generation or RAG).\n"
        "  - Episodic Memory: Recording step-by-step agent logs, decisions, and outcomes so the agent can learn "
        "from past trials (e.g., 'Last time I ran this tool it failed with Error X, so I will try Tool Y')."
    )
    pdf.multi_cell(0, 5, section3_text)
    pdf.ln(8)
    
    # Section 4: Multi-Agent Systems
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(44, 82, 130)
    pdf.cell(0, 8, '4. Multi-Agent Topologies', 0, 1, 'L')
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    section4_text = (
        "Complex tasks are best solved by specialized agents working together. Main design patterns include:\n"
        "  - Router Pattern: A single supervisor agent accepts user input and routes the task to one specialized "
        "worker agent (e.g., routing a bug fix to the coding agent vs a documentation fix to the docs agent).\n"
        "  - Supervisor/Orchestrator-Workers: An orchestrator coordinates a set of workers, assigns sub-tasks, "
        "gathers results, and synthesizes the final response. Perfect for workflows with step dependencies.\n"
        "  - Joint Collaboration (Mesh): Agents converse with each other in a shared room or thread, criticizing "
        "and refining the outputs (e.g., Code Generator -> Code Reviewer -> Security Evaluator -> Generator)."
    )
    pdf.multi_cell(0, 5, section4_text)
    pdf.ln(8)
    
    # Section 5: Production Best Practices
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(44, 82, 130)
    pdf.cell(0, 8, '5. Production Best Practices & Guardrails', 0, 1, 'L')
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    section5_text = (
        "  - Observability: Always log tracing data (e.g., using OpenTelemetry, LangSmith, or Langfuse). You must "
        "be able to trace every single turn, prompt template, tool input, output, and token cost.\n"
        "  - Guardrails: Implement strict input and output validation. Validate that user prompts are safe (no injections) "
        "and tool executions are restricted (e.g. read-only database connections, sandboxed file execution).\n"
        "  - Deterministic Fallbacks: When the agent hits maximum loop steps or is unable to solve a task, fall back "
        "gracefully to asking for human assistance instead of looping infinitely."
    )
    pdf.multi_cell(0, 5, section5_text)
    
    # Save PDF
    os.makedirs("books", exist_ok=True)
    pdf.output("books/Agent_Engineering_Handbook.pdf")
    print("Handbook PDF created successfully in 'books/Agent_Engineering_Handbook.pdf'")

if __name__ == "__main__":
    create_handbook()
