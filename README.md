# AI Agent Engineering Learning Roadmap

> Goal: Become a production-level AI Engineer capable of building intelligent agents, RAG systems, MCP servers, and multi-agent applications.

---

# 1. LLM Fundamentals

## Goal

- Understand how LLMs work internally.
- Learn tokens, embeddings, transformers, context windows, inference, hallucinations, prompting.

---

## Books (books folder)

### Hands-On Large Language Models
https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/

---

## Videos

### Andrej Karpathy: Intro to Large Language Models
https://www.youtube.com/watch?v=zjkBMFhNj_g

### Deep Dive into LLMs like ChatGPT
https://www.youtube.com/watch?v=7xTGNNLPyMI

### Let's build GPT: from scratch, in code, spelled out.
https://www.youtube.com/watch?v=kCc8FmEb1nY

### Let's reproduce GPT-2 (124M)
https://www.youtube.com/watch?v=l8pRSuU81PU

### Build GPT Tokenizer
https://www.youtube.com/watch?v=zduSFxRajkE

---

## Courses

### DeepLearning.AI - ChatGPT Prompt Engineering
https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/

### DeepLearning.AI - Building Systems with ChatGPT
https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/

### DeepLearning.AI - LLMOps
https://www.deeplearning.ai/short-courses/llmops/

---

## Practice

- PDF Chatbot
- Website Summarizer
- Code Explainer

---

# 2. Tool Calling / Function Calling

## Goal

Teach LLMs to call external APIs and tools.

---

## Official Documentation

### OpenAI Function Calling
https://platform.openai.com/docs/guides/function-calling

### Anthropic Tool Use
https://docs.anthropic.com/en/docs/build-with-claude/tool-use

### Google Gemini Function Calling
https://ai.google.dev/gemini-api/docs/function-calling

---

## Videos

### OpenAI Function Calling Playlist
https://www.youtube.com/results?search_query=openai+function+calling

---

## Build

- Weather Assistant
- SQL Agent
- Calculator Agent
- Email Sender

---

# 3. RAG (Retrieval-Augmented Generation)

## Goal

Learn how AI can answer using private documents.

---

## Learn

- Chunking
- Embeddings
- Vector Database
- Similarity Search
- Hybrid Search
- Reranking

---

## Documentation

### Pinecone Learn
https://www.pinecone.io/learn/

### Pinecone Documentation
https://docs.pinecone.io/

### Cohere RAG Guide
https://docs.cohere.com/

### Weaviate Academy
https://academy.weaviate.io/

---

## Books

Continue:

Hands-On Large Language Models

(RAG Chapters)

---

## Build

- Chat with PDFs
- Company Knowledge Base
- Banking Policy Assistant

---

# 4. Memory

## Goal

Allow AI to remember users and previous conversations.

---

## Learn

- Short-term Memory
- Long-term Memory
- Semantic Memory
- Episodic Memory
- Memory Compression

---

## Documentation

### LangChain Memory
https://python.langchain.com/docs/how_to/chatbots_memory/

### OpenAI Conversation Guide
https://platform.openai.com/docs/guides/conversation-state

---

## Build

- Personal AI Assistant
- AI Diary
- Customer Support Assistant

---

# 5. MCP (Model Context Protocol)

## Goal

Learn how AI securely connects to external tools.

---

## Official

### MCP Website
https://modelcontextprotocol.io/

### MCP Specification
https://spec.modelcontextprotocol.io/

### MCP GitHub
https://github.com/modelcontextprotocol

---

## Videos

### MCP YouTube Search
https://www.youtube.com/results?search_query=model+context+protocol

---

## Build

- Filesystem MCP Server
- PostgreSQL MCP Server
- Banking API MCP Server

---

# 6. Agent Frameworks

> Learn frameworks AFTER understanding how agents work.

---

## LangGraph

Documentation

https://langchain-ai.github.io/langgraph/

GitHub

https://github.com/langchain-ai/langgraph

---

## PydanticAI

Documentation

https://ai.pydantic.dev/

GitHub

https://github.com/pydantic/pydantic-ai

---

## CrewAI

Documentation

https://docs.crewai.com/

GitHub

https://github.com/crewAIInc/crewAI

---

## AutoGen

Documentation

https://microsoft.github.io/autogen/

GitHub

https://github.com/microsoft/autogen

---

## Build

- Research Agent
- Coding Agent
- Meeting Assistant

---

# 7. Multi-Agent Systems

## Goal

Multiple AI agents collaborating together.

---

## Learn

- Planner
- Researcher
- Reviewer
- Critic
- Executor

---

## Papers

### AutoGen Paper
https://arxiv.org/abs/2308.08155

### CAMEL
https://arxiv.org/abs/2303.17760

### MetaGPT
https://arxiv.org/abs/2308.00352

---

## Documentation

### AutoGen
https://microsoft.github.io/autogen/

### CrewAI
https://docs.crewai.com/

---

## Build

- AI Software Company
- AI Security Team
- AI Research Team

---

# 8. Deployment

## Docker

https://docs.docker.com/

---

## FastAPI

https://fastapi.tiangolo.com/

---

## AWS

https://skillbuilder.aws/

https://docs.aws.amazon.com/

---

## Build

Deploy every project.

---

# 9. Bonus Topics

## Vector Databases

- Pinecone
- Weaviate
- Qdrant
- Chroma

---

## Observability

LangSmith

https://www.langchain.com/langsmith

OpenTelemetry

https://opentelemetry.io/

---

## Evaluation

Ragas

https://docs.ragas.io/

DeepEval

https://github.com/confident-ai/deepeval

---

# 10. GitHub Repositories

## awesome-llm

https://github.com/Hannibal046/Awesome-LLM

---

## awesome-generative-ai

https://github.com/steven2358/awesome-generative-ai

---

## awesome-ai-agents

https://github.com/e2b-dev/awesome-ai-agents

---

# 11. Newsletters

### Latent Space
https://www.latent.space/

### Ben's Bites
https://www.bensbites.com/

### The Batch
https://www.deeplearning.ai/the-batch/

---

# 12. YouTube Channels

## Andrej Karpathy
https://www.youtube.com/@AndrejKarpathy

## DeepLearning.AI
https://www.youtube.com/@Deeplearningai

## AssemblyAI
https://www.youtube.com/@AssemblyAI

## IBM Technology
https://www.youtube.com/@IBMTechnology

## LangChain
https://www.youtube.com/@LangChain

## Anthropic
https://www.youtube.com/@AnthropicAI

---

# Final Project

Build an AI Banking Copilot featuring:

- Tool Calling
- RAG
- Memory
- MCP
- Multi-Agent Workflow
- PostgreSQL
- Redis
- Docker
- AWS Deployment

---

# Suggested Learning Order

- [ ] LLM Fundamentals
- [ ] Tool Calling
- [ ] RAG
- [ ] Memory
- [ ] MCP
- [ ] Agent Frameworks
- [ ] Multi-Agent Systems
- [ ] Deployment
- [ ] Final Project
