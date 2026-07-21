import json

cells = [
    # Title
    {"cell_type":"markdown","metadata":{},"source":[
        "# 🤖 Module 5: Build a Real Agent (Tool Calling + RAG + Memory)\n\n",
        "This notebook combines everything:\n",
        "- **RAG** (Module 4) — ChromaDB knowledge base\n",
        "- **Tool Calling** (Module 3) — weather, calculator, web search\n",
        "- **Conversation Memory** — multi-turn chat\n",
        "- **Real LLM** — Groq (free API)\n\n",
        "> Get your free Groq key at: https://console.groq.com"
    ]},

    # Step 1: Install
    {"cell_type":"markdown","metadata":{},"source":["---\n## Step 1: Install Dependencies"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "!pip install -q sentence-transformers chromadb groq\n",
        "print('✅ Done')"
    ]},

    # Step 2: Imports
    {"cell_type":"markdown","metadata":{},"source":["---\n## Step 2: Imports & API Key"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "import json\n",
        "from groq import Groq\n",
        "from sentence_transformers import SentenceTransformer\n",
        "import chromadb\n\n",
        "GROQ_API_KEY = 'your_key_here'  # paste from console.groq.com\n",
        "MODEL = 'llama-3.1-8b-instant'\n\n",
        "client = Groq(api_key=GROQ_API_KEY)\n",
        "embedder = SentenceTransformer('all-MiniLM-L6-v2')\n",
        "print('✅ Ready')"
    ]},

    # Step 3: Knowledge Base (RAG)
    {"cell_type":"markdown","metadata":{},"source":["---\n## Step 3: Build the Knowledge Base (RAG Component)"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "# Company documents\n",
        "DOCS = [\n",
        "    {'id':'d1','source':'refund_policy.txt','text':'Customers can return products within 30 days. Items must be in original condition. Digital products cannot be returned.'},\n",
        "    {'id':'d2','source':'refund_policy.txt','text':'Refunds are processed in 5 to 7 business days. Contact support@company.com to initiate a return. Refunds go to the original payment method.'},\n",
        "    {'id':'d3','source':'shipping.txt','text':'Standard shipping takes 3 to 5 business days. Express shipping delivers in 1 to 2 days for an extra fee. Free shipping on orders above $50.'},\n",
        "    {'id':'d4','source':'shipping.txt','text':'International shipping takes 7 to 14 business days. Tracking info is emailed once the order ships. Orders are dispatched within 24 hours of payment.'},\n",
        "    {'id':'d5','source':'pricing.txt','text':'All prices include taxes. We offer a price match guarantee. Contact sales@company.com with proof of a lower price elsewhere.'},\n",
        "]\n\n",
        "# Store in ChromaDB\n",
        "chroma = chromadb.Client()\n",
        "col = chroma.create_collection('docs', metadata={'hnsw:space':'cosine'})\n",
        "col.add(\n",
        "    ids=[d['id'] for d in DOCS],\n",
        "    embeddings=embedder.encode([d['text'] for d in DOCS]).tolist(),\n",
        "    documents=[d['text'] for d in DOCS],\n",
        "    metadatas=[{'source':d['source']} for d in DOCS]\n",
        ")\n",
        "print(f'✅ Stored {col.count()} chunks in ChromaDB')"
    ]},

    # Step 4: Tool Functions
    {"cell_type":"markdown","metadata":{},"source":["---\n## Step 4: Define All Tools"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "def search_knowledge_base(query: str) -> str:\n",
        "    \"\"\"Search company documents (RAG).\"\"\"\n",
        "    qvec = embedder.encode(query).tolist()\n",
        "    res = col.query(query_embeddings=[qvec], n_results=2, include=['documents','metadatas'])\n",
        "    chunks = res['documents'][0]\n",
        "    sources = [m['source'] for m in res['metadatas'][0]]\n",
        "    return '\\n'.join([f'[{s}] {c}' for s, c in zip(sources, chunks)])\n\n",
        "def get_weather(city: str) -> str:\n",
        "    \"\"\"Simulated weather (replace with real API).\"\"\"\n",
        "    db = {'dhaka':'32C Humid','london':'15C Rainy','tokyo':'22C Clear','paris':'18C Cloudy'}\n",
        "    return db.get(city.lower(), f'No data for {city}')\n\n",
        "def calculate(a: float, b: float, operation: str) -> str:\n",
        "    \"\"\"Arithmetic calculator.\"\"\"\n",
        "    ops = {'add':a+b,'subtract':a-b,'multiply':a*b,'divide':a/b if b else 'Error: div by zero'}\n",
        "    return str(ops.get(operation, 'Unknown operation'))\n\n",
        "def search_web(query: str) -> str:\n",
        "    \"\"\"Simulated web search.\"\"\"\n",
        "    return f'Web search result for \"{query}\": This is a simulated result. In production, connect to a real search API.'\n\n",
        "TOOL_REGISTRY = {\n",
        "    'search_knowledge_base': search_knowledge_base,\n",
        "    'get_weather': get_weather,\n",
        "    'calculate': calculate,\n",
        "    'search_web': search_web,\n",
        "}\n\n",
        "def execute_tool(name, args):\n",
        "    if name not in TOOL_REGISTRY:\n",
        "        return f'Error: tool {name} not found'\n",
        "    try:\n",
        "        return str(TOOL_REGISTRY[name](**args))\n",
        "    except Exception as e:\n",
        "        return f'Error: {e}'\n\n",
        "# Quick test\n",
        "print('RAG test:', search_knowledge_base('refund time')[:80])\n",
        "print('Calc test:', calculate(150, 3.5, 'multiply'))"
    ]},

    # Step 5: Tool Schemas
    {"cell_type":"markdown","metadata":{},"source":["---\n## Step 5: Tool Schemas (What the LLM Reads)"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "TOOL_SCHEMAS = [\n",
        "  {'name':'search_knowledge_base',\n",
        "   'description':'Search company documents and policies. Use this for ANY company-specific question.',\n",
        "   'parameters':{'type':'object','properties':{'query':{'type':'string','description':'search query'}},'required':['query']}},\n",
        "  {'name':'get_weather',\n",
        "   'description':'Get current weather for a city.',\n",
        "   'parameters':{'type':'object','properties':{'city':{'type':'string','description':'city name'}},'required':['city']}},\n",
        "  {'name':'calculate',\n",
        "   'description':'Perform math: add, subtract, multiply, divide.',\n",
        "   'parameters':{'type':'object','properties':{\n",
        "     'a':{'type':'number'},'b':{'type':'number'},\n",
        "     'operation':{'type':'string','description':'add|subtract|multiply|divide'}\n",
        "   },'required':['a','b','operation']}},\n",
        "  {'name':'search_web',\n",
        "   'description':'Search the web for general information not in company documents.',\n",
        "   'parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}},\n",
        "]\n\n",
        "GROQ_TOOLS = [{'type':'function','function':s} for s in TOOL_SCHEMAS]\n",
        "print(f'✅ {len(TOOL_SCHEMAS)} tools registered')"
    ]},

    # Step 6: Agent Loop
    {"cell_type":"markdown","metadata":{},"source":["---\n## Step 6: The Complete Agent Loop"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "SYSTEM_PROMPT = \"\"\"You are a helpful assistant for TechCorp customer service.\n",
        "You have access to: search_knowledge_base, get_weather, calculate, search_web.\n",
        "Rules:\n",
        "1. ALWAYS use search_knowledge_base for company policy questions.\n",
        "2. Use calculate for any math.\n",
        "3. Never invent company information - retrieve it.\n",
        "4. Be concise and clear.\"\"\"\n\n",
        "class Agent:\n",
        "    def __init__(self):\n",
        "        self.history = [{'role':'system','content':SYSTEM_PROMPT}]\n\n",
        "    def chat(self, user_message: str, max_steps: int = 10) -> str:\n",
        "        \"\"\"Process one user message with full tool-calling loop.\"\"\"\n",
        "        print(f'\\n{\"=\"*60}')\n",
        "        print(f'USER: {user_message}')\n",
        "        self.history.append({'role':'user','content':user_message})\n\n",
        "        for step in range(max_steps):\n",
        "            response = client.chat.completions.create(\n",
        "                model=MODEL,\n",
        "                messages=self.history,\n",
        "                tools=GROQ_TOOLS,\n",
        "                tool_choice='auto',\n",
        "                temperature=0\n",
        "            )\n",
        "            msg = response.choices[0].message\n\n",
        "            # Final answer\n",
        "            if not msg.tool_calls:\n",
        "                self.history.append({'role':'assistant','content':msg.content})\n",
        "                print(f'AGENT: {msg.content}')\n",
        "                return msg.content\n\n",
        "            # Tool calls\n",
        "            self.history.append(msg)\n",
        "            for tc in msg.tool_calls:\n",
        "                name = tc.function.name\n",
        "                args = json.loads(tc.function.arguments)\n",
        "                print(f'  → calls {name}({args})')\n",
        "                result = execute_tool(name, args)\n",
        "                print(f'    result: {result[:100]}')\n",
        "                self.history.append({\n",
        "                    'role':'tool',\n",
        "                    'tool_call_id':tc.id,\n",
        "                    'content':result\n",
        "                })\n\n",
        "        return 'Error: max steps reached'\n\n",
        "    def reset(self):\n",
        "        \"\"\"Clear conversation history.\"\"\"\n",
        "        self.history = [{'role':'system','content':SYSTEM_PROMPT}]\n",
        "        print('✅ Memory cleared')\n\n",
        "print('✅ Agent class ready')"
    ]},

    # Step 7: Single tool tests
    {"cell_type":"markdown","metadata":{},"source":["---\n## Step 7: Test Single-Tool Queries"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "if GROQ_API_KEY == 'your_key_here':\n",
        "    print('⚠️  Paste your Groq API key in Step 2 to run this cell.')\n",
        "else:\n",
        "    agent = Agent()\n\n",
        "    # Test RAG\n",
        "    agent.chat('What is the refund policy?')\n\n",
        "    # Test calculator\n",
        "    agent.chat('What is 150 multiplied by 3.5?')\n\n",
        "    # Test weather\n",
        "    agent.chat('What is the weather in Tokyo?')"
    ]},

    # Step 8: Multi-tool test
    {"cell_type":"markdown","metadata":{},"source":["---\n## Step 8: Test Multi-Tool Query (The Real Power)"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "if GROQ_API_KEY == 'your_key_here':\n",
        "    print('⚠️  Paste your Groq API key in Step 2 to run this cell.')\n",
        "else:\n",
        "    agent.reset()\n\n",
        "    # This requires TWO tools: RAG + calculator\n",
        "    agent.chat('What is our refund policy and what is 89 times 12?')\n\n",
        "    # This requires TWO tools: RAG + weather\n",
        "    agent.chat('What is the shipping time to London and what is the weather there?')"
    ]},

    # Step 9: Multi-turn memory
    {"cell_type":"markdown","metadata":{},"source":["---\n## Step 9: Test Conversation Memory (Multi-Turn)"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "if GROQ_API_KEY == 'your_key_here':\n",
        "    print('⚠️  Paste your Groq API key in Step 2 to run this cell.')\n",
        "else:\n",
        "    agent.reset()\n\n",
        "    agent.chat('What is the refund policy?')\n",
        "    agent.chat('How long does that take?')          # agent knows 'that' = refund\n",
        "    agent.chat('And what about shipping time?')      # agent knows context = our company"
    ]},

    # Step 10: Summary
    {"cell_type":"markdown","metadata":{},"source":[
        "---\n## ✅ What You Built\n\n",
        "| Component | Role |\n",
        "|---|---|\n",
        "| ChromaDB + SentenceTransformer | Knowledge base (RAG) |\n",
        "| `get_weather`, `calculate`, `search_web` | External tool functions |\n",
        "| Tool schemas (JSON) | Tell the LLM what tools exist |\n",
        "| `execute_tool()` | Runs the right function safely |\n",
        "| `Agent` class | Holds memory + runs the loop |\n",
        "| Groq Llama-3 | The brain making all decisions |\n\n",
        "### This is a production-ready agent pattern.\n",
        "To make it truly production-ready, add:\n",
        "- Real weather API (OpenWeatherMap)\n",
        "- Real web search (SerpAPI, Tavily)\n",
        "- Persistent vector DB (not in-memory)\n",
        "- Error logging and observability\n\n",
        "**Next → Module 6: Multi-Agent Systems** (orchestrator + specialist workers)"
    ]},
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("module5_real_agent_colab.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
print("✅ Notebook 'module5_real_agent_colab.ipynb' generated!")
