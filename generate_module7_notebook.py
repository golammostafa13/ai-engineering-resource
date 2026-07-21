import json

cells = [
    {"cell_type":"markdown","metadata":{},"source":[
        "# 🚦 Module 7 Capstone: Intelligent LLM Router\n\n",
        "Build an agent that automatically selects the best model for the job based on prompt complexity, saving cost and reducing latency.\n\n",
        "> Requires Groq API key from: https://console.groq.com"
    ]},

    {"cell_type":"markdown","metadata":{},"source":["---\n## Step 1: Install & Setup"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "!pip install -q groq\n",
        "import json\n",
        "import time\n",
        "from groq import Groq\n\n",
        "GROQ_API_KEY = 'your_key_here'  # Paste your key here\n",
        "client = Groq(api_key=GROQ_API_KEY)\n",
        "print('✅ Setup complete')"
    ]},

    {"cell_type":"markdown","metadata":{},"source":["---\n## Step 2: Define Models & Simulated Costs\n\nWe define our small and large models and give them simulated costs (per 1M tokens) to demonstrate the savings."]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "# The ultra-fast model used just for making the routing decision\n",
        "ROUTER_MODEL = 'llama-3.1-8b-instant'\n\n",
        "# The models we will route between\n",
        "MODELS = {\n",
        "    'small': {\n",
        "        'id': 'llama-3.1-8b-instant',\n",
        "        'cost_per_1m': 0.05,  # Simulated $0.05 per 1M tokens\n",
        "        'desc': 'Fast, cheap, good for simple tasks'\n",
        "    },\n",
        "    'large': {\n",
        "        'id': 'llama-3.3-70b-versatile',\n",
        "        'cost_per_1m': 0.50,  # Simulated $0.50 per 1M tokens (10x more expensive)\n",
        "        'desc': 'Highly capable, slower, good for complex reasoning'\n",
        "    }\n",
        "}\n",
        "print('✅ Models defined')"
    ]},

    {"cell_type":"markdown","metadata":{},"source":["---\n## Step 3: Build the Router Agent\n\nThis agent reads the prompt and outputs a JSON decision."]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "ROUTER_SYSTEM = \"\"\"\n",
        "Analyze the user's prompt and determine if it requires a 'small' model or a 'large' model.\n",
        "\n",
        "Route to 'small' for:\n",
        "- Greetings and basic conversation\n",
        "- Simple factual questions (e.g., capital of France, who is the president)\n",
        "- Translation of short sentences\n",
        "- Basic summarization\n",
        "\n",
        "Route to 'large' for:\n",
        "- Writing or debugging code\n",
        "- Complex logic puzzles or math problems\n",
        "- Creative writing or nuanced analysis\n",
        "- Multi-step reasoning\n",
        "\"\"\"\n\n",
        "# We use tool calling to force the router to output valid JSON\n",
        "ROUTER_TOOLS = [\n",
        "    {\n",
        "        'type': 'function',\n",
        "        'function': {\n",
        "            'name': 'route_prompt',\n",
        "            'description': 'Route the prompt to the appropriate model',\n",
        "            'parameters': {\n",
        "                'type': 'object',\n",
        "                'properties': {\n",
        "                    'model_choice': {'type': 'string', 'enum': ['small', 'large'], 'description': 'The model to use'},\n",
        "                    'reason': {'type': 'string', 'description': 'Why this model was chosen'}\n",
        "                },\n",
        "                'required': ['model_choice', 'reason']\n",
        "            }\n",
        "        }\n",
        "    }\n",
        "]\n\n",
        "def analyze_prompt(user_prompt: str) -> dict:\n",
        "    \"\"\"Asks the Router model which model to use.\"\"\"\n",
        "    print(f\"\\n[Router] Analyzing prompt complexity...\")\n",
        "    start_time = time.time()\n",
        "    \n",
        "    response = client.chat.completions.create(\n",
        "        model=ROUTER_MODEL,\n",
        "        messages=[\n",
        "            {'role': 'system', 'content': ROUTER_SYSTEM},\n",
        "            {'role': 'user', 'content': user_prompt}\n",
        "        ],\n",
        "        tools=ROUTER_TOOLS,\n",
        "        tool_choice={'type': 'function', 'function': {'name': 'route_prompt'}},\n",
        "        temperature=0\n",
        "    )\n",
        "    \n",
        "    router_time = time.time() - start_time\n",
        "    \n",
        "    # Parse the tool call arguments\n",
        "    tool_call = response.choices[0].message.tool_calls[0]\n",
        "    decision = json.loads(tool_call.function.arguments)\n",
        "    decision['router_time'] = router_time\n",
        "    \n",
        "    return decision\n\n",
        "print('✅ Router Agent ready')"
    ]},

    {"cell_type":"markdown","metadata":{},"source":["---\n## Step 4: Build the Smart Execution Engine\n\nThis function coordinates the router and the target model, executing the query and reporting the metrics."]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "def smart_chat(user_prompt: str):\n",
        "    print(f'\\n{\"=\"*60}')\n",
        "    print(f'USER PROMPT: \"{user_prompt}\"')\n",
        "    print(f'{\"=\"*60}')\n",
        "    \n",
        "    # 1. Ask the Router\n",
        "    decision = analyze_prompt(user_prompt)\n",
        "    model_key = decision['model_choice']\n",
        "    target_model_id = MODELS[model_key]['id']\n",
        "    \n",
        "    print(f\"[Router] Decision: Route to {model_key.upper()} model\")\n",
        "    print(f\"[Router] Reason: {decision['reason']}\")\n",
        "    print(f\"[Router] Routing overhead: {decision['router_time']:.2f} seconds\\n\")\n",
        "    \n",
        "    # 2. Execute on the chosen model\n",
        "    print(f\"[Engine] Generating response using {target_model_id}...\")\n",
        "    start_time = time.time()\n",
        "    \n",
        "    response = client.chat.completions.create(\n",
        "        model=target_model_id,\n",
        "        messages=[{'role': 'user', 'content': user_prompt}],\n",
        "        temperature=0.7\n",
        "    )\n",
        "    \n",
        "    gen_time = time.time() - start_time\n",
        "    answer = response.choices[0].message.content\n",
        "    tokens_used = response.usage.total_tokens\n",
        "    \n",
        "    print(f'\\nAGENT ANSWER:\\n{answer}\\n')\n",
        "    \n",
        "    # 3. Calculate simulated metrics\n",
        "    actual_cost = (tokens_used / 1_000_000) * MODELS[model_key]['cost_per_1m']\n",
        "    max_cost = (tokens_used / 1_000_000) * MODELS['large']['cost_per_1m']\n",
        "    savings = max_cost - actual_cost\n",
        "    \n",
        "    print(f'{\"-\"*60}')\n",
        "    print(f'📊 ROUTING METRICS')\n",
        "    print(f'Tokens used: {tokens_used}')\n",
        "    print(f'Generation time: {gen_time:.2f}s')\n",
        "    if model_key == 'small':\n",
        "        print(f'💰 Cost Savings: Saved ${savings:.6f} by not using the large model (10x cheaper!)')\n",
        "    else:\n",
        "        print(f'🧠 Complexity Required: Used large model for high reasoning quality.')\n",
        "    print(f'{\"=\"*60}\\n')\n\n",
        "print('✅ Smart Execution Engine ready')"
    ]},

    {"cell_type":"markdown","metadata":{},"source":["---\n## Step 5: Test Simple Queries (Routing to Small Model)"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "if GROQ_API_KEY == 'your_key_here':\n",
        "    print('⚠️ Paste your Groq API key in Step 1')\n",
        "else:\n",
        "    smart_chat(\"What is the capital of Japan? Just the name of the city.\")\n",
        "    smart_chat(\"Translate 'Hello, how are you' into French.\")"
    ]},

    {"cell_type":"markdown","metadata":{},"source":["---\n## Step 6: Test Complex Queries (Routing to Large Model)"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[
        "if GROQ_API_KEY == 'your_key_here':\n",
        "    print('⚠️ Paste your Groq API key in Step 1')\n",
        "else:\n",
        "    smart_chat(\"Write a Python script using asyncio that fetches data from 3 URLs concurrently, handles timeouts gracefully, and returns a consolidated JSON object. Include type hints.\")\n",
        "    smart_chat(\"A farmer has 10 sheep, all but 9 die. How many are left? Explain your reasoning step by step.\")"
    ]},

    {"cell_type":"markdown","metadata":{},"source":[
        "---\n## ✅ What You Built\n\n",
        "You have built a production-grade **LLM Router**.\n\n",
        "Instead of hardcoding a single expensive model for every request, your system dynamically scales its intelligence based on the task.\n",
        "- For simple tasks, it routes to a tiny, fast model (saving 90% of the cost).\n",
        "- For hard tasks, it routes to a massive model (ensuring high quality).\n\n",
        "**This is how real-world AI applications stay fast and profitable.**"
    ]}
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("module7_router_agent_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=2)
print("✅ Notebook 'module7_router_agent_colab.ipynb' generated!")
