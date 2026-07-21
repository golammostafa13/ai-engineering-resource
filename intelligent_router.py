import os
import json
import time
from openai import OpenAI

# We will use Groq for this demonstration because it offers distinct small/large models for free.
# Make sure to set your GROQ_API_KEY environment variable or paste it below.
# (If you prefer to use local Ollama, you would need to pull a large model like llama3:70b which requires heavy hardware)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")

try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )
except Exception as e:
    print(f"Failed to initialize client: {e}")
    exit(1)

# ---------------------------------------------------------
# 1. DEFINE OUR AVAILABLE MODELS
# ---------------------------------------------------------
ROUTER_MODEL = "llama-3.1-8b-instant"  # Ultra-fast, used only to make the decision

MODELS = {
    "small": {
        "id": "llama-3.1-8b-instant",
        "cost_per_1m": 0.05, 
        "description": "Fast and cheap. Good for casual chat, translation, formatting, or simple facts."
    },
    "large": {
        "id": "llama-3.3-70b-versatile",
        "cost_per_1m": 0.50, # 10x more expensive
        "description": "Highly capable. Best for coding, deep reasoning, logic, and complex analysis."
    }
}

# ---------------------------------------------------------
# 2. BUILD THE ROUTER AGENT
# ---------------------------------------------------------
def determine_best_model(user_prompt: str) -> dict:
    """
    Acts as the Router Agent. Analyzes the prompt and returns a JSON decision.
    """
    system_instruction = """
    You are an intelligent API Router. Analyze the user prompt and decide which model to use.
    
    Choose 'small' if the task is:
    - Simple greetings or basic questions
    - Translation or grammar correction
    - Asking for a single known fact (e.g. capital of a country)
    
    Choose 'large' if the task is:
    - Writing, debugging, or analyzing code
    - Complex math or logic puzzles
    - Synthesizing information or multi-step reasoning
    
    Return ONLY valid JSON in the following format:
    {"model": "small" or "large", "reason": "brief explanation"}
    """
    
    print("🚦 [Router] Analyzing prompt complexity...")
    start_time = time.time()
    
    # We use response_format={"type": "json_object"} to force JSON output
    response = client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    decision_text = response.choices[0].message.content
    decision = json.loads(decision_text)
    decision['routing_time'] = time.time() - start_time
    return decision

# ---------------------------------------------------------
# 3. THE EXECUTION ENGINE
# ---------------------------------------------------------
def smart_chat(prompt: str):
    print("\n" + "="*60)
    print(f"👤 USER: {prompt}")
    print("="*60)
    
    # Step A: The Router decides
    decision = determine_best_model(prompt)
    chosen_size = decision.get("model", "large") # fallback to large if unsure
    chosen_model_id = MODELS[chosen_size]["id"]
    
    print(f"🧠 [Decision] Selected Model : {chosen_size.upper()} ({chosen_model_id})")
    print(f"📝 [Reason]   : {decision['reason']}")
    print(f"⏱️  [Overhead] : {decision['routing_time']:.2f} seconds")
    print("-" * 60)
    
    # Step B: Route to the selected model
    print(f"🚀 [Execution] Generating answer with {chosen_model_id}...")
    start_time = time.time()
    
    response = client.chat.completions.create(
        model=chosen_model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    gen_time = time.time() - start_time
    answer = response.choices[0].message.content
    
    print("\n🤖 AGENT RESPONSE:\n")
    print(answer)
    print("\n" + "-" * 60)
    
    # Step C: Show the impact
    print(f"📊 METRICS (Generation took {gen_time:.2f}s)")
    if chosen_size == 'small':
        print("✅ SAVINGS: You used the 10x cheaper model because high reasoning wasn't needed!")
    else:
        print("💡 QUALITY: You used the heavy model to ensure the logic/code was handled correctly.")
    print("="*60 + "\n")

# ---------------------------------------------------------
# 4. RUN TESTS
# ---------------------------------------------------------
if __name__ == "__main__":
    if GROQ_API_KEY == "your_groq_api_key_here":
        print("⚠️ Please set your GROQ_API_KEY at the top of the file to run this.")
    else:
        # Test 1: Simple task (Should route to SMALL)
        smart_chat("What is the capital city of Australia?")
        
        # Test 2: Complex task (Should route to LARGE)
        smart_chat("Write a Python script that uses multiprocessing to calculate the first 1000 prime numbers. Explain how the GIL affects this.")
