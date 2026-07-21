import os
import sys
from openai import OpenAI

# 1. Setup the client to point to Local Ollama
# Ollama hosts an OpenAI-compatible API on port 11434 by default.
try:
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama' # Required by the client, but unused by Ollama
    )
except ImportError:
    print("Error: The 'openai' library is not installed.")
    print("Run: pip install openai")
    sys.exit(1)

# Set the model you pulled (e.g. qwen2.5-coder:7b or qwen2.5-coder:1.5b)
# We will use 1.5b in this script as a lightweight default, but you should change it 
# to whatever you pulled!
MODEL_NAME = "qwen2.5-coder:1.5b" 

def review_code(filepath: str):
    """Reads a file and asks the local LLM to review it."""
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"❌ Error: File '{filepath}' not found.")
        return

    # Read the code
    with open(filepath, 'r') as file:
        code_content = file.read()

    print(f"🔍 Reading '{filepath}'...")
    print(f"🤖 Sending to local model ({MODEL_NAME}) for review...")

    # The prompt instructs the model how to act
    system_prompt = """
    You are an expert, senior software engineer.
    Review the code provided by the user. 
    1. Identify any bugs or security issues.
    2. Suggest improvements for readability and performance.
    3. Output the refactored code block at the end.
    Keep your explanation concise.
    """

    try:
        # Call the local model
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the code:\n\n```python\n{code_content}\n```"}
            ],
            temperature=0.2, # Low temperature for analytical coding tasks
        )
        
        print("\n" + "="*50)
        print("💡 AI REVIEW & SUGGESTIONS")
        print("="*50)
        print(response.choices[0].message.content)
        print("="*50)

    except Exception as e:
        print(f"\n❌ Failed to connect to Ollama.")
        print(f"Error details: {e}")
        print("\nMake sure:")
        print("1. Ollama is installed (https://ollama.com)")
        print(f"2. You have pulled the model by running: ollama pull {MODEL_NAME}")
        print("3. Ollama is currently running in the background.")

if __name__ == "__main__":
    # If a file is passed via CLI, use it. Otherwise, use our dummy test file.
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = "bad_code.py"
        print("No file specified. Defaulting to 'bad_code.py'...")
        
    review_code(target_file)
