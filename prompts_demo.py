def system_prompt_template(agent_name, capabilities):
    return f"""You are a helpful AI agent named {agent_name}.
Your primary capabilities are: {capabilities}.

When answering queries:
1. Be concise.
2. If you do not know the answer, say "I don't know".
3. Always explain your reasoning before stating the final answer.
"""

def user_prompt_template(user_query, context=""):
    if context:
        return f"Context:\n{context}\n\nQuery: {user_query}"
    return f"Query: {user_query}"

def run_prompts_demo():
    print("--- Prompt Engineering for Agents ---")
    
    # 1. System and User Prompt Composition
    sys_prompt = system_prompt_template(
        agent_name="DataMiner", 
        capabilities="Extracting structured information from raw text"
    )
    user_prompt = user_prompt_template(
        user_query="Extract the founder and year founded for Acme Corp from the text.",
        context="Acme Corp was started in 1985 by a visionary named Jane Doe."
    )
    
    print("[SYSTEM PROMPT]")
    print(sys_prompt)
    print("[USER PROMPT]")
    print(user_prompt)
    print("=" * 50)
    
    # 2. Demonstration of Few-Shot Prompting (to enforce JSON)
    print("\n--- Few-Shot Prompting Example (Enforcing JSON Output) ---")
    few_shot_prompt = """Task: Extract entities (Name, Age, City) and return as valid JSON list.

Example 1:
Input: John is 29 years old and lives in New York.
Output: {"name": "John", "age": 29, "city": "New York"}

Example 2:
Input: Sarah, age 34, from Berlin, loves hiking.
Output: {"name": "Sarah", "age": 34, "city": "Berlin"}

Input: Bob is 42, living in London.
Output:"""
    print(few_shot_prompt)
    print("[Model will output: {\"name\": \"Bob\", \"age\": 42, \"city\": \"London\"}]")
    print("=" * 50)

    # 3. Demonstration of Chain-of-Thought (CoT) Prompting
    print("\n--- Chain-of-Thought (CoT) Prompting ---")
    cot_prompt = """Solve the following word problem:
A basket has 3 apples. You add 5 more apples. You eat 2. How many apples are left?

Let's think step by step:
1. Start with 3 apples in the basket.
2. 5 apples are added: 3 + 5 = 8 apples.
3. 2 apples are eaten: 8 - 2 = 6 apples.
Therefore, the final answer is 6.

Solve this problem:
A box contains 10 blue marbles. You remove 4. You add 8 red marbles. You double the total number of marbles in the box. How many marbles are left in the box?

Let's think step by step:"""
    print(cot_prompt)
    print("[Model will output the step-by-step derivation before the final number, reducing errors.]")

if __name__ == "__main__":
    run_prompts_demo()
