import tiktoken

def run_tokenization_demo():
    # Load the standard encoder used by GPT-4 and GPT-4o (o200k_base) or GPT-4/GPT-3.5 (cl100k_base)
    # We will use cl100k_base (used by GPT-4 and GPT-3.5-turbo)
    encoding = tiktoken.get_encoding("cl100k_base")
    
    text = "Building AI Agents is fun! Antigravity rules."
    
    # 1. Encode text to token IDs
    token_ids = encoding.encode(text)
    print(f"Original Text: '{text}'")
    print(f"Token IDs:      {token_ids}")
    print(f"Token Count:    {len(token_ids)}\n")
    
    # 2. Decode each token ID back to its string representation
    print("--- Individual Tokens ---")
    for token_id in token_ids:
        # We decode each individual token ID (it returns bytes, so we decode to string)
        token_bytes = encoding.decode_bytes([token_id])
        token_str = token_bytes.decode('utf-8', errors='replace')
        print(f"ID: {token_id:<6} -> String representation: {repr(token_str)}")
        
    print("\n--- Why Tokenization matters for Agents (Examples) ---")
    # Spaces matter!
    text1 = "hello"
    text2 = " hello"
    print(f"'{text1}' -> Tokens: {encoding.encode(text1)}")
    print(f"'{text2}' -> Tokens: {encoding.encode(text2)} (Notice how the space changes the starting token!)")
    
    # Capitalization matters!
    cap1 = "agent"
    cap2 = "Agent"
    print(f"'{cap1}' -> Tokens: {encoding.encode(cap1)}")
    print(f"'{cap2}' -> Tokens: {encoding.encode(cap2)}")

if __name__ == "__main__":
    run_tokenization_demo()
