import numpy as np

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits)) # stable softmax
    return exp_logits / np.sum(exp_logits)

def apply_temperature(logits, temperature):
    if temperature == 0:
        # Temperature of 0 is greedy search: put all probability on the max
        probs = np.zeros_like(logits)
        probs[np.argmax(logits)] = 1.0
        return probs
    return softmax(logits / temperature)

def apply_top_k(probs, k):
    # Get indices of top k probabilities
    top_k_indices = np.argsort(probs)[-k:]
    # Zero out everything else
    filtered_probs = np.zeros_like(probs)
    filtered_probs[top_k_indices] = probs[top_k_indices]
    # Re-normalize
    return filtered_probs / np.sum(filtered_probs)

def apply_top_p(probs, p):
    # Sort indices by probability in descending order
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Calculate cumulative probabilities
    cumulative_probs = np.cumsum(sorted_probs)
    
    # Find the cutoff index
    cutoff_idx = np.where(cumulative_probs >= p)[0][0]
    
    # Keep only indices up to the cutoff
    keep_indices = sorted_indices[:cutoff_idx + 1]
    
    filtered_probs = np.zeros_like(probs)
    filtered_probs[keep_indices] = probs[keep_indices]
    # Re-normalize
    return filtered_probs / np.sum(filtered_probs)

def run_decoding_demo():
    print("--- Decoding Strategies Simulator ---")
    
    # Vocabulary and raw logits predicted by the model for "The sky is"
    vocab = ["blue", "grey", "dark", "cloudy", "falling", "banana", "database"]
    raw_logits = np.array([6.0,    4.5,    4.0,    3.5,      1.0,       -2.0,     -5.0])
    
    base_probs = softmax(raw_logits)
    print("Base Probabilities (Temp=1.0):")
    for word, prob in zip(vocab, base_probs):
        print(f"  {word:<10} : {prob*100:5.2f}%")
        
    print("\n--- 1. Temperature Comparison ---")
    
    # Greedy (Temp = 0)
    greedy_probs = apply_temperature(raw_logits, temperature=0.0)
    print("Greedy Search (Temp=0.0):")
    for word, prob in zip(vocab, greedy_probs):
        if prob > 0:
            print(f"  {word:<10} : {prob*100:5.2f}%")
            
    # Low Temperature (Temp = 0.3)
    low_temp_probs = apply_temperature(raw_logits, temperature=0.3)
    print("\nLow Temp (Temp=0.3):")
    for word, prob in zip(vocab, low_temp_probs):
        print(f"  {word:<10} : {prob*100:5.2f}%")
        
    # High Temperature (Temp = 1.8)
    high_temp_probs = apply_temperature(raw_logits, temperature=1.8)
    print("\nHigh Temp (Temp=1.8):")
    for word, prob in zip(vocab, high_temp_probs):
        print(f"  {word:<10} : {prob*100:5.2f}%")

    print("\n--- 2. Top-K Sampling (K=3, Temp=1.0) ---")
    top_k_probs = apply_top_k(base_probs, k=3)
    for word, prob in zip(vocab, top_k_probs):
        print(f"  {word:<10} : {prob*100:5.2f}%")

    print("\n--- 3. Top-P Sampling (P=0.9, Temp=1.0) ---")
    top_p_probs = apply_top_p(base_probs, p=0.9)
    for word, prob in zip(vocab, top_p_probs):
        print(f"  {word:<10} : {prob*100:5.2f}%")

if __name__ == "__main__":
    run_decoding_demo()
