import numpy as np

def calculate_lora_parameters(d_in, d_out, rank):
    # Full fine-tuning parameters to update the weight matrix W
    full_params = d_in * d_out
    
    # LoRA decomposition parameters (Matrix A and Matrix B)
    # A is (rank x d_in), B is (d_out x rank)
    lora_params = (rank * d_in) + (d_out * rank)
    
    saving_percentage = (1 - (lora_params / full_params)) * 100
    return full_params, lora_params, saving_percentage

def run_lora_demo():
    print("--- LoRA (Low-Rank Adaptation) Parameter Savings Demo ---")
    
    # Common hidden dimension sizes in LLMs
    # Llama 3 8B: hidden dimension is 4096
    d_in = 4096
    d_out = 4096
    
    # We choose a small rank, e.g., r = 8
    r = 8
    
    full, lora, savings = calculate_lora_parameters(d_in, d_out, r)
    
    print(f"Base Matrix Dimension: Input={d_in} -> Output={d_out}")
    print(f"Chosen LoRA Rank (r): {r}")
    print("-" * 50)
    print(f"Full Fine-Tuning Trainable Parameters: {full:,}")
    print(f"LoRA Trainable Parameters (A + B):    {lora:,}")
    print(f"Parameter Savings:                     {savings:.2f}%")
    
    # Let's show how rank affects parameter count
    print("\n--- Impact of Rank (r) on Parameter Counts ---")
    print(f"{'Rank (r)':<10} | {'Trainable Params':<18} | {'Savings %':<10}")
    print("-" * 46)
    for test_r in [1, 4, 8, 16, 32, 64]:
        _, l_params, pct = calculate_lora_parameters(d_in, d_out, test_r)
        print(f"{test_r:<10} | {l_params:<18,} | {pct:.2f}%")

    print("\n💡 Intuition: Instead of updating a huge 16.7M weight matrix, we update two smaller matrices")
    print("   (e.g., 8x4096 and 4096x8) which multiply together to approximate the weight updates.")

if __name__ == "__main__":
    run_lora_demo()
