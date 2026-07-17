import numpy as np

def compute_dpo_loss(beta, model_logprobs_w, model_logprobs_l, ref_logprobs_w, ref_logprobs_l):
    """
    Computes a simplified DPO loss.
    w = winning (preferred) response
    l = losing (dispreferred) response
    
    Formula:
        loss = -log(sigmoid(beta * ( (model_log_w - ref_log_w) - (model_log_l - ref_log_l) )))
    """
    # Calculate the log-ratio of the model's probabilities relative to the reference model
    ratio_w = model_logprobs_w - ref_logprobs_w
    ratio_l = model_logprobs_l - ref_logprobs_l
    
    # DPO objective: we want ratio_w to be much higher than ratio_l
    difference = beta * (ratio_w - ratio_l)
    
    # Sigmoid function
    prob = 1.0 / (1.0 + np.exp(-difference))
    
    # Loss is negative log of the sigmoid probability
    loss = -np.log(prob)
    return loss, prob

def run_alignment_demo():
    print("--- DPO (Direct Preference Optimization) Simulation ---")
    
    # Scenario: A user asks: "Write a function to delete a directory."
    # Preferred output (w): A safe function that checks permissions and prompts the user first.
    # Dispreferred output (l): A reckless "rm -rf" style function that deletes immediately without validation.
    
    # Let's define the log-probabilities of these answers under a Reference Model
    # Both are moderately likely under a general base model
    ref_logprob_w = -2.5
    ref_logprob_l = -2.8
    
    # Case 1: Model A is starting to align (model values are close to reference)
    model_a_logprob_w = -2.4  # Slightly more likely
    model_a_logprob_l = -2.9  # Slightly less likely
    
    # Case 2: Model B is highly aligned (model values are heavily optimized for the preferred choice)
    model_b_logprob_w = -1.2  # Highly likely
    model_b_logprob_l = -4.5  # Highly unlikely
    
    beta = 0.5  # Scaling hyperparameter (commonly between 0.1 and 0.5)
    
    loss_a, prob_a = compute_dpo_loss(beta, model_a_logprob_w, model_a_logprob_l, ref_logprob_w, ref_logprob_l)
    loss_b, prob_b = compute_dpo_loss(beta, model_b_logprob_w, model_b_logprob_l, ref_logprob_w, ref_logprob_l)
    
    print(f"Beta (regularization scale): {beta}")
    print("-" * 50)
    print("Model A (Starting alignment):")
    print(f"  Log-Prob of Safe Output (w):   {model_a_logprob_w}")
    print(f"  Log-Prob of Dangerous Output (l): {model_a_logprob_l}")
    print(f"  Calculated Loss:               {loss_a:.4f}")
    print(f"  Probability of picking 'w':    {prob_a * 100:.2f}%")
    
    print("\nModel B (Highly aligned):")
    print(f"  Log-Prob of Safe Output (w):   {model_b_logprob_w}")
    print(f"  Log-Prob of Dangerous Output (l): {model_b_logprob_l}")
    print(f"  Calculated Loss:               {loss_b:.4f}")
    print(f"  Probability of picking 'w':    {prob_b * 100:.2f}%")
    
    print("\n💡 Intuition: DPO loss decreases as the model increases the likelihood of the preferred response")
    print("   and decreases the likelihood of the dispreferred response, relative to the starting model.")

if __name__ == "__main__":
    run_alignment_demo()
