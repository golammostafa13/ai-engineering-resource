import numpy as np

def cosine_similarity(v1, v2):
    # Formula: (A . B) / (||A|| * ||B||)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def run_embeddings_demo():
    print("--- Semantic Representation & Similarity Demo ---")
    
    # Imagine we have an embedding model that maps sentences to a 3-dimensional space.
    # Dimensions represents: [Pets/Animals, Technology/Computers, Sports/Games]
    
    # 1. "I love playing with my golden retriever." -> High in Animals
    v_dog = np.array([0.9, 0.1, 0.0])
    
    # 2. "A puppy is the best companion." -> High in Animals
    v_puppy = np.array([0.85, 0.05, 0.1])
    
    # 3. "I am building a high-performance database cluster." -> High in Technology
    v_db = np.array([0.0, 0.95, 0.05])
    
    # 4. "Our team scored three goals in the football match." -> High in Sports
    v_sports = np.array([0.1, 0.1, 0.85])
    
    # Compute similarities
    sim_dog_puppy = cosine_similarity(v_dog, v_puppy)
    sim_dog_db = cosine_similarity(v_dog, v_db)
    sim_dog_sports = cosine_similarity(v_dog, v_sports)
    
    print(f"Dog Vector:   {v_dog}")
    print(f"Puppy Vector: {v_puppy}")
    print(f"DB Vector:    {v_db}")
    print(f"Sport Vector: {v_sports}\n")
    
    print(f"Similarity (Dog vs Puppy):      {sim_dog_puppy:.4f}  <- Very similar (both animal-related)")
    print(f"Similarity (Dog vs Database):   {sim_dog_db:.4f}  <- Not similar")
    print(f"Similarity (Dog vs Sports):     {sim_dog_sports:.4f}  <- Not similar")

if __name__ == "__main__":
    run_embeddings_demo()
