# 🧠 Build a GPT From Scratch — Step-by-Step Notes
*(Based on Andrej Karpathy's Zero to Hero Notebook)*

---

## STAGE 1: The Dataset — Tiny Shakespeare

```python
!wget https://raw.githubusercontent.com/karpathy/.../input.txt
```

The training data is **1.1 million characters** of Shakespeare text.

```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.
```

**Key insight:** This is not word-based. It is **character-by-character**. Every letter, space, newline, and punctuation mark counts as one unit.

---

## STAGE 2: Character-Level Tokenization (vocab_size = 65)

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)  # 65
```

The notebook finds every UNIQUE character in the file. There are exactly **65** unique characters:
```
\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
```

Then it creates a simple lookup table (dictionary):

```
CHARACTER → INTEGER (encoder)
'A' → 0
'B' → 1
'a' → 39
'b' → 40
...

INTEGER → CHARACTER (decoder)
0 → 'A'
1 → 'B'
...
```

**Example:**
```
encode("hii there") → [46, 47, 47, 1, 58, 46, 43, 56, 43]
decode([46, 47, 47, 1, 58, 46, 43, 56, 43]) → "hii there"
```

**Visual:**
```
TEXT:   h    i    i         t    h    e    r    e
         ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
TOKENS: 46   47   47    1   58   46   43   56   43
```

---

## STAGE 3: Train/Validation Split

```python
n = int(0.9 * len(data))   # 90% = training
train_data = data[:n]       # First 1,003,854 tokens
val_data   = data[n:]       # Last 111,540 tokens (never seen during training)
```

**Why split?** To check if the model is generalizing or just memorizing. If train loss is low but val loss is high → **overfitting**.

---

## STAGE 4: Blocks and Batches (The Fundamental Training Unit)

### block_size = 8 (context window)

When you take 9 characters from the data, you actually get **8 training examples** inside it:

```
Data:    [18, 47, 56, 57, 58, 1, 15, 47, 58]
          F    i   r   s   t      C   i   t

Input  →  Target
[18]         47       (Given 'F', predict 'i')
[18,47]      56       (Given 'Fi', predict 'r')
[18,47,56]   57       (Given 'Fir', predict 's')
[18,47,56,57] 58      (Given 'Firs', predict 't')
... and so on up to block_size=8
```

**One block trains the model on 8 different prediction tasks simultaneously.**

### batch_size = 4 (parallelism)

We process 4 independent blocks at the same time for efficiency:

```
BATCH SHAPE: (4, 8)

Row 0: [24, 43, 58,  5, 57,  1, 46, 43]   ← Block from position X
Row 1: [44, 53, 56,  1, 58, 46, 39, 58]   ← Block from position Y
Row 2: [52, 58,  1, 58, 46, 39, 58,  1]   ← Block from position Z
Row 3: [25, 17, 27, 10,  0, 21,  1, 54]   ← Block from position W

Total training examples in one step: 4 × 8 = 32 predictions
```

---

## STAGE 5: The Simple Bigram Language Model

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

### What is nn.Embedding(65, 65)?

It is a **lookup table** of shape 65×65. Each row represents one character. Each row contains 65 numbers (logits = raw scores for predicting the next character).

```
token_embedding_table:
       next_char_A  next_char_B  next_char_C ... (65 columns)
'A' → [  0.3,        -1.2,        0.8,      ... ]
'B' → [  -0.5,        2.1,       -0.3,      ... ]
'a' → [  1.2,        -0.1,        0.5,      ... ]
... (65 rows total)
```

When you input character `'a'` (index 39), it simply **looks up row 39** and returns 65 scores.

### Forward Pass (making a prediction):

```
Input token: 'a' (index 39)
     ↓
Lookup row 39 of embedding table
     ↓
logits = [0.1, -0.5, 0.8, 1.2, -0.3, ...]  ← 65 raw scores
     ↓
Softmax → probabilities = [0.08, 0.04, 0.16, 0.24, 0.05, ...]
     ↓
Sample → next character = 'e' (index with probability 0.24)
```

### Loss Calculation (Cross-Entropy):

```
Prediction:  [0.08, 0.04, 0.16, 0.24, 0.05, ...]  (our output)
Correct:     [0,    0,    0,    1,    0,    ...]   (target was 'e')

Cross-Entropy Loss = -log(probability of correct answer)
                   = -log(0.24) = 1.42

If model was random (65 chars, equal probability = 1/65):
Loss = -log(1/65) = 4.17   ← This matches the initial loss of 4.8786!
```

---

## STAGE 6: Training Loop

```python
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for steps in range(100):
    xb, yb = get_batch('train')     # Step 1: Get data
    logits, loss = m(xb, yb)        # Step 2: Forward pass
    optimizer.zero_grad()            # Step 3: Clear old gradients
    loss.backward()                  # Step 4: Backpropagation
    optimizer.step()                 # Step 5: Update weights
```

**Visual of one training step:**

```
[Batch of 32 tokens]
        ↓
[Forward Pass]  →  Predictions (logits)
        ↓
[Compare with Targets]  →  Loss = 4.87
        ↓
[Backward Pass]  →  Calculate gradients (which weights caused the error?)
        ↓
[Optimizer Step]  →  Nudge weights slightly to reduce loss
        ↓
Next step: Loss = 4.65... 4.40... 3.10... (getting lower)
```

After 100 steps: Loss ≈ 4.65 (still bad, needs more steps)

---

## STAGE 7: The Mathematical Trick — Self-Attention Setup

The Bigram model is limited: it only looks at **the last 1 token** to predict the next.

**Goal:** Each token should be able to "look back" at all previous tokens and learn which ones are relevant.

### The Lower-Triangular Matrix Trick

```python
tril = torch.tril(torch.ones(T, T))
```

For T=4, this creates:

```
tril =
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

**What does this mean?**
- Token at position 0 can only see itself
- Token at position 1 can see positions 0 and 1
- Token at position 2 can see positions 0, 1, and 2
- Token at position 3 can see all positions

This is **causal masking** — preventing tokens from "cheating" by looking at future tokens.

### Version 1: Simple Averaging (xbow)

```python
xbow[b,t] = torch.mean(x[b, :t+1], dim=0)
```

For each position t, average all vectors from 0 to t.

```
Position 0:  vector[0]
Position 1:  (vector[0] + vector[1]) / 2
Position 2:  (vector[0] + vector[1] + vector[2]) / 3
```

**Problem:** Every past token gets **equal weight**. But in language, some past words matter much more than others.

### Version 3: Masking + Softmax

```python
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
```

Step 1: Start with zeros, then fill future positions with -infinity:
```
[[0,   -inf, -inf, -inf],
 [0,    0,   -inf, -inf],
 [0,    0,    0,   -inf],
 [0,    0,    0,    0  ]]
```

Step 2: Apply Softmax (e^x / sum):
- e^0 = 1.0, e^(-inf) = 0.0

```
Result:
[[1.00,  0.00,  0.00,  0.00],
 [0.50,  0.50,  0.00,  0.00],
 [0.33,  0.33,  0.33,  0.00],
 [0.25,  0.25,  0.25,  0.25]]
```

This is just equal averaging — the weights are uniform. The KEY INSIGHT is that in real self-attention, these weights will be **learned**, not uniform.

---

## STAGE 8: Self-Attention — Keys, Queries, Values

This is the heart of the Transformer.

```python
key   = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
```

**Intuition using a library analogy:**

```
QUERY  = "What am I looking for?"
         (Each token broadcasts what it wants to know)

KEY    = "What information do I contain?"
         (Each token broadcasts what it offers)

VALUE  = "What is my actual content?"
         (What gets extracted if you look at this token)
```

### Attention Score Calculation:

```
Step 1: Each token produces a Query vector (q) and a Key vector (k)

Token 0: q0=[0.2, 0.5],  k0=[0.3, 0.1]
Token 1: q1=[0.8, 0.1],  k1=[0.7, 0.4]
Token 2: q2=[0.1, 0.9],  k2=[0.2, 0.8]

Step 2: Compute attention scores = q @ k^T (dot product)
        High score = "these two tokens are relevant to each other"

        score(token2, token0) = q2 · k0 = 0.1×0.3 + 0.9×0.1 = 0.12
        score(token2, token1) = q2 · k1 = 0.1×0.7 + 0.9×0.4 = 0.43

        Token 2 finds Token 1 more relevant (score 0.43 > 0.12)

Step 3: Mask future tokens (set to -inf), apply Softmax → probabilities

Step 4: Use these probabilities to weighted-average the Value vectors
        output[2] = 0.27 × v0 + 0.73 × v1
        (Token 2's final representation is mostly influenced by Token 1)
```

### Visual Flow of One Attention Head:

```
Input x (B, T, C=32)
        │
        ├──→ Key   Linear   → k (B, T, head_size=16)
        ├──→ Query Linear   → q (B, T, head_size=16)
        └──→ Value Linear   → v (B, T, head_size=16)

Attention weights:
  wei = q @ k^T                  (B, T, T)   [raw scores]
  wei = wei / sqrt(head_size)    [scale to keep variance stable]
  wei = masked_fill(-inf)        [block future tokens]
  wei = softmax(wei)             [normalize to probabilities]

Output:
  out = wei @ v                  (B, T, head_size)
```

### Why divide by sqrt(head_size)?

Without scaling, dot products grow large → Softmax becomes very "peaky" (one token gets all the weight, others get 0). Dividing by sqrt(head_size) keeps the variance stable.

```
WITHOUT scaling (too sharp):  [0.03, 0.00, 0.16, 0.00, 0.80]
WITH scaling    (spread out): [0.19, 0.14, 0.24, 0.14, 0.29]
```

---

## STAGE 9: Full GPT Architecture

```python
class BigramLanguageModel(nn.Module):
    def __init__(self):
        self.token_embedding_table    # char → 64-dim vector
        self.position_embedding_table # position → 64-dim vector
        self.blocks                   # 4× [Attention + FeedForward]
        self.ln_f                     # Final LayerNorm
        self.lm_head                  # 64-dim → 65 vocab logits
```

### Full Forward Pass:

```
Input: "First" → [18, 47, 56, 57, 58]

Step 1: Token Embedding
        18 → [0.2, -0.5, 0.8, ...]    (64-dim vector for 'F')
        47 → [0.1,  0.3, -0.1, ...]   (64-dim vector for 'i')
        ...

Step 2: Position Embedding (added ON TOP of token embedding)
        Position 0 → [0.01, 0.02, ...]
        Position 1 → [0.03, -0.01, ...]
        ...
        (Tells the model WHERE each token is, since Attention has no order)

Step 3: x = token_embedding + position_embedding

Step 4: Pass through 4 Transformer Blocks
        Each Block:
          → LayerNorm
          → Multi-Head Self-Attention (4 heads × 16 dims = 64 dims)
          → Residual connection: x = x + attention_output
          → LayerNorm
          → FeedForward (64 → 256 → 64 with ReLU)
          → Residual connection: x = x + ff_output

Step 5: Final LayerNorm

Step 6: Linear projection → 65 logits (one per character)

Step 7: Softmax → probabilities → sample next character
```

### Visual of One Transformer Block:

```
        x (input)
        │
        ├────────────────────────────┐
        ↓                            │  (residual / skip connection)
    LayerNorm                        │
        ↓                            │
  Self-Attention                     │
        ↓                            │
       (+) ←────────────────────────┘
        │
        ├────────────────────────────┐
        ↓                            │
    LayerNorm                        │
        ↓                            │
  Feed Forward (MLP)                 │
        ↓                            │
       (+) ←────────────────────────┘
        │
        ↓
    x (output, same shape)
```

**Why Residual Connections?**
Without them, gradients vanish through deep networks. Adding the original input `x` back ensures gradients flow directly backward, enabling training of deeper models.

---

## STAGE 10: Training Results

```
step 0:    train loss 4.4116  ← random (like flipping 65-sided dice)
step 500:  train loss 2.2970  ← learning character patterns
step 1000: train loss 2.1020  ← learning word patterns
step 2000: train loss 1.8834  ← learning sentence structure
step 5000: train loss 1.6635  ← generating Shakespeare-like text
```

**Final Generated Text (0.2M params, trained on Shakespeare only):**

```
FlY BOLINGLO:
Them thrumply towiter arts the
muscue rike begatt the sea it
What satell in rowers that some than othis Marrity.

LUCENTVO:
But userman these that, where can is not diesty rege...
```

Not perfect English, but it learned:
- ✅ Character names followed by colons
- ✅ Line breaks and spacing
- ✅ Approximate word lengths and patterns
- ✅ Some grammatical structure

---

## Summary: The Full Pipeline

```
Raw Text
    ↓
Character Tokenization (65 unique chars)
    ↓
Integer Encoding (each char → integer ID)
    ↓
Batched Training Blocks (batch=16, block=32)
    ↓
Token Embedding (ID → 64-dim vector)
    ↓
Positional Embedding (position → 64-dim vector)
    ↓
4× Transformer Blocks:
   [LayerNorm → Multi-Head Attention (4 heads) → Residual]
   [LayerNorm → FeedForward MLP → Residual]
    ↓
Final LayerNorm
    ↓
Linear Head (64 → 65 logits)
    ↓
Softmax → Probability Distribution over 65 characters
    ↓
Sample → Next Character
    ↓
Append to input → Repeat (autoregressive generation)
```

**Model size: 0.21 Million parameters**
*(GPT-3 has 175 Billion — this is 800,000× smaller)*
