# 🔍 Attention Is All You Need — Every Inch Explained

---

## THE BIG PICTURE FIRST

The original "Attention Is All You Need" paper (2017) introduced this formula:

```
Attention(Q, K, V) = Softmax( Q @ K^T / sqrt(d_k) ) @ V
```

This one formula is the entire engine of GPT. Everything else is just wiring around it.
Let's build it from scratch, step by step with real numbers.

---

## SETUP: Our Tiny Example

We have a 4-word sentence: `"the cat sat"`

After tokenization and embedding, each word becomes a vector of size C=4:

```
Input Matrix X  (shape: 3 tokens × 4 dimensions)

         dim0   dim1   dim2   dim3
"the" → [ 1.0,   0.0,   1.0,   0.0 ]   ← row 0
"cat" → [ 0.0,   2.0,   0.0,   2.0 ]   ← row 1
"sat" → [ 1.0,   1.0,   1.0,   1.0 ]   ← row 2
```

We will use head_size = 3, so each W matrix is (4 × 3).

---

## STEP 1: The Three Learned Weight Matrices (W_Q, W_K, W_V)

These are initialized randomly at the start of training.
Imagine they happened to be learned as:

```
W_Q (4 × 3):         W_K (4 × 3):         W_V (4 × 3):
[1, 0, 0]            [1, 0, 1]            [0, 1, 0]
[1, 0, 0]            [0, 1, 0]            [1, 0, 0]
[0, 0, 1]            [1, 0, 0]            [0, 1, 1]
[0, 1, 0]            [0, 1, 0]            [1, 0, 0]
```

---

## STEP 2: Compute Q, K, V by Matrix Multiplication

```
Q = X @ W_Q      K = X @ W_K      V = X @ W_V
```

**Computing Q = X @ W_Q:**

```
X (3×4):                W_Q (4×3):           Q (3×3):
[ 1, 0, 1, 0 ]                              ?
[ 0, 2, 0, 2 ]    @    [W_Q matrix]    =    ?
[ 1, 1, 1, 1 ]                              ?
```

Row by row calculation for Q:

```
Q row 0 ("the"):  [1,0,1,0] @ W_Q = [1+0, 0+0, 0+0] = [1, 0, 0]
Q row 1 ("cat"):  [0,2,0,2] @ W_Q = [0+2, 0+2, 0+0] = [2, 2, 0]
Q row 2 ("sat"):  [1,1,1,1] @ W_Q = [1+1, 0+1, 0+1] = [2, 1, 1]

Q (Query matrix, 3×3):
         q_dim0  q_dim1  q_dim2
"the" → [  1,      0,      0  ]   ← "What is 'the' looking for?"
"cat" → [  2,      2,      0  ]   ← "What is 'cat' looking for?"
"sat" → [  2,      1,      1  ]   ← "What is 'sat' looking for?"
```

Similarly computed (assume):

```
K (Key matrix, 3×3):
         k_dim0  k_dim1  k_dim2
"the" → [  1,      0,      0  ]   ← "What does 'the' advertise?"
"cat" → [  2,      2,      0  ]   ← "What does 'cat' advertise?"
"sat" → [  2,      1,      1  ]   ← "What does 'sat' advertise?"

V (Value matrix, 3×3):
         v_dim0  v_dim1  v_dim2
"the" → [  0,      1,      0  ]   ← "The actual content of 'the'"
"cat" → [  2,      0,      2  ]   ← "The actual content of 'cat'"
"sat" → [  2,      2,      2  ]   ← "The actual content of 'sat'"
```

---

## STEP 3: Compute Raw Attention Scores = Q @ K^T

**This is the dot product step — the core of attention.**

K^T means transpose of K (flip rows and columns):

```
K^T (3×3):
      "the"  "cat"  "sat"
dim0 [  1,     2,     2  ]
dim1 [  0,     2,     1  ]
dim2 [  0,     0,     1  ]
```

Now compute Q @ K^T:

```
Q (3×3) @ K^T (3×3) = Raw Scores (3×3)
```

Calculating each cell (dot product of each query with each key):

```
Score("the" queries "the"):  [1,0,0]·[1,0,0] = 1+0+0 = 1
Score("the" queries "cat"):  [1,0,0]·[2,2,0] = 2+0+0 = 2
Score("the" queries "sat"):  [1,0,0]·[2,1,1] = 2+0+0 = 2

Score("cat" queries "the"):  [2,2,0]·[1,0,0] = 2+0+0 = 2
Score("cat" queries "cat"):  [2,2,0]·[2,2,0] = 4+4+0 = 8
Score("cat" queries "sat"):  [2,2,0]·[2,1,1] = 4+2+0 = 6

Score("sat" queries "the"):  [2,1,1]·[1,0,0] = 2+0+0 = 2
Score("sat" queries "cat"):  [2,1,1]·[2,2,0] = 4+2+0 = 6
Score("sat" queries "sat"):  [2,1,1]·[2,1,1] = 4+1+1 = 6
```

**Raw Score Matrix (3×3):**

```
             ← keys →
            "the"  "cat"  "sat"
↑  "the" → [  1,     2,     2  ]   ← "the" finds "cat" and "sat" equally relevant
q  "cat" → [  2,     8,     6  ]   ← "cat" strongly finds "cat" itself (8!)
u  "sat" → [  2,     6,     6  ]   ← "sat" finds "cat" and "sat" equally relevant
e
r
i
e
s
```

**Intuition:** High score = "these two tokens are strongly related."
"cat" queries "cat" → score 8 (very high, a word is similar to itself).
"cat" queries "the" → score 2 (low, articles are not very relevant to nouns).

---

## STEP 4: Scale by 1/sqrt(head_size)

head_size = 3, so sqrt(3) ≈ 1.73

```
Scaled Scores = Raw Scores / 1.73

[ 1/1.73,  2/1.73,  2/1.73 ]     [0.58,  1.16,  1.16]
[ 2/1.73,  8/1.73,  6/1.73 ]  =  [1.16,  4.62,  3.46]
[ 2/1.73,  6/1.73,  6/1.73 ]     [1.16,  3.46,  3.46]
```

**Why scale?** Without scaling, dot products grow very large when vectors are high-dimensional. Large scores → Softmax becomes too "sharp" (one token gets 99% attention, rest get nearly 0). Scaling keeps the distribution spread out so gradients flow well during training.

```
WITHOUT scaling (scores too large, very sharp):
  "cat" → [0.00,  0.99,  0.01]   ← almost ignores everything except itself

WITH scaling (healthy spread):
  "cat" → [0.01,  0.73,  0.26]   ← still prefers itself but considers others
```

---

## STEP 5: Causal Masking (Decoder only)

In a language model, token at position t cannot see tokens at position t+1, t+2...
We mask future positions with -infinity.

For our 3 tokens:

```
BEFORE masking:           AFTER masking (upper triangle → -inf):
[0.58,  1.16,  1.16]      [0.58,  -inf,  -inf]
[1.16,  4.62,  3.46]  →   [1.16,   4.62,  -inf]
[1.16,  3.46,  3.46]      [1.16,   3.46,  3.46]

"the" (pos 0): can only see itself
"cat" (pos 1): can see "the" and "cat" but NOT "sat"
"sat" (pos 2): can see all three
```

---

## STEP 6: Apply Softmax (Row by Row)

Convert each row to probabilities. Each row must sum to 1.0.

```
Softmax formula for each value: e^x / (e^x1 + e^x2 + ...)
```

**Row 0 ("the"):** [0.58, -inf, -inf]
```
e^0.58 = 1.79,   e^(-inf) = 0,   e^(-inf) = 0
Sum = 1.79
Probabilities: [1.79/1.79,  0/1.79,  0/1.79] = [1.00, 0.00, 0.00]
```
→ "the" pays 100% attention to itself (no other tokens visible yet)

**Row 1 ("cat"):** [1.16, 4.62, -inf]
```
e^1.16 = 3.19,   e^4.62 = 101.5,   e^(-inf) = 0
Sum = 104.69
Probabilities: [3.19/104.69,  101.5/104.69,  0] = [0.03, 0.97, 0.00]
```
→ "cat" pays 97% attention to itself, 3% to "the"

**Row 2 ("sat"):** [1.16, 3.46, 3.46]
```
e^1.16 = 3.19,   e^3.46 = 31.82,   e^3.46 = 31.82
Sum = 66.83
Probabilities: [0.05, 0.48, 0.48]
```
→ "sat" pays equal attention to "cat" and itself, with small attention to "the"

**Final Attention Weight Matrix:**

```
              "the"   "cat"   "sat"
"the"  →  [  1.00,   0.00,   0.00 ]
"cat"  →  [  0.03,   0.97,   0.00 ]
"sat"  →  [  0.05,   0.48,   0.48 ]
```

Each row tells us: **"When processing THIS word, how much do I look at EACH past word?"**

---

## STEP 7: Weighted Sum of Values

Now we use these weights to mix the Value vectors.

```
Output = Attention Weights @ V

Attention Weights (3×3):        V (3×3):             Output (3×3):
[1.00,  0.00,  0.00]            [0, 1, 0]
[0.03,  0.97,  0.00]    @       [2, 0, 2]     =       ?
[0.05,  0.48,  0.48]            [2, 2, 2]
```

**Computing Output row by row:**

```
Output for "the":
  1.00 × [0,1,0]  +  0.00 × [2,0,2]  +  0.00 × [2,2,2]
= [0, 1, 0]
→ "the" only used its own Value vector (100%)

Output for "cat":
  0.03 × [0,1,0]  +  0.97 × [2,0,2]  +  0.00 × [2,2,2]
= [0+1.94+0,  0.03+0+0,  0+1.94+0]
= [1.94, 0.03, 1.94]
→ "cat" mostly used its own Value, tiny bit from "the"

Output for "sat":
  0.05 × [0,1,0]  +  0.48 × [2,0,2]  +  0.48 × [2,2,2]
= [0+0.96+0.96,  0.05+0+0.96,  0+0.96+0.96]
= [1.92, 1.01, 1.92]
→ "sat" blended "cat" and itself equally, small bit from "the"
```

**Final Output Matrix:**

```
         out0   out1   out2
"the" → [0.00,  1.00,  0.00]   ← purely its own representation
"cat" → [1.94,  0.03,  1.94]   ← mostly itself (97%)
"sat" → [1.92,  1.01,  1.92]   ← blend of "cat" and "sat" (48% each)
```

**This is the magic:** "sat" now contains information from "cat" (48%) inside it — it "knows" that "cat" came before it and was relevant!

---

## COMPLETE VISUAL FLOW DIAGRAM

```
INPUT X (3 tokens, 4-dim each)
  "the" = [1, 0, 1, 0]
  "cat" = [0, 2, 0, 2]
  "sat" = [1, 1, 1, 1]
       │
       ├──────────────────────────────────────────┐
       │                                          │
       ├─── @ W_Q ──→  Q (Query)                 │
       │               "the": [1, 0, 0]          │
       │               "cat": [2, 2, 0]          │
       │               "sat": [2, 1, 1]          │
       │                    │                    │
       ├─── @ W_K ──→  K (Key)                   │
       │               "the": [1, 0, 0]          │
       │               "cat": [2, 2, 0]          │
       │               "sat": [2, 1, 1]          │
       │                    │                    │
       │         Q @ K^T  = Raw Scores           │
       │                    │                    │
       │         ÷ sqrt(3) = Scaled Scores       │
       │                    │                    │
       │         mask future → -inf              │
       │                    │                    │
       │         Softmax → Attention Weights     │
       │               "the": [1.00, 0.00, 0.00] │
       │               "cat": [0.03, 0.97, 0.00] │
       │               "sat": [0.05, 0.48, 0.48] │
       │                    │                    │
       └─── @ W_V ──→  V (Value)                 │
                       "the": [0, 1, 0]          │
                       "cat": [2, 0, 2]          │
                       "sat": [2, 2, 2]          │
                            │                    │
                  Attention Weights @ V          │
                            │                    │
                       OUTPUT                    │
                  "the": [0.00, 1.00, 0.00]      │
                  "cat": [1.94, 0.03, 1.94]      │
                  "sat": [1.92, 1.01, 1.92]      │
                            │                    │
                            └────────────────────┘
                         (shape same as input but enriched)
```

---

## THE FORMULA IN HUMAN LANGUAGE

```
Attention(Q, K, V) = Softmax( Q @ K^T / sqrt(d_k) ) @ V
```

Translated step by step:

```
1. Q @ K^T          → "How much does each word's question match each word's answer?"
2. / sqrt(d_k)      → "Normalize so scores don't explode"
3. Softmax(...)     → "Convert scores to probabilities (they must sum to 1)"
4. mask -inf        → "Block future tokens — no cheating allowed"
5. ... @ V          → "Use the probabilities to blend the actual content"
```

---

## WHY THIS IS REVOLUTIONARY

Before Attention (RNNs): Information flowed token-by-token
```
"the" → "cat" → "sat"
         ↑         ↑
         Only      Only sees
         sees      "the","cat"
         "the"
```
Information about "the" had to travel through "cat" to reach "sat" — it got diluted.

After Attention: Every token connects DIRECTLY to every other token
```
"the" ←──────────────────→ "sat"   (direct connection, no intermediary!)
"cat" ←──────────────────→ "sat"   (direct connection!)
"the" ←──────────────────→ "cat"   (direct connection!)
```
No information loss over distance. "sat" directly attends to "the" with whatever weight it needs.

This is why Transformers defeated RNNs — **O(1) distance** between any two tokens regardless of how far apart they are.
