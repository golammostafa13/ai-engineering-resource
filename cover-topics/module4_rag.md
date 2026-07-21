# Module 4: RAG — Retrieval-Augmented Generation

*Giving your agent long-term memory and access to external knowledge.*

---

## The Problem RAG Solves

An LLM has two types of knowledge:

```
1. PARAMETRIC KNOWLEDGE (inside weights)
   → Learned during pre-training
   → Fixed after training
   → Gets outdated
   → Can hallucinate on specific facts

2. CONTEXTUAL KNOWLEDGE (inside the prompt)
   → Whatever you inject into the current conversation
   → Dynamic, always fresh
   → Limited by context window size
```

**RAG = Retrieving relevant documents at query time and injecting them into the prompt.**

```
WITHOUT RAG:
  User: "What does our company's refund policy say?"
  LLM:  "I don't have information about your company's policies." ❌

WITH RAG:
  User: "What does our company's refund policy say?"
  Agent retrieves: policy_document.pdf → page 3 → "Refunds within 30 days..."
  LLM:  "According to your refund policy, customers can request a refund
         within 30 days of purchase." ✅
```

---

## The Core Idea: Semantic Search

RAG works because of **vector embeddings**. Any text can be converted to a vector (list of numbers) that captures its meaning.

```
"The cat sat on the mat"  →  [0.21, -0.43, 0.88, 0.12, ...]  (384 numbers)
"A feline rested on a rug" →  [0.19, -0.41, 0.85, 0.14, ...]  (384 numbers)
"The stock market crashed"  →  [-0.52, 0.31, -0.14, 0.77, ...] (384 numbers)
```

The first two sentences have very **similar vectors** (close in vector space) because they mean the same thing. The third is very different.

**Similarity = dot product of two vectors (Cosine Similarity)**

```
cos_sim("cat on mat", "feline on rug") = 0.94  ← very similar!
cos_sim("cat on mat", "stock market")  = 0.12  ← very different
```

RAG uses this to find the most relevant documents for any user query.

---

## The RAG Pipeline — 4 Stages

```
┌─────────────────────────────────────────────────────────────┐
│  OFFLINE (done once, before users arrive)                   │
│                                                             │
│  Your Documents (PDFs, text files, database records)       │
│      ↓                                                      │
│  Split into small CHUNKS (e.g., 500 characters each)       │
│      ↓                                                      │
│  Embed each chunk → vector                                  │
│      ↓                                                      │
│  Store vectors in VECTOR DATABASE                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ONLINE (happens at query time)                             │
│                                                             │
│  User Query: "What is the refund policy?"                  │
│      ↓                                                      │
│  Embed the query → query vector                             │
│      ↓                                                      │
│  Search vector DB for top-K most similar chunks            │
│      ↓                                                      │
│  Retrieved chunks injected into LLM prompt                  │
│      ↓                                                      │
│  LLM generates answer grounded in retrieved context         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Chunking

You cannot embed an entire 100-page PDF as one vector — the meaning gets diluted. Instead, split it into small overlapping chunks.

```
Original document:
"Our return policy allows customers to return items within 30 days.
 Items must be in original condition. Refunds are processed in 5-7 days.
 To initiate a return, contact support@company.com..."

Chunk 1 (chars 0-500):
  "Our return policy allows customers to return items within 30 days.
   Items must be in original condition. Refunds are processed..."

Chunk 2 (chars 400-900):   ← overlaps with chunk 1!
  "...Refunds are processed in 5-7 days.
   To initiate a return, contact support@company.com..."
```

**Why overlap?** So no important sentence is cut in half between two chunks.

---

## Stage 2: Embedding

Each chunk is converted to a dense vector using an **embedding model**:

```
Embedding Model (e.g., all-MiniLM-L6-v2, 384 dimensions):

Input:  "Refunds are processed in 5-7 business days."
Output: [0.12, -0.34, 0.88, 0.02, -0.71, 0.44, ...]  (384 numbers)
```

The embedding model is a neural network trained specifically to make similar texts produce similar vectors.

---

## Stage 3: Vector Database

A vector database stores:
- The chunk text
- The chunk's vector
- Optional metadata (source file, page number, date)

```
Vector DB contents:

ID  | Vector (384 dims)      | Text                          | Metadata
----|------------------------|-------------------------------|------------------
001 | [0.12, -0.34, 0.88...] | "Return within 30 days..."   | {file: policy.pdf, page: 3}
002 | [0.09, -0.31, 0.85...] | "Refunds in 5-7 days..."     | {file: policy.pdf, page: 3}
003 | [0.77,  0.21, -0.12..] | "Contact CEO at ceo@..."     | {file: contacts.pdf, page: 1}
```

Popular vector databases: **ChromaDB** (local, free), Pinecone, Weaviate, pgvector.

---

## Stage 4: Retrieval and Augmentation

When a user asks a question:

```
User: "How long does a refund take?"

Step 1: Embed the query
  query_vector = embed("How long does a refund take?")
  → [0.10, -0.30, 0.84, ...]

Step 2: Cosine similarity search across all stored vectors
  sim(query, chunk_001) = 0.78  ← about returns, relevant
  sim(query, chunk_002) = 0.94  ← about refund timing, very relevant!
  sim(query, chunk_003) = 0.11  ← about CEO contact, not relevant

Step 3: Return top-K chunks (K=2)
  → chunk_002: "Refunds in 5-7 days..."
  → chunk_001: "Return within 30 days..."

Step 4: Inject into LLM prompt:
  System: "Answer using ONLY the following context:
           [chunk_002 text]
           [chunk_001 text]"
  User: "How long does a refund take?"

Step 5: LLM generates grounded answer:
  "According to the policy, refunds are processed in 5-7 business days."
```

---

## The Grounded Prompt Template

```
SYSTEM:
  You are a helpful assistant. Answer questions ONLY based on the
  provided context. If the answer is not in the context, say
  "I don't have that information."

  Context:
  ─────────────────────────────────
  [Retrieved Chunk 1 text here]
  ─────────────────────────────────
  [Retrieved Chunk 2 text here]
  ─────────────────────────────────

USER:
  [User's actual question]
```

The phrase **"ONLY based on the provided context"** is critical. Without it, the LLM may mix retrieved facts with its own (possibly outdated) parametric knowledge.

---

## RAG as an Agent Tool

In an agentic system, RAG is implemented as a **tool**:

```python
def search_knowledge_base(query: str) -> str:
    """Search company documents for relevant information."""
    results = vector_db.query(query, top_k=3)
    return "\n\n".join([r.text for r in results])
```

The agent decides when to call this tool — just like weather or calculator tools. This gives the agent on-demand access to a private knowledge base.

---

## Key Concepts to Remember

| Term | Meaning |
|---|---|
| **Embedding** | Converting text to a vector of numbers |
| **Chunk** | A small piece of a larger document |
| **Vector DB** | Database that stores and searches by vector similarity |
| **Top-K retrieval** | Returning the K most similar chunks |
| **Cosine Similarity** | Measure of angle between two vectors (1 = identical, 0 = unrelated) |
| **Grounding** | Making the LLM answer based on retrieved facts, not memory |
| **Hallucination prevention** | RAG's main benefit — anchors answers to real documents |
