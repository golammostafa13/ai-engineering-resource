# LLM Core Concepts: How Language Models Work

This document summarizes the 10-step mental model of how text is processed, represented, generated, and optimized under the hood in Large Language Models (LLMs).

---

## 🪙 Part 1: Tokenization & Representation (Steps 1–5)

### 1. Characters to Bits
* **Concept:** A sentence is entered as text. Computers represent text as binary digits (**bits**: 0s and 1s). 
* **Details:** Under the UTF-8 encoding standard, English characters (ASCII) use 8 bits (1 byte) each, while emojis or non-English characters use 16 to 32 bits (2 to 4 bytes) to support a global set of symbols.

### 2. Bits to Bytes
* **Concept:** Bits are grouped into blocks of 8, which form **bytes**.
* **Details:** Bytes represent the base unit of raw data transmission and storage in computers.

### 3. Bytes to Decimal Values
* **Concept:** Each byte (8 bits) represents a binary number that translates to a **decimal value between 0 and 255**.
* **Details:** For example, the binary sequence `01000001` represents the decimal value `65`, which maps to the capital letter `A` in ASCII/Unicode.

### 4. Grouping Decimals into Vocabulary Symbols
* **Concept:** Rather than processing character-by-character (which is computationally slow), tokenizers search for recurring sequences of bytes and merge them into larger **symbols**.
* **Details:** Modern algorithms like **Byte-Pair Encoding (BPE)** count which byte sequences frequently appear next to each other in a training dataset (e.g., `t` + `h` $\to$ `th` $\to$ `the`) and merge them iteratively, building a vocabulary of thousands of unique symbols.

### 5. Symbols are the Tokens
* **Concept:** The resulting merged sequences of characters/bytes are the **tokens** representing a text.
* **Details:** Every unique token in the vocabulary has a corresponding unique integer ID (e.g., `'the'` $\to$ `262`, `' agent'` $\to$ `17230`). A sentence is converted into a list of these Token IDs before being sent to the neural network.

---

## 🧠 Part 2: Neural Networks & Inference (Steps 6–8)

### 6. Tokens Enter the Neural Network for Prediction
* **Concept:** The sequence of Token IDs is fed into the Neural Network to predict the next token in the sequence.
* **Details:** This is called **Autoregressive Generation**—the model predicts one token at a time, appends that predicted token to the input, and runs again to predict the subsequent token.

### 7. The Neural Network is a Massive Weight Expression
* **Concept:** The network itself is a giant mathematical function consisting of billions of tunable parameters called **weights**.
* **Details:** When tokens pass through the network, they are transformed using dense matrices of weight values, self-attention calculations, and non-linear activation functions to project the input meaning.

### 8. Probability Distribution Over the Vocabulary
* **Concept:** The model outputs a probability value for every possible token in its vocabulary for the next slot.
* **Details:** The network calculates raw scores (logits) for all items in its vocabulary. A **Softmax** function is applied to turn these scores into a probability distribution (summing to 100%), allowing us to choose the next token (using decoding parameters like Temperature or Top-p).

---

## ⚙️ Part 3: Evaluation & Optimization (Steps 9–10)

### 9. Correct Prediction (Inference)
* **Concept:** During generation (using the model), if the predicted token aligns with expectation, the text generates correctly.
* **Details:** During inference, the weights are **frozen** and static. No learning or parameters change dynamically based on correctness.

### 10. Weight Tweaking (Training / Fine-Tuning)
* **Concept:** If predictions are wrong during training, the network's weights are adjusted ("tweaked") to improve accuracy.
* **Details:** The difference between prediction and truth is measured using a **Loss Function**. Through **Backpropagation** and **Gradient Descent**, the algorithm adjusts the model's weights slightly in the opposite direction of the error, ensuring the model is more likely to make the correct prediction the next time it sees that pattern.
