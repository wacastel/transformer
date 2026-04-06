# Custom Decoder-Only Large Language Model

A from-scratch, state-of-the-art decoder-only Large Language Model (LLM) implemented in PyTorch. This project demonstrates the fundamental architecture and training pipeline of modern models (akin to the LLaMA family), optimized natively for Apple Silicon (MPS) to take full advantage of unified memory architectures.

## Overview

This repository contains the complete pipeline for pre-training and instruction-tuning a language model on the TinyStories dataset. It includes custom implementations of modern architectural breakthroughs that improve upon the vanilla 2017 Transformer, specifically Rotary Positional Embeddings (RoPE), Grouped-Query Attention (GQA), Root Mean Square Normalization (RMSNorm), and SwiGLU activations.

## Architecture & Mathematical Theory

The model is built in a highly modular, object-oriented structure. Below is the theoretical breakdown of each core component found in `model.py`.

### 1. Root Mean Square Normalization (`RMSNorm`)
Standard Layer Normalization centers data by subtracting the mean and dividing by the variance. RMSNorm is a computational optimization that removes the mean-centering entirely, as scaling by the root mean square is mathematically sufficient and faster to compute.

**Theory & Math:**
For an input vector $x$ of dimension $d$, the RMS normalized output is:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma$$
Where $\epsilon$ is a small constant to prevent division by zero, $\gamma$ is a learnable scaling parameter, and $\odot$ represents element-wise multiplication.

### 2. SwiGLU Feed Forward Network (`SwiGLUFeedForward`)
Traditional transformers use a Feed Forward Network (FFN) with a ReLU activation function. This model replaces it with a Swish Gated Linear Unit (SwiGLU), which provides a smoother gradient flow and acts as a learned gating mechanism.

**Theory & Math:**
The activation of one linear projection is gated by the Swish (SiLU) activation of another, before a final linear projection:
$$\text{SwiGLU}(x) = (\text{SiLU}(xW_1) \odot xW_3) W_2$$
Where the SiLU function is defined as $x \cdot \sigma(x)$. To maintain parameter parity with standard transformers, the hidden dimension is scaled by $\frac{2}{3} \times 4d$.

### 3. Rotary Positional Embeddings (`RotaryPositionalEmbedding`)
Instead of adding absolute positional vectors to token embeddings, RoPE rotates the Query and Key vectors in a complex plane. This ensures that the dot product between any Query and Key is purely a function of their relative distance, vastly improving the model's understanding of sequence order and proximity.

**Theory & Math:**
For a vector $x$ at position $m$, a rotation matrix $R_{\Theta,m}$ is applied based on a set of frequencies $\theta$:
$$f_q(x_m, m) = (x_m \cos m\theta_1 - x_{m+1} \sin m\theta_1, x_m \sin m\theta_1 + x_{m+1} \cos m\theta_1, \dots)$$
Because of the mathematical properties of rotation matrices, the attention score $q_m^T k_n$ cleanly reduces to a function of $(m - n)$.

### 4. Grouped-Query Attention (`MultiHeadAttention`)
To address the memory bandwidth bottleneck of loading distinct Key and Value matrices for every single Query head during inference, this module implements Grouped-Query Attention (GQA).

**Theory & Math:**
Standard attention is computed as:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
In GQA, multiple Query heads share a single Key-Value (KV) pair. If there are 8 Query heads and 2 KV heads, the KV representations are repeated (broadcasted) across 4 Query heads each. This significantly reduces VRAM usage without degrading reasoning capabilities.

### 5. The Core Block (`ModernTransformerBlock`)
This class binds the above components together using a pre-normalization architecture, which is more stable for deep networks than the post-normalization used in the original "Attention is All You Need" paper.

**Theory & Math:**
The forward pass implements two sequential residual connections:
$$h = x + \text{Attention}(\text{RMSNorm}(x))$$
$$output = h + \text{SwiGLU}(\text{RMSNorm}(h))$$

### 6. The Wrapper (`LanguageModel`)
The top-level class orchestrates the token embedding, the sequential stack of transformer blocks, the final normalization, and the projection back into the vocabulary space. It also utilizes **weight tying**, sharing the weights of the input embedding layer and the final output linear layer to save parameters and enforce latent space alignment.

---

## Phase 1: Pre-Training the Model

The pre-training pipeline tokenizes the dataset, batches the sequences, and optimizes the model using the AdamW optimizer to teach foundational grammar and structure.

1. Ensure the required libraries are installed:
   ```bash
   pip install torch datasets tokenizers
   ```
2. Execute the training script. The script will automatically download the HuggingFace dataset, train the BPE tokenizer, and begin the training loop.
   ```bash
   python train.py
   ```
3. **Hardware Note:** The script automatically routes tensor operations to the `mps` device backend for Apple Silicon. If run on a standard PC, it will default to `cuda` (if available) or `cpu`.

---

## Phase 2: Instruction Tuning (Alignment)

To convert the model from a document-completer into a conversational assistant, the pre-trained weights undergo Supervised Fine-Tuning (SFT). The `finetune.py` script automatically wraps the raw dataset in a synthetic conversational template (`<|user|>` and `<|assistant|>`) and trains at a significantly lower learning rate to prevent catastrophic forgetting.

1. Ensure the pre-training loop has completed and the weights file (`tinystories_model_v1.pt`) is present.
2. Execute the fine-tuning script:
   ```bash
   python finetune.py
   ```

---

## Interactive Chat Interface

Once the model has been instruction-tuned, you can interact with it using the dedicated chat script. This interface automatically formats your inputs into the template the model was trained on and streams the generation autoregressively using temperature scaling and Top-K filtering.

1. Ensure the fine-tuned weights file (`tinystories_chat_v1.pt`) is present in the root directory.
2. Run the interactive generator:
   ```bash
   python chat.py
   ```
3. Enter a prompt when requested to start the conversation. 

**Example Usage:**
```text
==================================================
🤖 TinyStories Assistant is online.
Type 'quit' or 'exit' to end the conversation.
==================================================

You: Please tell me a story about a brave little cat.
Assistant: Once upon a time, there was a brave little cat. The cat wanted to explore the world...
```