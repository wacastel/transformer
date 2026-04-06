import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure we route computation to the Mac's GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class ModernTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads) # To be implemented with RoPE
        self.feed_forward = SwiGLUFeedForward(d_model)        # To be implemented
        self.norm1 = RMSNorm(d_model)                         # To be implemented
        self.norm2 = RMSNorm(d_model)

    def forward(self, x):
        # Pre-normalization architecture
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # The learnable scaling parameter (gamma)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate the mean of the squares of the features along the last dimension
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        # Scale x by the reciprocal of the square root (rsqrt) for speed
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        # Apply the learnable weight
        return self.weight * x_normed

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        # Standard LLaMA sizing: scale hidden_dim to maintain parameter parity
        if hidden_dim is None:
            hidden_dim = int(4 * d_model * 2 / 3)
            
        # We use bias=False which is standard practice in modern LLMs
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x):
        # The gating mechanism: SiLU(xW1) * xW3
        gated_activation = F.silu(self.w1(x)) * self.w3(x)
        # Final projection back to d_model
        return self.w2(gated_activation)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        # Calculate the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        
        # Calculate the rotation angles
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Duplicate the frequencies to match the vector dimensions
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # FIX: Change the unsqueeze dimensions to [1, max_seq_len, 1, dim]
        # This aligns perfectly with [batch, seq_len, n_heads, head_dim]
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(2))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(2))

    def forward(self, q, k, seq_len):
        # FIX: Update the slicing to pull from the correct sequence dimension
        cos = self.cos_cached[:, :seq_len, ...]
        sin = self.sin_cached[:, :seq_len, ...]
        
        # Helper function to rotate half the hidden dimensions
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        
        # Apply the rotary transformation
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None):
        super().__init__()
        self.n_heads = n_heads
        # If n_kv_heads isn't specified, we default to standard MHA
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = d_model // n_heads
        
        # Notice the wq matrix is larger than wk and wv if n_kv_heads < n_heads
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 1. Linear projections
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # 2. Reshape into heads: [batch, seq, num_heads, head_dim]
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # 3. Apply Rotary Positional Embeddings
        xq, xk = self.rope(xq, xk, seq_len)
        
        # 4. Transpose for attention computation: [batch, num_heads, seq, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # 5. Repeat KV heads to match the number of Query heads (GQA logic)
        if self.n_kv_heads < self.n_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            xk = torch.repeat_interleave(xk, dim=1, repeats=num_repeat)
            xv = torch.repeat_interleave(xv, dim=1, repeats=num_repeat)
        
        # 6. Compute highly optimized causal attention
        # is_causal=True automatically applies the triangular mask so tokens can't see the future
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        
        # 7. Reshape back and project to final output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()
        # 1. Token Embedding Layer: Converts token IDs to d_model vectors
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        
        # Note: We do NOT need an absolute positional embedding layer here 
        # because our Rotary Positional Embeddings (RoPE) handle positions 
        # dynamically inside the MultiHeadAttention layers!

        # 2. The Deep Network: A stack of our custom transformer blocks
        self.layers = nn.ModuleList(
            [ModernTransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        
        # 3. Final Normalization
        self.norm = RMSNorm(d_model)
        
        # 4. Output Projection: Maps hidden states back to the vocabulary size
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        
        # 5. Weight Tying: Sharing the embedding and output weights 
        # This is a standard trick that saves roughly 20% of the model's total parameters
        # and grounds the output predictions in the same latent space as the inputs.
        self.tok_embeddings.weight = self.output.weight

    def forward(self, x, targets=None):
        # x is a batch of token IDs: shape (batch_size, seq_len)
        
        # Get the token embeddings
        h = self.tok_embeddings(x)
        
        # Pass sequentially through the transformer blocks
        for layer in self.layers:
            h = layer(h)
            
        # Apply the final RMSNorm
        h = self.norm(h)
        
        # Get the final predictions: shape (batch_size, seq_len, vocab_size)
        logits = self.output(h)
        
        loss = None
        if targets is not None:
            # PyTorch's cross_entropy expects a 2D input (Batch * Seq_Len, Vocab_Size)
            # and a 1D target (Batch * Seq_Len), so we flatten them.
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss
