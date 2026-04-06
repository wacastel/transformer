import torch
from model import (
    RMSNorm, 
    SwiGLUFeedForward, 
    RotaryPositionalEmbedding, 
    MultiHeadAttention, 
    ModernTransformerBlock,
    LanguageModel
)

def run_tests():
    # 1. Setup device and dummy dimensions
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Running Unit Tests on Device: {device} ---\n")

    # We use small dimensions just to test the logic
    batch_size = 2
    seq_len = 16
    d_model = 128
    n_heads = 4
    head_dim = d_model // n_heads

    # Create a dummy input tensor [Batch, Sequence Length, Hidden Dimension]
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    print(f"Dummy Input Tensor Shape: {x.shape}\n")

    # 2. Test RMSNorm
    try:
        norm = RMSNorm(d_model).to(device)
        out_norm = norm(x)
        assert out_norm.shape == x.shape, "RMSNorm altered the tensor shape!"
        print("✅ RMSNorm: PASSED")
    except Exception as e:
        print(f"❌ RMSNorm: FAILED - {e}")

    # 3. Test SwiGLU Feed Forward
    try:
        ffn = SwiGLUFeedForward(d_model).to(device)
        out_ffn = ffn(x)
        assert out_ffn.shape == x.shape, "SwiGLU altered the tensor shape!"
        print("✅ SwiGLUFeedForward: PASSED")
    except Exception as e:
        print(f"❌ SwiGLUFeedForward: FAILED - {e}")

    # 4. Test Rotary Positional Embedding (RoPE)
    try:
        rope = RotaryPositionalEmbedding(head_dim).to(device)
        # Dummy Queries and Keys [Batch, Seq_Len, Heads, Head_Dim]
        q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)
        
        q_rot, k_rot = rope(q, k, seq_len)
        assert q_rot.shape == q.shape, "RoPE altered the Query shape!"
        assert k_rot.shape == k.shape, "RoPE altered the Key shape!"
        print("✅ RotaryPositionalEmbedding: PASSED")
    except Exception as e:
        print(f"❌ RotaryPositionalEmbedding: FAILED - {e}")

    # 5. Test Multi-Head Attention (with GQA capability)
    try:
        # Testing standard MHA (n_heads == n_kv_heads)
        mha = MultiHeadAttention(d_model, n_heads).to(device)
        out_mha = mha(x)
        assert out_mha.shape == x.shape, "MultiHeadAttention altered the tensor shape!"
        print("✅ MultiHeadAttention (Standard MHA): PASSED")

        # Testing GQA (n_heads > n_kv_heads)
        gqa = MultiHeadAttention(d_model, n_heads, n_kv_heads=2).to(device)
        out_gqa = gqa(x)
        assert out_gqa.shape == x.shape, "Grouped-Query Attention altered the tensor shape!"
        print("✅ MultiHeadAttention (Grouped-Query GQA): PASSED")
    except Exception as e:
        print(f"❌ MultiHeadAttention: FAILED - {e}")

    # 6. Test the Complete Modern Transformer Block
    try:
        block = ModernTransformerBlock(d_model, n_heads).to(device)
        out_block = block(x)
        assert out_block.shape == x.shape, "Transformer Block altered the tensor shape!"
        print("✅ ModernTransformerBlock: PASSED")
    except Exception as e:
        print(f"❌ ModernTransformerBlock: FAILED - {e}")

    # 7. Test the Full Language Model
    try:
        vocab_size = 10000 # Matching our data.py configuration
        n_layers = 2
        max_seq_len = 512
        
        # Create the full model
        llm = LanguageModel(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
        
        # Create dummy token IDs (integers instead of floats)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        target_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Forward pass with loss calculation
        logits, loss = llm(token_ids, targets=target_ids)
        
        assert logits.shape == (batch_size, seq_len, vocab_size), "LLM output shape is incorrect!"
        assert loss is not None, "Loss was not calculated!"
        print("✅ Full LanguageModel: PASSED")
        print(f"   -> Initial untrained loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Full LanguageModel: FAILED - {e}")

    print("\n--- All Tests Completed ---")

if __name__ == "__main__":
    run_tests()
