# test.py
from data import dataloader, tokenizer

print("\n--- Running Pipeline Validation ---")

# Grab one batch
inputs, targets = next(iter(dataloader))

# Take the very first sequence in the batch
sample_input = inputs[0]
sample_target = targets[0]

print(f"Input tensor shape: {inputs.shape}")
print(f"Target tensor shape: {targets.shape}")

# Let's decode the first 10 tokens to prove they are shifted correctly
print("\nDecoding the first 10 tokens:")
for i in range(10):
    in_token = sample_input[i].item()
    out_token = sample_target[i].item()
    
    in_text = tokenizer.decode([in_token])
    out_text = tokenizer.decode([out_token])
    
    print(f"Step {i}: Model sees '{in_text}' (ID: {in_token}) --> Must predict '{out_text}' (ID: {out_token})")
