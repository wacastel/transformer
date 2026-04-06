import torch
import torch.optim as optim
import math
from model import LanguageModel
from data import dataloader, VOCAB_SIZE, CONTEXT_LENGTH

# --- Hyperparameters ---
D_MODEL = 256        # Hidden dimension size (kept small for rapid testing)
N_HEADS = 8          # Number of attention heads
N_LAYERS = 4         # Number of transformer blocks
LEARNING_RATE = 5e-4 # Standard starting LR for small transformers
EPOCHS = 1           # Number of times to loop through the dataset
EVAL_INTERVAL = 100  # How often to print the loss

def train():
    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Starting training on device: {device}")

    # 2. Initialize the Model
    print("Initializing Language Model...")
    model = LanguageModel(
        vocab_size=VOCAB_SIZE, 
        d_model=D_MODEL, 
        n_heads=N_HEADS, 
        n_layers=N_LAYERS, 
        max_seq_len=CONTEXT_LENGTH
    ).to(device)

    # Print total parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # 3. Setup the Optimizer
    # AdamW is the industry standard for training LLMs. 
    # It handles weight decay much better than standard Adam.
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # 4. The Training Loop
    model.train() # Set model to training mode
    
    print("\n--- Beginning Training ---")
    for epoch in range(EPOCHS):
        for step, (inputs, targets) in enumerate(dataloader):
            # Move data to the Mac's GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero out the gradients from the previous step
            optimizer.zero_grad()

            # Forward Pass: calculate predictions and loss
            logits, loss = model(inputs, targets=targets)

            # Backward Pass: calculate gradients
            loss.backward()

            # Gradient Clipping: prevents gradients from exploding and crashing training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            # Logging
            if step % EVAL_INTERVAL == 0:
                # Calculate perplexity (a more human-readable metric for language models)
                perplexity = math.exp(loss.item())
                print(f"Epoch {epoch} | Step {step:04d} | Loss: {loss.item():.4f} | Perplexity: {perplexity:.2f}")

    # 5. Save the trained weights
    print("\nTraining complete! Saving checkpoint...")
    torch.save(model.state_dict(), "tinystories_model_v1.pt")
    print("✅ Model saved as 'tinystories_model_v1.pt'")

if __name__ == "__main__":
    train()
