import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import LanguageModel
from data import VOCAB_SIZE, CONTEXT_LENGTH

# --- Configuration (Must match train.py exactly) ---
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
MODEL_PATH = "tinystories_model_v1.pt"
TOKENIZER_PATH = "tinystories-bpe.json"

def load_model_and_tokenizer(device):
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    print("Loading model architecture...")
    model = LanguageModel(
        vocab_size=VOCAB_SIZE, 
        d_model=D_MODEL, 
        n_heads=N_HEADS, 
        n_layers=N_LAYERS, 
        max_seq_len=CONTEXT_LENGTH
    ).to(device)
    
    print("Loading trained weights...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # model.eval() turns off training-specific layers like Dropout 
    # (if we had them) and locks the weights.
    model.eval() 
    return model, tokenizer

def generate(model, tokenizer, prompt, max_new_tokens=150, temperature=0.8, top_k=40, device="cpu"):
    # 1. Encode the user's starting prompt into token IDs
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0).to(device)
    
    print(f"\n--- Generating ---")
    print(prompt, end="", flush=True)

    # 2. Autoregressive loop
    # torch.no_grad() tells PyTorch to stop tracking history for backpropagation,
    # which saves a massive amount of memory and speeds up generation.
    with torch.no_grad(): 
        for _ in range(max_new_tokens):
            # Crop the context window if it exceeds our max sequence length (512)
            context = input_ids[:, -CONTEXT_LENGTH:]
            
            # Forward pass to get predictions
            logits, _ = model(context)
            
            # We only care about the predictions for the very last token in the sequence
            next_token_logits = logits[0, -1, :]
            
            # 3. Apply Temperature
            # Dividing logits by temperature changes the probability distribution curve
            next_token_logits = next_token_logits / temperature
            
            # 4. Apply Top-K filtering
            # Find the top 'K' values, and set everything else to negative infinity
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[-1]] = -float('Inf')
            
            # 5. Convert raw logits into probabilities (0.0 to 1.0)
            probs = F.softmax(next_token_logits, dim=-1)
            
            # 6. Sample the next token based on those probabilities
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 7. Stop early if the model generates the End-Of-Text token
            if next_token.item() == tokenizer.token_to_id("<|endoftext|>"):
                break
                
            # 8. Append the new token to our running sequence
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
            
            # 9. Decode and print the single token dynamically
            word = tokenizer.decode([next_token.item()])
            print(word, end="", flush=True)
            
    print("\n------------------\n")

if __name__ == "__main__":
    # Ensure we use the Mac's GPU for inference too
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    try:
        model, tokenizer = load_model_and_tokenizer(device)
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find {MODEL_PATH}.")
        print("Make sure your train.py script has completely finished running and saved the model.")
        exit(1)
        
    print("✅ Model loaded successfully!")
    
    # Create an interactive chat loop
    while True:
        user_prompt = input("Enter a prompt to start a story (or 'quit' to exit): ")
        if user_prompt.lower() in ['quit', 'exit']:
            break
            
        generate(
            model, 
            tokenizer, 
            user_prompt, 
            max_new_tokens=150, 
            temperature=0.8, 
            top_k=40, 
            device=device
        )
