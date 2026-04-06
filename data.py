import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import DataLoader, Dataset

# Configuration
VOCAB_SIZE = 10000
CONTEXT_LENGTH = 512 # The maximum sequence length the model will process at once
BATCH_SIZE = 16

# ---------------------------------------------------------
# Step 1: Load the Dataset and Train the Tokenizer
# ---------------------------------------------------------
print("Loading TinyStories dataset...")
# We use a small subset just to train the tokenizer quickly
dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]") 

def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

print("Training custom BPE tokenizer...")
tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(
    batch_iterator(dataset),
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=["<|endoftext|>"]
)

# Save it so we don't have to train it every time
tokenizer.save("tinystories-bpe.json")
EOT_TOKEN_ID = tokenizer.token_to_id("<|endoftext|>")

# ---------------------------------------------------------
# Step 2: Tokenize and Chunk the Data
# ---------------------------------------------------------
class LanguageModelingDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.tokens = []
        
        # Tokenize everything and pack into a single massive 1D array
        # In a production environment, we'd stream this to save RAM, 
        # but TinyStories is small enough to hold in memory.
        print("Tokenizing dataset...")
        for item in hf_dataset:
            # Encode text and append the End-Of-Text token
            encoded = tokenizer.encode(item["text"]).ids + [EOT_TOKEN_ID]
            self.tokens.extend(encoded)
            
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        print(f"Total tokens in dataset: {len(self.tokens):,}")

    def __len__(self):
        # Calculate how many full context-length blocks we have
        return len(self.tokens) // self.seq_length

    def __getitem__(self, idx):
        # Grab a chunk of tokens of size seq_length + 1
        # The +1 is because the target for each token is the very next token
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1
        chunk = self.tokens[start_idx:end_idx]
        
        # x is the input sequence, y is the target sequence (shifted by 1)
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# Load the full training split (or a larger subset) for actual training
full_dataset = load_dataset("roneneldan/TinyStories", split="train[:10%]")
lm_dataset = LanguageModelingDataset(full_dataset, tokenizer, CONTEXT_LENGTH)

# ---------------------------------------------------------
# Step 3: Create the DataLoader
# ---------------------------------------------------------
dataloader = DataLoader(lm_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Test the pipeline
inputs, targets = next(iter(dataloader))
print(f"\nInput shape: {inputs.shape}")   # Expected: [16, 512]
print(f"Target shape: {targets.shape}") # Expected: [16, 512]
