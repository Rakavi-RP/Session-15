from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

# Create an IterableDataset for streaming text data.
class StreamingTextDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, max_length):
        self.hf_dataset = hf_dataset  # A streaming Hugging Face dataset.
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __iter__(self):
        for sample in self.hf_dataset:
            text = sample["text"]
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            yield encoding["input_ids"].squeeze(0)  # [max_length]

print("Loading streaming dataset...")
# Load the full train split in streaming mode.
hf_dataset = load_dataset("HuggingFaceTB/Cosmopedia-V2", "cosmopedia-v2", split="train", streaming=True)
