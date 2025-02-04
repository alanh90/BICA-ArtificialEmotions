#!/usr/bin/env python
import os
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, AutoTokenizer
from emotion_aware_gpt2 import EmotionAwareGPT2LMHeadModel  # Ensure we're using the custom model

# -----------------------------------------------------------------------------
# Define the expected emotion keys
# -----------------------------------------------------------------------------
EXPECTED_EMOTION_KEYS = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]


# -----------------------------------------------------------------------------
# Dataset Class: Loads and filters valid samples
# -----------------------------------------------------------------------------
class SyntheticDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.samples = []
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        sample = json.loads(line.strip())
                        if (
                                "prompt" in sample and
                                "added_context" in sample and
                                "emotion_vector" in sample and
                                all(key in sample["emotion_vector"] for key in EXPECTED_EMOTION_KEYS)
                        ):
                            self.samples.append(sample)
                        else:
                            print(f"⚠️ Skipping sample due to missing keys: {line.strip()[:80]}...")
                    except json.JSONDecodeError:
                        print(f"⚠️ Skipping invalid JSON sample: {line.strip()[:80]}...")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = sample["prompt"].strip() + " " + sample["added_context"].strip()
        input_ids = self.tokenizer.encode(input_text, truncation=True, max_length=512)
        emotion_vals = [sample["emotion_vector"].get(key, 0.0) for key in EXPECTED_EMOTION_KEYS]
        emotion_tensor = torch.tensor(emotion_vals, dtype=torch.float)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "emotion": emotion_tensor
        }


def collate_fn(batch, pad_token_id):
    input_ids = [item["input_ids"] for item in batch]
    emotion = torch.stack([item["emotion"] for item in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = (input_ids != pad_token_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "emotion": emotion}


# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------
def train(args):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load the **correct** emotion-aware model
    model = EmotionAwareGPT2LMHeadModel.from_pretrained("gpt2")

    # Check if the model has an emotion adapter
    if not hasattr(model, "emotion_adapter"):
        print("❌ Error: Model does not have an 'emotion_adapter'. Make sure you're using the correct model!")
        return

    # Ensure only emotion adapter parameters are trainable
    for name, param in model.named_parameters():
        param.requires_grad = False

    for param in model.emotion_adapter.parameters():
        param.requires_grad = True  # Only train LoRA-based emotion adapter

    model.to(args.device)
    model.train()

    dataset = SyntheticDataset(args.data_file, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    )

    adapter_params = [param for name, param in model.named_parameters() if param.requires_grad]
    if not adapter_params:
        print("❌ Error: No trainable parameters found! Ensure LoRA is correctly integrated.")
        return

    optimizer = AdamW(adapter_params, lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            target_emotion = batch["emotion"].to(args.device)

            # Ensure `get_global_emotion_state` exists
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                pred_emotion = model.get_global_emotion_state()
            except AttributeError:
                print("❌ Error: Model does not have `get_global_emotion_state()`. Are you using the correct model?")
                return

            loss = torch.nn.functional.mse_loss(pred_emotion, target_emotion)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.num_epochs} - Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), args.output_model)
    print(f"✅ Model saved to {args.output_model}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LoRA adapter using synthetic emotion data.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to synthetic data JSONL file.")
    parser.add_argument("--output_model", type=str, default="trained_lora_adapter.pt", help="Path to save the trained model.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    args = parser.parse_args()

    train(args)
