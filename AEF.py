import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import numpy as np
from torch.nn import functional as F

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

model_name = "distilgpt2"  # Or any compatible model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_name)
base_model.config.output_hidden_states = True
base_model.to(device)
base_model.eval()

# 2. Emotion State Management
class EmotionState:
    def __init__(self, emotion_dim=4):
        self.emotion_vector = torch.zeros(1, emotion_dim).to(device)
        self.emotion_names = ["joy", "sadness", "anger", "fear"]

    def update(self, user_input):
        self.emotion_vector.fill_(0.0)  # Reset emotions for each input
        user_input = user_input.lower()

        if "happy" in user_input or "excited" in user_input or "joyful" in user_input:
            self.emotion_vector[0, self.emotion_names.index("joy")] = 1.0
        if "sad" in user_input or "unhappy" in user_input or "depressed" in user_input:
            self.emotion_vector[0, self.emotion_names.index("sadness")] = 1.0
        if "angry" in user_input or "furious" in user_input or "mad" in user_input:
            self.emotion_vector[0, self.emotion_names.index("anger")] = 1.0
        if "scared" in user_input or "frightened" in user_input or "terrified" in user_input or "fear" in user_input:
            self.emotion_vector[0, self.emotion_names.index("fear")] = 1.0

        self.emotion_vector = torch.clamp(self.emotion_vector, 0, 1)

    def get_vector(self):
        return self.emotion_vector

    def __str__(self):
        # Squeeze the tensor and convert to a list for display
        emotion_list = self.emotion_vector.squeeze(0).tolist()
        emotion_str = ", ".join([f"{name}: {val:.2f}" for name, val in zip(self.emotion_names, emotion_list)])
        return f"[{emotion_str}]"

emotion_state = EmotionState()

# 3. Emotion Adapter (LORA-like)
class EmotionAdapter(nn.Module):
    def __init__(self, emotion_dim, hidden_dim, adapter_dim=128):
        super().__init__()
        self.down_proj = nn.Linear(emotion_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, hidden_dim)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, emotion_vector):
        down = self.down_proj(emotion_vector)
        up = self.up_proj(down)
        return up * self.scale

emotion_adapter = EmotionAdapter(4, base_model.config.n_embd).to(device)

# 4. Model with Emotion Prediction Head
class EmotionAwareModel(nn.Module):
    def __init__(self, base_model, emotion_adapter, emotion_dim=4):
        super().__init__()
        self.base_model = base_model
        self.emotion_adapter = emotion_adapter
        self.emotion_head = nn.Linear(base_model.config.n_embd, emotion_dim)

    def forward(self, input_ids=None, attention_mask=None, labels=None, emotion_labels=None, **kwargs):
        # Use provided emotion_labels or create zeros for inference.
        if emotion_labels is not None:
            emotion_vector = emotion_labels  # Expected shape: (batch, 4)
        else:
            emotion_vector = torch.zeros(input_ids.size(0), 4).to(device)

        outputs = self.base_model(input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        # Use the last element of hidden_states as the final hidden state
        raw_hidden_state = outputs.hidden_states[-1]
        seq_length = raw_hidden_state.size(1)

        adapter_output = self.emotion_adapter(emotion_vector)  # Shape: (batch, hidden_dim)
        # Expand adapter_output to match the sequence length
        adapter_output_expanded = adapter_output.unsqueeze(1).expand(-1, seq_length, -1)
        hidden_states = raw_hidden_state + adapter_output_expanded

        # Compute emotion prediction using the last token's hidden state
        emotion_logits = self.emotion_head(hidden_states[:, -1, :])
        loss = outputs.loss
        if emotion_labels is not None:
            emotion_loss = F.mse_loss(emotion_logits, emotion_labels)
            loss = loss + 0.5 * emotion_loss

        # Return a dictionary containing the loss and logits.
        return {"loss": loss, "logits": outputs.logits, "hidden_states": hidden_states}

    def generate(self, input_ids=None, attention_mask=None, emotion_vector=None, **kwargs):
        """
        For generation, we modify the input embeddings by adding the emotion adapter's output.
        """
        if emotion_vector is None:
            emotion_vector = torch.zeros(input_ids.size(0), 4).to(device)
        if emotion_vector.ndim == 1:
            emotion_vector = emotion_vector.unsqueeze(0)

        # Get the input embeddings from the base model's embedding layer
        input_embeds = self.base_model.transformer.wte(input_ids)  # (batch, seq_len, hidden_dim)
        adapter_output = self.emotion_adapter(emotion_vector)       # (batch, hidden_dim)
        # Expand adapter output across the sequence length
        adapter_output_expanded = adapter_output.unsqueeze(1).expand(-1, input_embeds.size(1), -1)
        modified_input_embeds = input_embeds + adapter_output_expanded

        # Call generate with modified input embeddings.
        return self.base_model.generate(
            attention_mask=attention_mask,
            input_embeds=modified_input_embeds,
            **kwargs
        )

# 5. Data Preparation
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")  # Example, replace with your dataset
train_dataset = dataset["train"].select(range(500))  # Reduced for demonstration
val_dataset = dataset["validation"].select(range(100))  # Reduced for demonstration

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Dummy emotion labels (replace with your actual labels)
def add_emotion_labels(example):
    # Create random emotion labels; in practice, use your own logic.
    example['emotion_labels'] = np.random.rand(4).tolist()
    return example

tokenized_train = tokenized_train.map(add_emotion_labels)
tokenized_val = tokenized_val.map(add_emotion_labels)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. Training
emotion_aware_model = EmotionAwareModel(base_model, emotion_adapter).to(device)

training_args = TrainingArguments(
    output_dir="./emotion_finetuned",
    num_train_epochs=3,               # Adjust as needed
    per_device_train_batch_size=4,    # Adjust as needed
    per_device_eval_batch_size=4,     # Adjust as needed
    learning_rate=5e-5,               # Adjust as needed
    weight_decay=0.01,                # Adjust as needed
    fp16=False,                       # Set to True if you have a compatible GPU
    save_safetensors=False            # Disable safetensors saving to avoid shared memory issues
    # ... other training arguments ...
)


trainer = Trainer(
    model=emotion_aware_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

trainer.train()

# 7. Inference (Modified to use the trained model and emotion state)
def generate_text_with_emotion(model, prompt, emotion_vector, max_new_tokens=150, do_sample=True, top_p=0.95, temperature=1.0):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    if emotion_vector.ndim == 1:
        emotion_vector = emotion_vector.unsqueeze(0)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        emotion_vector=emotion_vector
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def chat():
    print("Welcome to the Emotion-Integrated Chatbot!")
    print("Type 'quit' to exit.\n")
    conversation_history = ""

    while True:
        user_input = input("User: ")
        if user_input.lower().strip() == "quit":
            print("Goodbye!")
            break

        emotion_state.update(user_input)  # Update emotion based on input
        print("Current Emotion State:", emotion_state)

        conversation_history += f"User: {user_input}\n"
        prompt_text = conversation_history + "Assistant:"

        # Use the trained model for inference
        response = generate_text_with_emotion(emotion_aware_model, prompt_text, emotion_state.get_vector(), max_new_tokens=150)

        # Extract assistant response (split on the prompt token if necessary)
        if "Assistant:" in response:
            assistant_response = response.split("Assistant:")[-1].strip()
        else:
            assistant_response = response.strip()

        conversation_history += f"Assistant: {assistant_response}\n"
        print("Assistant:", assistant_response, "\n")

if __name__ == "__main__":
    chat()
