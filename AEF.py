import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import numpy as np

##############################################
# 1. Setup: Device, Seeds, and Model Download
##############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set

# Load the base model.
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()


##############################################
# 2. Dynamic Emotion State Module
##############################################
class EmotionState:
    """
    Maintains continuous emotion values (0.0 to 1.0) for a set of emotions.
    Updates gradually via an exponential moving average based on user input.
    """

    def __init__(self):
        self.state = {
            "curiosity": 0.5,
            "happiness": 0.5,
            "anger": 0.5,
            "sadness": 0.5,
        }
        self.decay = 0.9

    def update(self, user_input: str):
        increments = {"curiosity": 0.0, "happiness": 0.0, "anger": 0.0, "sadness": 0.0}
        if re.search(r"\b(curious|wonder|intrigued)\b", user_input, re.IGNORECASE):
            increments["curiosity"] = 0.3
        if re.search(r"\b(happy|good|joy)\b", user_input, re.IGNORECASE):
            increments["happiness"] = 0.3
        if re.search(r"\b(angry|frustrated|irate)\b", user_input, re.IGNORECASE):
            increments["anger"] = 0.3
        if re.search(r"\b(sad|depressed|mournful)\b", user_input, re.IGNORECASE):
            increments["sadness"] = 0.3

        for emo in self.state:
            self.state[emo] = self.decay * self.state[emo] + (1 - self.decay) * increments[emo]
            self.state[emo] = max(0.0, min(1.0, self.state[emo]))

    def get_vector(self) -> torch.Tensor:
        vec = [self.state["curiosity"], self.state["happiness"],
               self.state["anger"], self.state["sadness"]]
        return torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)

    def __str__(self):
        return str(self.state)


emotion_state = EmotionState()


##############################################
# 3. Emotion Adapter (LoRA-like Module)
##############################################
class EmotionAdapter(nn.Module):
    """
    Maps the 4-dimensional emotion state vector to a bias vector of dimension equal to the model's hidden size.
    """

    def __init__(self, emotion_dim: int, hidden_dim: int):
        super(EmotionAdapter, self).__init__()
        self.adapter = nn.Linear(emotion_dim, hidden_dim)

    def forward(self, emotion_vector: torch.Tensor) -> torch.Tensor:
        return self.adapter(emotion_vector)


hidden_dim_model = model.config.n_embd  # typically 768 for DistilGPT2
emotion_adapter = EmotionAdapter(emotion_dim=4, hidden_dim=hidden_dim_model).to(device)


##############################################
# 4. Modify LM Head to Integrate Emotion
##############################################
class EmotionEnhancedLMHead(nn.Module):
    """
    Wraps the original LM head so that it adds an emotion bias (via the EmotionAdapter)
    to the hidden states before computing logits.
    """

    def __init__(self, original_lm_head: nn.Module, emotion_adapter: nn.Module, emotion_state: EmotionState, scale: float = 0.1):
        super(EmotionEnhancedLMHead, self).__init__()
        self.original_lm_head = original_lm_head
        self.emotion_adapter = emotion_adapter
        self.emotion_state = emotion_state
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, hidden_dim)
        emo_vec = self.emotion_state.get_vector()  # (1, 4)
        bias = self.emotion_adapter(emo_vec)  # (1, hidden_dim)
        bias = self.scale * bias  # Scale the bias down
        seq_len = hidden_states.shape[1]
        bias_expanded = bias.unsqueeze(1).expand(-1, seq_len, -1)
        new_hidden = hidden_states + bias_expanded
        return self.original_lm_head(new_hidden)


original_lm_head = model.lm_head
model.lm_head = EmotionEnhancedLMHead(original_lm_head, emotion_adapter, emotion_state, scale=0.1)

##############################################
# 5. Quick Fine-Tuning Setup (Very Rough)
##############################################
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Use WikiText-2 and select a larger subset for a quick fine-tuning phase.
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
small_train = dataset["train"].select(range(500))
small_val = dataset["validation"].select(range(100))


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


tokenized_train = small_train.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_val = small_val.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./aef_quick_finetune",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="no",
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=1,
    fp16=False,
    push_to_hub=False,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

# Option: Ask the user whether to fine-tune again or load the previously fine-tuned model.
if os.path.exists("./aef_quick_finetune/pytorch_model.bin"):
    choice = input("A fine-tuned model already exists. Fine-tune again? (y/n): ").strip().lower()
    if choice == "y":
        print("Starting quick fine-tuning...")
        trainer.train()
        print("Quick fine-tuning complete.")
    else:
        print("Loading fine-tuned model from checkpoint...")
        model = AutoModelForCausalLM.from_pretrained("./aef_quick_finetune").to(device)
else:
    print("No fine-tuned model found. Starting quick fine-tuning...")
    trainer.train()
    print("Quick fine-tuning complete.")


##############################################
# 6. Generation Function for the Emotion-Integrated Model
##############################################
def generate_text_with_emotion(model, prompt, max_new_tokens=150, do_sample=True, top_p=0.95, temperature=1.0):
    """
    Generate text using the emotion-integrated model.
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


##############################################
# 7. Chat Interface for Testing
##############################################
def chat():
    print("Welcome to the Emotion-Integrated Chatbot!")
    print("Type 'quit' to exit.\n")
    conversation_history = ""

    while True:
        user_input = input("User: ")
        if user_input.lower().strip() == "quit":
            print("Goodbye!")
            break

        # Update emotion state based on user input.
        emotion_state.update(user_input)
        print("Current Emotion State:", emotion_state)

        conversation_history += f"User: {user_input}\n"
        prompt_text = conversation_history + "Assistant:"
        response = generate_text_with_emotion(model, prompt_text, max_new_tokens=150)
        if "Assistant:" in response:
            assistant_response = response.split("Assistant:")[-1].strip()
        else:
            assistant_response = response.strip()
        conversation_history += f"Assistant: {assistant_response}\n"
        print("Assistant:", assistant_response, "\n")


if __name__ == "__main__":
    chat()
