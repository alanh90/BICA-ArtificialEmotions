import time
import threading
from typing import Dict, List

import torch
import torch.nn as nn
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from peft import LoraConfig, get_peft_model, TaskType, PeftModel


# --------------------------
# CONFIG
# --------------------------
MODEL_NAME = "gpt2"
EMOTION_TAGS = ["anger", "fear", "joy", "love", "sadness"]
MAX_GOEMOTIONS_SAMPLES = 2000
TRAIN_EPOCHS = 3
BATCH_SIZE = 4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
MAX_LENGTH = 128

# Decay rates (per second)
EMOTION_DECAY_RATES = {
    "anger": 0.0001,
    "fear": 0.0002,
    "joy": 0.0003,
    "love": 0.00015,
    "sadness": 0.0001,
}

# Emotion keywords and their associated weights
EMOTION_KEYWORDS = {
    "anger": ["angry", "furious", "frustrated", "irritated", "annoyed"],
    "fear": ["scared", "afraid", "terrified", "anxious", "nervous"],
    "joy": ["happy", "joyful", "glad", "excited", "elated"],
    "love": ["love", "affection", "fond", "adore", "cherish"],
    "sadness": ["sad", "unhappy", "depressed", "miserable", "gloomy"],
}


# --------------------------
# Emotional Memory
# --------------------------

class EmotionalMemory:
    """
    Manages emotional intensities over time. A background thread
    continuously decays intensities. Read/write with thread safety.
    """
    def __init__(self, emotions: List[str], decay_rates: Dict[str, float]):
        self.emotions = emotions
        self.decay_rates = decay_rates
        self.state = {e: 0.0 for e in emotions}

        self.last_update_time = time.time()
        self.lock = threading.Lock()

        # Start background decay
        self.decay_thread = threading.Thread(target=self._continuous_decay, daemon=True)
        self.decay_thread.start()

    def update(self, emotion: str, intensity: float):
        """
        Increment the given emotion by 'intensity', capped at 1.0
        """
        with self.lock:
            if emotion in self.state:
                self.state[emotion] = min(1.0, self.state[emotion] + intensity)

    def get_state(self) -> Dict[str, float]:
        with self.lock:
            return dict(self.state)

    def set_state(self, new_state: Dict[str, float]):
        """
        Overwrite entire emotional state if needed
        (e.g., from a saved checkpoint).
        """
        with self.lock:
            for e in self.emotions:
                if e in new_state:
                    self.state[e] = new_state[e]

    def _continuous_decay(self):
        """
        Reduce intensities based on time elapsed, every second.
        """
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update_time
                for e in self.emotions:
                    dec = self.decay_rates[e] * elapsed
                    self.state[e] = max(0.0, self.state[e] - dec)
                self.last_update_time = now
            time.sleep(1)


# --------------------------
# EmotionalLoRAModel
# --------------------------

class EmotionalLoRAModel(GPT2LMHeadModel):
    """
    Subclass GPT2LMHeadModel to:
      1) Keep GPT-2's "transformer" submodule so PEFT can find it.
      2) Inject emotional vectors in the token embeddings before forward.
      3) Remain generation-compatible.

    This solves the "no attribute 'transformer'" error and the
    "prepare_inputs_for_generation" issue.
    """

    def __init__(self, base_model: GPT2LMHeadModel, memory: EmotionalMemory):
        # Initialize this GPT2LMHeadModel from the same config
        super().__init__(base_model.config)

        # Copy over the essential GPT-2 submodules
        self.transformer = base_model.transformer  # main GPT2 stack
        self.lm_head = base_model.lm_head          # output layer

        # Custom emotional memory
        self.memory = memory

        # We'll map from [num_emotions] -> [hidden_size]
        self.emotions = list(self.memory.state.keys())
        self.num_emotions = len(self.emotions)
        hidden_size = base_model.config.hidden_size

        self.emotion2hidden = nn.Linear(self.num_emotions, hidden_size)
        self.gate = nn.Parameter(torch.zeros(1))

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs
    ):
        # If input_ids is None, we just delegate to super() which
        # calls the standard GPT2LMHeadModel forward.  This might
        # happen during generation for partial calls, e.g. if
        # "past_key_values" are in use.
        if input_ids is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )

        # 1) Convert input_ids -> embeddings
        input_embeds = self.transformer.wte(input_ids)

        # 2) Build the emotion vector from memory
        memory_state = self.memory.get_state()  # {emotion: value}
        intensities = [memory_state[e] for e in self.emotions]
        intensities_tensor = torch.tensor(
            intensities,
            device=input_embeds.device,
            dtype=input_embeds.dtype
        )
        emotion_vector = self.emotion2hidden(intensities_tensor)  # shape [hidden_size]
        emotion_vector = emotion_vector.unsqueeze(0).unsqueeze(1) # shape [1,1,hidden_size]

        # Expand for entire batch/sequence
        batch_size, seq_len, hidden_size = input_embeds.shape
        emotion_vector = emotion_vector.expand(batch_size, seq_len, hidden_size)

        # Scale by gating parameter
        emotion_offset = self.gate * emotion_vector

        # 3) Inject into token embeddings
        input_embeds = input_embeds + emotion_offset

        # 4) Forward the rest of the Transformer
        transformer_outputs = self.transformer(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            **kwargs
        )

        hidden_states = transformer_outputs[0]  # last hidden state

        # 5) LM head to get logits
        lm_logits = self.lm_head(hidden_states)

        # 6) Return a standard CausalLMOutput
        return CausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        If the HF generation pipeline calls this,
        we can either pass it up or do minimal forwarding.
        """
        # GPT2LMHeadModel basically does the same:
        return {"input_ids": input_ids, **kwargs}


# --------------------------
# Dataset & Preprocessing
# --------------------------

def load_goemotions_subset(emotion_tags, max_samples=2000):
    ds = load_dataset("go_emotions", "simplified")["train"]
    label_names = ds.features["labels"].feature.names
    valid_label_indices = [label_names.index(e) for e in emotion_tags if e in label_names]

    def is_relevant(row):
        # Keep only samples with exactly one of our target emotions
        return len(set(row["labels"]).intersection(valid_label_indices)) == 1

    filtered_ds = ds.filter(is_relevant)
    filtered_ds = filtered_ds.select(range(min(max_samples, len(filtered_ds))))

    def create_prompt(row):
        emotion_idx = row["labels"][0]
        emotion_str = label_names[emotion_idx]
        user_text = row["text"]
        return {"text": f"User: {user_text}\nBot:", "emotion_label": emotion_str}

    return filtered_ds.map(create_prompt)


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )


# --------------------------
# LoRA Setup & Training
# --------------------------

def create_lora_config():
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["c_attn"],  # typical for GPT-2
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

def train_model(emotional_model, tokenizer, train_dataset, val_dataset, output_dir="lora-emotion"):
    # 1) Create LoRA config
    lora_config = create_lora_config()

    # 2) Wrap the EmotionalLoRAModel with LoRA
    peft_model = get_peft_model(emotional_model, lora_config)

    # 3) Trainer config
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=200,
        logging_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    # 4) Save final LoRA adapter
    peft_model.save_pretrained(output_dir)


# --------------------------
# Inference
# --------------------------

def update_emotional_state(memory, user_input, emotion_keywords):
    """
    Bump emotional intensities if the user's text has certain keywords.
    """
    txt_lower = user_input.lower()
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in txt_lower:
                memory.update(emotion, 0.5)

def generate_response(peft_model, tokenizer, memory, prompt, max_length=100):
    """
    Just call generate on peft_model.base_model so that the emotional injection occurs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        # Because our EmotionalLoRAModel is the .base_model, we call .generate on it
        outputs = peft_model.base_model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_k=50
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# --------------------------
# Main
# --------------------------

def main():
    # 1) Emotional memory
    memory = EmotionalMemory(EMOTION_TAGS, EMOTION_DECAY_RATES)

    # 2) Load a GoEmotions subset
    dataset = load_goemotions_subset(EMOTION_TAGS, MAX_GOEMOTIONS_SAMPLES)

    # 3) Train/test split
    split_ds = dataset.train_test_split(test_size=0.1)
    train_dataset = split_ds["train"]
    val_dataset = split_ds["test"]

    # 4) Load GPT-2 base and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    base_model.resize_token_embeddings(len(tokenizer))

    # 5) Wrap the base model in our EmotionalLoRAModel
    emotional_model = EmotionalLoRAModel(base_model, memory)

    # 6) Preprocess data
    train_dataset = train_dataset.map(
        lambda ex: preprocess_function(ex, tokenizer),
        batched=True,
        remove_columns=["emotion_label", "text"]
    )
    val_dataset = val_dataset.map(
        lambda ex: preprocess_function(ex, tokenizer),
        batched=True,
        remove_columns=["emotion_label", "text"]
    )

    # 7) Train (LoRA)
    train_model(emotional_model, tokenizer, train_dataset, val_dataset, "lora-emotion")

    # 8) Load from disk to demonstrate reloading
    loaded_peft_model = PeftModel.from_pretrained(emotional_model, "lora-emotion")
    loaded_peft_model.eval()

    # 9) Quick test
    test_messages = [
        "I'm so happy today, everything is going great!",
        "I just lost my job, I'm feeling devastated.",
        "This is so frustrating! Nothing is working out.",
        "I'm terrified of what might happen next.",
        "I'm in love with this new song, it's amazing."
    ]

    for msg in test_messages:
        # Update memory
        update_emotional_state(memory, msg, EMOTION_KEYWORDS)
        # Generate
        prompt = f"User: {msg}\nBot:"
        response = generate_response(loaded_peft_model, tokenizer, memory, prompt)
        # Display
        print(f"User: {msg}")
        print(f"Bot: {response}")
        print(f"Current Emotional State: {memory.get_state()}")
        print("-" * 50)


if __name__ == "__main__":
    main()
