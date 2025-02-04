#!/usr/bin/env python
# emotion_aware_gpt2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

# ------------------------------
# 0. Setup and Global Variables
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# Define emotion dimensions and names (using a consolidated eight-emotion set)
EMOTION_DIM = 8
EMOTION_NAMES = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]


# ------------------------------
# 1. Auxiliary Emotion Extractor
# ------------------------------
class AuxiliaryEmotionExtractor(nn.Module):
    """
    Extracts a latent emotion vector from hidden states.
    For simplicity, we average pool the hidden states over the sequence dimension
    and pass them through an MLP (with sigmoid activation) to produce values in [0,1].
    """

    def __init__(self, hidden_dim, emotion_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, emotion_dim)

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_dim)
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_dim)
        emotion = torch.sigmoid(self.fc(pooled))  # (batch, emotion_dim)
        return emotion


# ------------------------------
# 2. Emotion State Updater (Recurrent)
# ------------------------------
class EmotionStateUpdater(nn.Module):
    """
    Maintains a global emotion state via a GRUCell.
    """

    def __init__(self, emotion_dim):
        super().__init__()
        self.gru = nn.GRUCell(emotion_dim, emotion_dim)
        self.state = None  # will be (batch, emotion_dim)
        self.emotion_dim = emotion_dim

    def forward(self, new_emotion):
        # new_emotion: (batch, emotion_dim)
        if self.state is None:
            self.state = new_emotion
        else:
            self.state = self.gru(new_emotion, self.state)
        return self.state


# ------------------------------
# 3. Emotion Attention Adapter (LoRA-like)
# ------------------------------
class EmotionAttentionAdapter(nn.Module):
    """
    A LoRA-inspired module that projects the emotion vector into a bias for each attention head.
    This bias is then added to the attention logits to nudge the model in a given emotional direction.
    """

    def __init__(self, emotion_dim, num_heads, head_dim, adapter_dim=32):
        super().__init__()
        self.down_proj = nn.Linear(emotion_dim, num_heads * adapter_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(adapter_dim, head_dim)
        self.num_heads = num_heads

    def forward(self, emotion_vector):
        # emotion_vector: (batch, emotion_dim)
        x = self.down_proj(emotion_vector)  # (batch, num_heads * adapter_dim)
        x = self.activation(x)
        x = x.view(x.size(0), self.num_heads, -1)  # (batch, num_heads, adapter_dim)
        bias = self.up_proj(x)  # (batch, num_heads, head_dim)
        return bias


# ------------------------------
# 4. Custom Attention with Emotion Injection for GPT-2
# ------------------------------
class EmotionAwareGPT2Attention(GPT2Attention):
    """
    Extends GPT2Attention by injecting an emotion-derived bias into the attention logits.
    GPT-2 uses a single linear layer (self.c_attn) for Q, K, and V, which we split into three parts.
    This module also handles caching by returning the key/value pair (present).
    """

    def __init__(self, config, emotion_adapter, global_emotion_state):
        super().__init__(config)
        # Manually set hidden_size from config.
        self.hidden_size = config.hidden_size
        self.emotion_adapter = emotion_adapter
        self.global_emotion_state = global_emotion_state  # (batch, emotion_dim)
        self.split_size = self.hidden_size  # GPT-2 uses hidden_size as split_size for Q, K, V.

    def _split_heads(self, tensor, num_heads, head_dim):
        # Reshape from (batch, seq_len, num_heads*head_dim) to (batch, num_heads, seq_len, head_dim)
        new_shape = tensor.size()[:-1] + (num_heads, head_dim)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, head_dim):
        # Inverse of _split_heads: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads*head_dim)
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * head_dim,)
        return tensor.view(*new_shape)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None,
                use_cache=False, output_attentions=False, **kwargs):
        # 1. Compute Q, K, V from the combined projection.
        x = self.c_attn(hidden_states)  # (batch, seq_len, 3*hidden_dim)
        query, key, value = x.split(self.split_size, dim=2)

        # 2. Split heads for each: (batch, num_heads, seq_len, head_dim)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # 3. If a past key/value exists, concatenate it.
        if layer_past is not None:
            past_key, past_value = layer_past
            if past_key.dim() == 3:  # if not already split, split them
                past_key = self._split_heads(past_key, self.num_heads, self.head_dim)
                past_value = self._split_heads(past_value, self.num_heads, self.head_dim)
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        # 4. Prepare caching output.
        present = (key, value) if use_cache else None

        # 5. Compute scaled dot-product attention.
        attn_scores = torch.matmul(query, key.transpose(-1, -2))  # (batch, num_heads, seq_len, seq_len)
        attn_scores = attn_scores / (float(self.head_dim) ** 0.5)

        # 6. Compute emotion bias.
        emotion_bias = self.emotion_adapter(self.global_emotion_state)  # (batch, num_heads, head_dim)
        bias_score = (query * emotion_bias.unsqueeze(2)).sum(dim=-1)  # (batch, num_heads, seq_len)
        attn_scores = attn_scores + bias_score.unsqueeze(-1)

        # 7. Apply attention mask if provided.
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # 8. Softmax to compute attention probabilities.
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        # 9. Weighted sum of values.
        attn_output = torch.matmul(attn_probs, value)
        # 10. Merge heads.
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        # 11. Final linear projection.
        attn_output = self.c_proj(attn_output)

        # 12. Return outputs (including caching if enabled).
        if use_cache and output_attentions:
            outputs = (attn_output, present, attn_probs)
        elif use_cache:
            outputs = (attn_output, present)
        elif output_attentions:
            outputs = (attn_output, attn_probs)
        else:
            outputs = (attn_output,)
        return outputs


# ------------------------------
# 5. Emotion-Aware GPT-2 Model
# ------------------------------
class EmotionAwareGPT2LMHeadModel(GPT2LMHeadModel):
    """
    Extends GPT2LMHeadModel to integrate:
      - A latent emotion extractor,
      - A recurrent emotion state updater,
      - And a LoRA-inspired emotion adapter injected into the attention modules.
    """

    def __init__(self, config, emotion_adapter, emotion_extractor, emotion_updater):
        super().__init__(config)
        self.emotion_adapter = emotion_adapter
        self.emotion_extractor = emotion_extractor
        self.emotion_updater = emotion_updater

        # Replace each transformer block's attention module with our custom version.
        for block in self.transformer.h:
            block.attn = EmotionAwareGPT2Attention(
                config,
                self.emotion_adapter,
                self.get_global_emotion_state()
            )

    def get_global_emotion_state(self):
        # For a batch size of 1: if state is not set, return a zero vector.
        if self.emotion_updater.state is None:
            return torch.zeros(1, self.emotion_updater.emotion_dim).to(device)
        return self.emotion_updater.state

    def update_emotion_state(self, hidden_states):
        """
        Extracts a latent emotion vector from hidden_states, updates the global emotion state,
        and propagates it to all attention modules.
        """
        emotion_vector = self.emotion_extractor(hidden_states)  # (batch, emotion_dim)
        updated_state = self.emotion_updater(emotion_vector)
        for block in self.transformer.h:
            if hasattr(block.attn, 'global_emotion_state'):
                block.attn.global_emotion_state = updated_state
        return updated_state


# ------------------------------
# 6. Instantiate Components and Model
# ------------------------------
model_name = "gpt2"  # Using base GPT-2.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the pre-trained base GPT-2 model.
base_model = GPT2LMHeadModel.from_pretrained(model_name)
base_model.to(device)
base_model.eval()

# Retrieve configuration details.
config = base_model.config
hidden_dim = config.hidden_size  # e.g., 768 for GPT-2.
num_heads = config.num_attention_heads  # e.g., 12 for GPT-2.
head_dim = hidden_dim // num_heads  # e.g., 64 for GPT-2.

# Instantiate the emotion modules.
emotion_extractor = AuxiliaryEmotionExtractor(hidden_dim, EMOTION_DIM).to(device)
emotion_updater = EmotionStateUpdater(EMOTION_DIM).to(device)
emotion_adapter = EmotionAttentionAdapter(EMOTION_DIM, num_heads, head_dim, adapter_dim=32).to(device)

# Create the emotion-aware GPT-2 model.
emotion_aware_model = EmotionAwareGPT2LMHeadModel(config, emotion_adapter, emotion_extractor, emotion_updater)
emotion_aware_model.to(device)
emotion_aware_model.eval()

# Copy only matching parameters from the base model into our custom model.
base_state_dict = base_model.state_dict()
custom_state_dict = emotion_aware_model.state_dict()
for name, param in base_state_dict.items():
    if name in custom_state_dict and param.size() == custom_state_dict[name].size():
        custom_state_dict[name] = param
emotion_aware_model.load_state_dict(custom_state_dict, strict=False)


# ------------------------------
# 7. Custom Generation Loop
# ------------------------------
def custom_generate(model, input_ids, attention_mask=None, past_key_values=None, max_new_tokens=50, temperature=1.0, do_sample=False):
    """
    Generates text token-by-token.
    Before generation, the model updates its global emotion state based on the input prompt.
    Caching (past_key_values) is used to speed up generation.
    """
    with torch.no_grad():
        if past_key_values is None:
            outputs = model.transformer(input_ids, attention_mask=attention_mask, use_cache=True)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
            model.update_emotion_state(hidden_states)
            generated = input_ids
            past_key_values = outputs.past_key_values
        else:
            outputs = model.transformer(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            generated = input_ids
            past_key_values = outputs.past_key_values

        for _ in range(max_new_tokens):
            outputs = model(generated[:, -1:], attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits  # (batch, 1, vocab_size)
            next_token_logits = logits[:, -1, :] / temperature

            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat((generated, next_token), dim=1)
            if (next_token == tokenizer.eos_token_id).all():
                break

    return generated, past_key_values


def generate_text_with_emotion(model, prompt, past_key_values=None, max_new_tokens=100, temperature=1.0, do_sample=False):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids, past_key_values = custom_generate(model, input_ids=inputs["input_ids"],
                                                     attention_mask=inputs.get("attention_mask", None),
                                                     past_key_values=past_key_values,
                                                     max_new_tokens=max_new_tokens,
                                                     temperature=temperature,
                                                     do_sample=do_sample)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True), past_key_values


# ------------------------------
# 8. Chat Loop Demo
# ------------------------------
def chat():
    print("Welcome to the Emotion-Aware GPT-2 Chatbot!")
    print("Type 'quit' to exit.\n")
    conversation_history = ""
    past_key_values = None

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "quit":
            print("Goodbye!")
            break

        conversation_history += f"User: {user_input}\n"
        prompt_text = conversation_history + "Assistant:"
        response, past_key_values = generate_text_with_emotion(emotion_aware_model, prompt_text, past_key_values=past_key_values, max_new_tokens=100, temperature=1.2, do_sample=True)
        if "Assistant:" in response:
            assistant_response = response.split("Assistant:")[-1].strip()
        else:
            assistant_response = response.strip()

        conversation_history += f"Assistant: {assistant_response}\n"

        current_emotion = emotion_aware_model.get_global_emotion_state().squeeze(0).detach().cpu().numpy()
        emotion_str = ", ".join([f"{name}: {val:.2f}" for name, val in zip(EMOTION_NAMES, current_emotion)])
        print("Current Emotion State:", emotion_str)
        print("Assistant:", assistant_response, "\n")


if __name__ == "__main__":
    chat()
