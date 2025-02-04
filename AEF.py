#!/usr/bin/env python
# emotion_aware_gptneo.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from transformers import AutoTokenizer, GPTNeoForCausalLM, GPTNeoConfig
# Import GPTNeoAttention from the GPT-Neo model implementation
# (This import path may change with versions; adjust if necessary.)
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoAttention

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
    and pass them through an MLP (with sigmoid activation) to get values in [0,1].
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
    Projects the emotion vector into a bias for each attention head.
    """

    def __init__(self, emotion_dim, num_heads, head_dim, adapter_dim=32):
        super().__init__()
        self.down_proj = nn.Linear(emotion_dim, num_heads * adapter_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(adapter_dim, head_dim)
        self.num_heads = num_heads

    def forward(self, emotion_vector):
        # emotion_vector: (batch, emotion_dim)
        x = self.down_proj(emotion_vector)  # shape: (batch, num_heads * adapter_dim)
        x = self.activation(x)
        x = x.view(-1, self.num_heads, -1)  # (batch, num_heads, adapter_dim)
        bias = self.up_proj(x)  # (batch, num_heads, head_dim)
        return bias


# ------------------------------
# 4. Custom Attention with Emotion Injection for GPT-Neo
# ------------------------------
class EmotionAwareGPTNeoAttention(GPTNeoAttention):
    """
    Subclass GPTNeoAttention to inject an emotion-derived bias into the attention logits.
    """

    def __init__(self, config, emotion_adapter, global_emotion_state):
        super().__init__(config)
        self.emotion_adapter = emotion_adapter
        # global_emotion_state: a tensor (batch, emotion_dim); updated externally.
        self.global_emotion_state = global_emotion_state

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        # Follow GPT-Neo attention: compute query, key, value
        # The original GPT-Neo attention module computes these via its internal linear layers.
        # For simplicity, we call the parent method to get the initial attention logits.
        # However, we then inject our emotion bias.

        # Compute query, key, value using GPT-Neo internals
        # (This is based on GPTNeoAttention implementation; adjust if necessary.)
        query = self.query(hidden_states)  # shape: (batch, seq_len, num_heads * head_dim)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        query = self._split_heads(query, self.num_heads, self.head_dim)  # (batch, num_heads, seq_len, head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Standard scaled dot-product attention logits
        attn_scores = torch.matmul(query, key.transpose(-1, -2))  # (batch, num_heads, seq_len, seq_len)
        attn_scores = attn_scores / (float(self.head_dim) ** 0.5)

        # Inject emotion bias
        # Get bias from adapter (shape: (batch, num_heads, head_dim))
        emotion_bias = self.emotion_adapter(self.global_emotion_state)
        # Compute a bias score: dot product between query and emotion_bias.
        # query: (batch, num_heads, seq_len, head_dim)
        # emotion_bias: (batch, num_heads, head_dim) -> unsqueeze seq_len dimension.
        bias_score = (query * emotion_bias.unsqueeze(2)).sum(dim=-1)  # (batch, num_heads, seq_len)
        # Expand bias_score to add to attn_scores along the key dimension.
        attn_scores = attn_scores + bias_score.unsqueeze(-1)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, value)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out(attn_output)

        outputs = (attn_output, attn_probs) if output_attentions else (attn_output,)
        return outputs


# ------------------------------
# 5. Emotion-Aware GPT-Neo Model
# ------------------------------
class EmotionAwareGPTNeoLMHeadModel(GPTNeoForCausalLM):
    """
    A GPT-Neo model that integrates:
      - Latent emotion extraction,
      - Recurrent emotion state updating,
      - And emotion injection into attention.
    """

    def __init__(self, config, emotion_adapter, emotion_extractor, emotion_updater):
        super().__init__(config)
        self.emotion_adapter = emotion_adapter
        self.emotion_extractor = emotion_extractor
        self.emotion_updater = emotion_updater

        # Replace attention modules in each transformer block with our emotion-aware version.
        for block in self.transformer.h:
            block.attn = EmotionAwareGPTNeoAttention(config, self.emotion_adapter, self.get_global_emotion_state())

    def get_global_emotion_state(self):
        # For batch size 1: if not set, return a zero vector.
        if self.emotion_updater.state is None:
            return torch.zeros(1, self.emotion_updater.emotion_dim).to(device)
        return self.emotion_updater.state

    def update_emotion_state(self, hidden_states):
        """
        Update the global emotion state based on hidden states from the prompt.
        """
        # Extract latent emotion from hidden states.
        emotion_vector = self.emotion_extractor(hidden_states)  # (batch, emotion_dim)
        updated_state = self.emotion_updater(emotion_vector)
        # Propagate updated state to all attention modules.
        for block in self.transformer.h:
            if hasattr(block.attn, 'global_emotion_state'):
                block.attn.global_emotion_state = updated_state
        return updated_state


# ------------------------------
# 6. Instantiate Components and Model
# ------------------------------
# For local testing, we'll use the smaller GPT-Neo model.
model_name = "EleutherAI/gpt-neo-125M"  # Change to "EleutherAI/gpt-neo-2.7B" if resources allow.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)
model.to(device)
model.eval()

# Get configuration details.
config = model.config
hidden_dim = config.hidden_size
num_heads = config.num_attention_heads
head_dim = hidden_dim // num_heads

# Instantiate our emotion modules.
emotion_extractor = AuxiliaryEmotionExtractor(hidden_dim, EMOTION_DIM).to(device)
emotion_updater = EmotionStateUpdater(EMOTION_DIM).to(device)
emotion_adapter = EmotionAttentionAdapter(EMOTION_DIM, num_heads, head_dim, adapter_dim=32).to(device)

# Create our Emotion-Aware GPT-Neo model.
emotion_aware_model = EmotionAwareGPTNeoLMHeadModel(config, emotion_adapter, emotion_extractor, emotion_updater)
# Load the GPT-Neo weights into our custom model.
emotion_aware_model.load_state_dict(model.state_dict(), strict=False)
emotion_aware_model.to(device)
emotion_aware_model.eval()


# ------------------------------
# 7. Custom Generation Loop
# ------------------------------
def custom_generate(model, input_ids, attention_mask=None, max_new_tokens=50, temperature=1.0, do_sample=False):
    """
    Simple token-by-token generation loop.
    Before generation, we update the global emotion state based on the input.
    """
    with torch.no_grad():
        outputs = model.transformer(input_ids, attention_mask=attention_mask)
    hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
    model.update_emotion_state(hidden_states)

    generated = input_ids
    for _ in range(max_new_tokens):
        outputs = model(generated, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab_size)
        next_token_logits = logits[:, -1, :] / temperature
        if do_sample:
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
        if (next_token == tokenizer.eos_token_id).all():
            break
    return generated


def generate_text_with_emotion(model, prompt, max_new_tokens=100, temperature=1.0, do_sample=False):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids = custom_generate(model, input_ids=inputs["input_ids"],
                                    attention_mask=inputs.get("attention_mask", None),
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature,
                                    do_sample=do_sample)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


# ------------------------------
# 8. Chat Loop Demo
# ------------------------------
def chat():
    print("Welcome to the Emotion-Aware GPT-Neo Chatbot!")
    print("Type 'quit' to exit.\n")
    conversation_history = ""

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "quit":
            print("Goodbye!")
            break

        conversation_history += f"User: {user_input}\n"
        prompt_text = conversation_history + "Assistant:"
        response = generate_text_with_emotion(emotion_aware_model, prompt_text, max_new_tokens=100, temperature=1.2, do_sample=True)

        # Extract assistant response after "Assistant:" if possible.
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
