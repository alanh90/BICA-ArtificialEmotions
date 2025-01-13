import os
import time
import json
import logging
import threading
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    pipeline
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('emotional_llm.log')
    ]
)
logger = logging.getLogger(__name__)

# --------------------------
# CONFIG
# --------------------------
MODEL_NAME = "gpt2"
EMOTION_TAGS = ["anger", "fear", "joy", "love", "sadness", "surprise", "neutral"]
MAX_SAMPLES_PER_EMOTION = 3000
TRAIN_EPOCHS = 5
BATCH_SIZE = 8
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
MAX_LENGTH = 128
LEARNING_RATE = 2e-4
SAVE_DIR = Path("model_outputs")
SAVE_DIR.mkdir(exist_ok=True)

# Base decay rates (per second)
BASE_DECAY_RATES = {
    "anger": 0.0002,
    "fear": 0.00025,
    "joy": 0.0003,
    "love": 0.00015,
    "sadness": 0.0001,
    "surprise": 0.0004,
    "neutral": 0.0001,
}

# Cross-emotion influence matrix
EMOTION_INFLUENCES = {
    "anger": {"joy": -0.2, "love": -0.3, "neutral": 0.1},
    "fear": {"anger": 0.1, "joy": -0.2, "neutral": -0.1},
    "joy": {"sadness": -0.3, "anger": -0.2, "fear": -0.2},
    "love": {"anger": -0.3, "fear": -0.2, "neutral": 0.1},
    "sadness": {"joy": -0.2, "love": -0.1, "neutral": -0.1},
    "surprise": {"neutral": -0.2},
    "neutral": {},  # neutral emotion doesn't influence others
}

# Temperature scaling parameters
TEMP_CONFIG = {
    "base_temp": 0.7,
    "intensity_scale_factor": 0.3,
    "neutral_scale_factor": 0.3
}


# --------------------------
# Utility Functions
# --------------------------

class GPUMonitor:
    """Monitors GPU usage and memory"""

    @staticmethod
    def is_gpu_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def get_memory_stats() -> Dict[str, int]:
        if not GPUMonitor.is_gpu_available():
            return {}

        return {
            "allocated": torch.cuda.memory_allocated() // 1024 // 1024,  # MB
            "cached": torch.cuda.memory_reserved() // 1024 // 1024,  # MB
            "max_allocated": torch.cuda.max_memory_allocated() // 1024 // 1024
        }

    @staticmethod
    def log_gpu_stats():
        if not GPUMonitor.is_gpu_available():
            return

        stats = GPUMonitor.get_memory_stats()
        logger.info(
            f"GPU Memory (MB) - Allocated: {stats['allocated']}, "
            f"Cached: {stats['cached']}, Max: {stats['max_allocated']}"
        )


class EmotionalState:
    """Represents a single snapshot of emotional state with metadata"""

    def __init__(self, emotions: Dict[str, float], timestamp: float):
        self.emotions = emotions
        self.timestamp = timestamp

    def to_vector(self, emotion_tags: List[str]) -> np.ndarray:
        """Convert to normalized vector representation"""
        return np.array([self.emotions.get(e, 0.0) for e in emotion_tags])

    def to_dict(self) -> Dict[str, Union[Dict[str, float], float]]:
        """Convert to serializable dictionary"""
        return {
            "emotions": self.emotions,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Union[Dict[str, float], float]]) -> 'EmotionalState':
        """Create EmotionalState from dictionary"""
        return cls(
            emotions=data["emotions"],
            timestamp=data["timestamp"]
        )


# --------------------------
# Model Architecture
# --------------------------

class MultiHeadEmotionProjection(nn.Module):
    """
    Projects emotional state into transformer hidden space using
    multiple attention heads for different emotional aspects
    """

    def __init__(
            self,
            num_emotions: int,
            hidden_size: int,
            num_heads: int = 4,
            dropout: float = 0.1
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        # Initial projection to full hidden size
        self.initial_projection = nn.Sequential(
            nn.Linear(num_emotions, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Head projections project from full hidden size to head dimension
        self.head_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, self.head_dim),
                nn.LayerNorm(self.head_dim),
                nn.Tanh(),
            )
            for _ in range(num_heads)
        ])

        # Learned head mixing weights
        self.head_mixing = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.mix_norm = nn.LayerNorm(hidden_size)

        # Final output layers
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh()
        )

        # Initialize with small weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, emotion_vector: torch.Tensor) -> torch.Tensor:
        """
        Project emotions through multiple heads and combine

        Args:
            emotion_vector: [batch_size, num_emotions]

        Returns:
            projected: [batch_size, hidden_size]
        """
        batch_size = emotion_vector.shape[0]
        logger.debug(f"Emotion vector shape: {emotion_vector.shape}")

        # Initial projection to hidden size
        hidden = self.initial_projection(emotion_vector)  # [batch_size, hidden_size]
        logger.debug(f"After initial projection: {hidden.shape}")

        # Project through each head
        head_outputs = []
        for head_proj in self.head_projections:
            head_output = head_proj(hidden)  # [batch_size, head_dim]
            head_outputs.append(head_output)

        # Concatenate head outputs
        multi_head = torch.cat(head_outputs, dim=-1)  # [batch_size, num_heads * head_dim]
        logger.debug(f"After head concatenation: {multi_head.shape}")

        # Mix heads
        mixed = self.head_mixing(multi_head)  # [batch_size, hidden_size]
        mixed = self.mix_norm(mixed)
        logger.debug(f"After head mixing: {mixed.shape}")

        # Final output projection
        output = self.output(mixed)  # [batch_size, hidden_size]
        logger.debug(f"Final output shape: {output.shape}")

        return output


# --------------------------
# Emotional Memory System
# --------------------------

class EmotionalMemory:
    """
    Enhanced emotional memory system with:
    - Cross-emotion influences
    - Historical state tracking
    - Emotional momentum
    - Contextual decay rates
    """

    def __init__(
            self,
            emotions: List[str],
            base_decay_rates: Dict[str, float],
            influence_matrix: Dict[str, Dict[str, float]],
            history_size: int = 10
    ):
        self.emotions = emotions
        self.base_decay_rates = base_decay_rates
        self.influence_matrix = influence_matrix
        self.history_size = history_size

        # Current state and history
        self.state = {e: 0.0 for e in emotions}
        self.history: List[EmotionalState] = []
        self.momentum = {e: 0.0 for e in emotions}

        # Thread safety
        self.lock = threading.Lock()
        self.last_update_time = time.time()

        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None
            )
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {e}")
            self.sentiment_analyzer = None

        # Start decay thread
        self.decay_thread = threading.Thread(target=self._continuous_decay, daemon=True)
        self.decay_thread.start()

    def update(self, emotion: str, intensity: float) -> None:
        """Update a single emotion and apply cross-emotion influences"""
        with self.lock:
            if emotion not in self.state:
                return

            # Update primary emotion
            old_value = self.state[emotion]
            self.state[emotion] = min(1.0, max(0.0, old_value + intensity))

            # Update momentum
            self.momentum[emotion] = self.state[emotion] - old_value

            # Apply influences to other emotions
            if emotion in self.influence_matrix:
                for target, strength in self.influence_matrix[emotion].items():
                    if target in self.state:
                        influence = strength * intensity
                        self.state[target] = min(
                            1.0, max(0.0, self.state[target] + influence)
                        )

            # Save state to history
            self._update_history()

    def update_from_text(self, text: str) -> None:
        """Update emotional state based on text sentiment"""
        if not self.sentiment_analyzer:
            logger.warning("Sentiment analyzer not available")
            return

        try:
            results = self.sentiment_analyzer(text)[0]
            emotion_scores = {label['label']: label['score'] for label in results}

            # Update each matching emotion
            for emotion in self.emotions:
                if emotion in emotion_scores:
                    self.update(emotion, emotion_scores[emotion])
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")

    def get_state(self) -> Dict[str, float]:
        """Get current emotional state"""
        with self.lock:
            return dict(self.state)

    def get_momentum(self) -> Dict[str, float]:
        """Get emotional momentum (rate of change)"""
        with self.lock:
            return dict(self.momentum)

    def get_history(self) -> List[EmotionalState]:
        """Get emotional history"""
        with self.lock:
            return list(self.history)

    def save_state(self, filepath: Union[str, Path]) -> None:
        """Save emotional state to file"""
        filepath = Path(filepath)
        state_data = {
            "current_state": self.get_state(),
            "history": [state.to_dict() for state in self.get_history()],
            "momentum": self.get_momentum(),
            "last_update_time": self.last_update_time
        }

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save emotional state: {e}")

    def load_state(self, filepath: Union[str, Path]) -> None:
        """Load emotional state from file"""
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)

            with self.lock:
                self.state = state_data["current_state"]
                self.history = [
                    EmotionalState.from_dict(h) for h in state_data["history"]
                ]
                self.momentum = state_data["momentum"]
                self.last_update_time = state_data["last_update_time"]
        except Exception as e:
            logger.error(f"Failed to load emotional state: {e}")

    def _update_history(self) -> None:
        """Add current state to history"""
        current_state = EmotionalState(
            emotions=dict(self.state),
            timestamp=time.time()
        )
        self.history.append(current_state)
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def _get_contextual_decay_rate(self, emotion: str) -> float:
        """
        Calculate decay rate based on:
        - Base decay rate
        - Current intensity
        - Emotional momentum
        - Recent history
        """
        base_rate = self.base_decay_rates[emotion]

        # Adjust based on current intensity (higher intensities decay faster)
        intensity_factor = 1.0 + self.state[emotion]

        # Adjust based on momentum (faster change = faster decay)
        momentum_factor = 1.0 + abs(self.momentum[emotion])

        # Adjust based on recent history volatility
        if len(self.history) > 1:
            recent_values = [state.emotions[emotion] for state in self.history[-5:]]
            volatility = np.std(recent_values)
            history_factor = 1.0 + volatility
        else:
            history_factor = 1.0

        return base_rate * intensity_factor * momentum_factor * history_factor

    def _continuous_decay(self) -> None:
        """Background thread for continuous emotional decay"""
        while True:
            try:
                with self.lock:
                    now = time.time()
                    elapsed = now - self.last_update_time

                    # Apply decay to each emotion
                    for emotion in self.emotions:
                        decay_rate = self._get_contextual_decay_rate(emotion)
                        decay_amount = decay_rate * elapsed
                        old_value = self.state[emotion]
                        self.state[emotion] = max(0.0, old_value - decay_amount)

                        # Update momentum
                        self.momentum[emotion] = self.state[emotion] - old_value

                    self.last_update_time = now
                    self._update_history()

            except Exception as e:
                logger.error(f"Error in decay thread: {e}")

            time.sleep(0.1)  # Check every 100ms


# --------------------------
# Enhanced Emotional Model
# --------------------------

class EmotionalLoRAModel(GPT2LMHeadModel):
    """
    Enhanced emotional model with simplified architecture and robust error handling
    """

    def __init__(
            self,
            base_model: GPT2LMHeadModel,
            memory: EmotionalMemory,
            emotion_projection_dropout: float = 0.1
    ):
        super().__init__(base_model.config)

        # Copy GPT-2 modules
        self.transformer = base_model.transformer
        self.lm_head = base_model.lm_head

        # Get model dimensions
        self.n_heads = base_model.config.n_head  # Usually 12 for GPT2
        self.hidden_size = base_model.config.n_embd  # 768 for GPT2
        self.head_dim = self.hidden_size // self.n_heads

        # Emotional components
        self.memory = memory
        self.emotions = list(self.memory.state.keys())
        self.num_emotions = len(self.emotions)

        # Simplified emotion projection
        self.emotion_projection = nn.Sequential(
            nn.Linear(self.num_emotions, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(emotion_projection_dropout),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )

        # Attention biasing - properly shaped and scaled
        self.emotional_attention_bias = nn.Parameter(
            torch.randn(self.num_emotions, self.n_heads) * 0.02  # [7, 12] for GPT2
        )

        # Temperature scaling parameters
        self.temp_scaling = nn.Sequential(
            nn.Linear(self.num_emotions, 1),
            nn.Sigmoid()
        )

        # Initialize our custom weights
        self._init_emotional_weights()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for modules - compatible with PyTorch's apply()"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _init_emotional_weights(self) -> None:
        """Initialize weights for emotional components"""
        # Initialize emotion projection
        for layer in self.emotion_projection:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

        # Initialize attention bias with small values
        nn.init.normal_(self.emotional_attention_bias, mean=0.0, std=0.02)

        # Initialize temperature scaling
        for layer in self.temp_scaling:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def _get_emotion_vector(self, batch_size: int = 1) -> torch.Tensor:
        """Get current emotional state as tensor with proper device placement"""
        state = self.memory.get_state()
        vector = torch.tensor(
            [state[e] for e in self.emotions],
            device=self.device,
            dtype=torch.float32
        )
        # Expand for batch size if needed
        if batch_size > 1:
            vector = vector.unsqueeze(0).expand(batch_size, -1)
        else:
            vector = vector.unsqueeze(0)
        return vector

    def _get_attention_bias(self, emotion_vector: torch.Tensor) -> torch.Tensor:
        """
        Calculate attention bias with proper broadcasting
        Args:
            emotion_vector: shape [batch_size, num_emotions]
        Returns:
            bias: shape [batch_size, n_layer, n_head]
        """
        # Transpose emotional_attention_bias for correct multiplication
        # emotion_vector: [batch_size, num_emotions]
        # emotional_attention_bias: [num_emotions, n_head] -> [n_head, num_emotions]
        attention_weights = self.emotional_attention_bias.t()  # [n_head, num_emotions]

        # Project emotions to attention space
        # [batch_size, num_emotions] @ [n_head, num_emotions].t() -> [batch_size, n_head]
        bias = torch.matmul(emotion_vector, attention_weights.t())

        # Scale the bias to be subtle
        bias = 0.1 * torch.tanh(bias)  # Keep values in reasonable range

        # Expand for layers: [batch_size, 1, n_head] -> [batch_size, n_layer, n_head]
        return bias.unsqueeze(1).expand(-1, self.config.n_layer, -1)

    def _get_temperature(self, emotion_vector: torch.Tensor) -> torch.Tensor:
        """Calculate dynamic temperature scaling"""
        return 0.7 + (0.6 * self.temp_scaling(emotion_vector))

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            **kwargs
    ) -> CausalLMOutputWithCrossAttentions:

        if input_ids is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )

        try:
            # Get batch info
            batch_size = input_ids.shape[0] if input_ids is not None else 1

            # Get emotional state
            emotion_vector = self._get_emotion_vector(batch_size)

            # Get embeddings
            inputs_embeds = self.transformer.wte(input_ids)

            # Project and add emotional context
            emotion_hidden = self.emotion_projection(emotion_vector)
            emotion_hidden = emotion_hidden.unsqueeze(1)  # Add sequence dimension

            # Broadcast to sequence length
            seq_len = inputs_embeds.shape[1]
            emotion_hidden = emotion_hidden.expand(-1, seq_len, -1)

            # Combine embeddings
            inputs_embeds = inputs_embeds + 0.1 * emotion_hidden  # Scaled addition

            # Get attention bias
            attention_bias = self._get_attention_bias(emotion_vector)

            # Forward through transformer
            transformer_outputs = self.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                head_mask=attention_bias,
                **kwargs
            )

            # Get logits
            hidden_states = transformer_outputs[0]
            lm_logits = self.lm_head(hidden_states)

            # Calculate loss if needed
            loss = None
            if labels is not None:
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )

        except Exception as e:
            logger.error(f"Forward pass error: {e}")
            raise

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.Tensor,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for generation step"""
        return {
            "input_ids": input_ids,
            **kwargs
        }

# --------------------------
# Dataset Processing
# --------------------------

def process_go_emotions(dataset, emotion_tags: List[str], max_samples: int) -> "Dataset":
    """Process GoEmotions dataset"""
    logger.info("Processing GoEmotions dataset...")

    # Get valid label indices
    label_names = dataset.features["labels"].feature.names
    valid_label_indices = [label_names.index(e) for e in emotion_tags if e in label_names]

    def is_relevant(row):
        """Keep samples with exactly one of our target emotions"""
        return len(set(row["labels"]).intersection(valid_label_indices)) == 1

    # Filter and limit samples
    filtered = dataset.filter(is_relevant)
    filtered = filtered.select(range(min(max_samples, len(filtered))))

    def create_prompt(row):
        """Create conversation format"""
        emotion_idx = row["labels"][0]
        emotion_str = label_names[emotion_idx]
        return {
            "text": f"User: {row['text']}\nBot:",
            "emotion_label": emotion_str
        }

    return filtered.map(create_prompt)


def process_empathetic_dialogues(dataset, emotion_tags: List[str], max_samples: int) -> "Dataset":
    """Process EmpatheticDialogues dataset"""
    logger.info("Processing EmpatheticDialogues dataset...")

    # Map EmpatheticDialogues emotions to our emotion space
    emotion_mapping = {
        "angry": "anger",
        "afraid": "fear",
        "happy": "joy",
        "sad": "sadness",
        "surprised": "surprise",
        "caring": "love",
        "neutral": "neutral",
        "annoyed": "anger",
        "anxious": "fear",
        "content": "joy",
        "devastated": "sadness",
        "excited": "joy",
        "grateful": "love",
        "lonely": "sadness",
        "proud": "joy",
        "terrified": "fear"
    }

    def process_row(row):
        """Map emotions and format conversation"""
        if row["emotion"] in emotion_mapping:
            mapped_emotion = emotion_mapping[row["emotion"]]
            if mapped_emotion in emotion_tags:
                return {
                    "text": f"User: {row['utterance']}\nBot:",
                    "emotion_label": mapped_emotion
                }
        return None

    # Process and filter
    processed = dataset.map(process_row)
    filtered = processed.filter(lambda x: x is not None)

    # Limit samples
    return filtered.select(range(min(max_samples, len(filtered))))


def load_emotional_datasets(emotion_tags: List[str], max_samples: int) -> "Dataset":
    """Load and combine multiple emotion datasets"""
    from datasets import concatenate_datasets

    datasets_to_combine = []

    # Load GoEmotions
    logger.info("Loading GoEmotions dataset...")
    try:
        go_emotions = load_dataset("go_emotions", "simplified")["train"]
        processed_go = process_go_emotions(go_emotions, emotion_tags, max_samples)
        datasets_to_combine.append(processed_go)
        logger.info(f"GoEmotions samples: {len(processed_go)}")
    except Exception as e:
        logger.warning(f"Could not load GoEmotions: {e}")

    # Load EmpatheticDialogues
    logger.info("Loading EmpatheticDialogues dataset...")
    try:
        empathetic = load_dataset("empathetic_dialogues", trust_remote_code=True)["train"]
        processed_emp = process_empathetic_dialogues(empathetic, emotion_tags, max_samples)
        datasets_to_combine.append(processed_emp)
        logger.info(f"EmpatheticDialogues samples: {len(processed_emp)}")
    except Exception as e:
        logger.warning(f"Could not load EmpatheticDialogues: {e}")

    if not datasets_to_combine:
        raise ValueError("No datasets could be loaded")

    # Combine datasets
    logger.info("Combining datasets...")
    combined = concatenate_datasets(datasets_to_combine)
    logger.info(f"Total combined samples: {len(combined)}")

    # Shuffle
    combined = combined.shuffle(seed=42)

    return combined


def preprocess_for_training(examples: Dict, tokenizer, max_length: int) -> Dict:
    """Preprocess examples for training"""
    return tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )


# --------------------------
# Training Functions
# --------------------------

def create_emotion_lora_config(
        r: int = LORA_R,
        alpha: int = LORA_ALPHA,
        dropout: float = LORA_DROPOUT
) -> LoraConfig:
    """Create LoRA config optimized for emotional fine-tuning"""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=[
            "c_attn",  # Attention weights
            "c_proj",  # Output projection
            "c_fc"     # Feed-forward
        ],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def train_emotional_model(
        emotional_model: EmotionalLoRAModel,
        tokenizer,
        train_dataset,
        val_dataset,
        output_dir: str = "emotional-lora",
        num_epochs: int = TRAIN_EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE
) -> PeftModel:
    """Train the emotional model with LoRA"""
    logger.info("Initializing LoRA training...")

    # Create LoRA config
    lora_config = create_emotion_lora_config()

    # Wrap model with LoRA
    logger.info("Applying LoRA to model...")
    peft_model = get_peft_model(emotional_model, lora_config)

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Configure training
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=200,
        save_total_limit=3,
        logging_steps=50,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        gradient_accumulation_steps=4,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training loop...")
    trainer.train()

    # Save final model
    logger.info("Saving final model...")
    final_dir = Path(output_dir) / "final"
    peft_model.save_pretrained(final_dir)

    return peft_model


# --------------------------
# Training System
# --------------------------

class EmotionalTrainer:
    """Manages the training process for the emotional model"""

    def __init__(
            self,
            model_name: str = MODEL_NAME,
            emotion_tags: List[str] = EMOTION_TAGS,
            max_samples: int = MAX_SAMPLES_PER_EMOTION,
            output_dir: Union[str, Path] = SAVE_DIR
    ):
        self.model_name = model_name
        self.emotion_tags = emotion_tags
        self.max_samples = max_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.tokenizer = None
        self.base_model = None
        self.emotional_model = None
        self.memory = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self) -> None:
        """Initialize all components"""
        logger.info("Setting up training components...")

        # Initialize memory system
        self.memory = EmotionalMemory(
            emotions=self.emotion_tags,
            base_decay_rates=BASE_DECAY_RATES,
            influence_matrix=EMOTION_INFLUENCES
        )

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        logger.info("Loading base model...")
        self.base_model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        # Create emotional model
        logger.info("Creating emotional model...")
        self.emotional_model = EmotionalLoRAModel(
            base_model=self.base_model,
            memory=self.memory
        )

    def load_datasets(self) -> None:
        """Load and prepare datasets"""
        logger.info("Loading datasets...")
        try:
            dataset = load_emotional_datasets(
                self.emotion_tags,
                self.max_samples
            )

            # Split dataset
            split = dataset.train_test_split(test_size=0.1)
            self.train_dataset = split["train"]
            self.val_dataset = split["test"]

            # Preprocess
            logger.info("Preprocessing datasets...")
            self.train_dataset = self.train_dataset.map(
                lambda ex: preprocess_for_training(ex, self.tokenizer, MAX_LENGTH),
                batched=True,
                remove_columns=dataset.column_names,
                desc="Preprocessing training data"
            )
            self.val_dataset = self.val_dataset.map(
                lambda ex: preprocess_for_training(ex, self.tokenizer, MAX_LENGTH),
                batched=True,
                remove_columns=dataset.column_names,
                desc="Preprocessing validation data"
            )
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise

    def train(self) -> PeftModel:
        """Train the model"""
        logger.info("Starting training...")
        try:
            trained_model = train_emotional_model(
                emotional_model=self.emotional_model,
                tokenizer=self.tokenizer,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                output_dir=self.output_dir / "checkpoints"
            )
            return trained_model
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise

    def run_training_pipeline(self) -> PeftModel:
        """Run complete training pipeline"""
        try:
            self.setup()
            self.load_datasets()
            return self.train()
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise


# --------------------------
# Chat Interface
# --------------------------

class EmotionalChat:
    """Interactive chat interface with emotional awareness"""

    def __init__(
            self,
            model: PeftModel,
            tokenizer,
            memory: EmotionalMemory,
            save_dir: Union[str, Path] = SAVE_DIR,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.memory = memory
        self.save_dir = Path(save_dir)
        self.device = device
        self.conversation_history = []

        # Monitor
        self.gpu_monitor = GPUMonitor()

        # Ensure save directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def generate_response(
            self,
            user_input: str,
            max_length: int = 100,
            min_length: int = 20,
    ) -> str:
        """Generate response with emotional awareness"""
        try:
            # Update emotional state from input
            self.memory.update_from_text(user_input)

            # Build context from history
            context = "\n".join(self.conversation_history[-5:])  # Last 5 turns
            prompt = f"{context}\nUser: {user_input}\nBot:" if context else f"User: {user_input}\nBot:"

            # Prepare inputs
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(self.device)

            # Get temperature from emotional state
            # FIXED: Use self.model._get_temperature_scaling() instead of self.model.base_model._get_temperature_scaling()
            temperature = self.model._get_temperature_scaling()

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=50,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Extract just the bot's response
            response = response.split("Bot:")[-1].strip()

            # Update history
            self.conversation_history.append(f"User: {user_input}")
            self.conversation_history.append(f"Bot: {response}")

            # Update emotional state from response
            self.memory.update_from_text(response)

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"

    def save_conversation(self) -> None:
        """Save conversation history and emotional state"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Save conversation
        conv_file = self.save_dir / f"conversation_{timestamp}.txt"
        try:
            with open(conv_file, 'w') as f:
                f.write("\n".join(self.conversation_history))
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")

        # Save emotional state
        state_file = self.save_dir / f"emotional_state_{timestamp}.json"
        self.memory.save_state(state_file)

    def load_conversation(self, conv_file: Union[str, Path], state_file: Union[str, Path]) -> None:
        """Load conversation history and emotional state"""
        try:
            # Load conversation
            with open(conv_file, 'r') as f:
                self.conversation_history = f.read().splitlines()

            # Load emotional state
            self.memory.load_state(state_file)
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")

    def get_emotional_state(self) -> Dict[str, float]:
        """Get current emotional state"""
        return self.memory.get_state()

    def get_emotional_history(self) -> List[Dict[str, Union[Dict[str, float], float]]]:
        """Get emotional history"""
        return [state.to_dict() for state in self.memory.get_history()]

    def plot_emotional_state(self, save_path: Optional[Path] = None) -> None:
        """Plot current emotional state and history"""
        try:
            import matplotlib.pyplot as plt
            history = self.memory.get_history()
            if not history:
                logger.warning("No emotional history to plot")
                return

            # Prepare data
            emotions = list(history[0].emotions.keys())
            timestamps = [(state.timestamp - history[0].timestamp) for state in history]

            # Create plot
            plt.figure(figsize=(12, 6))
            for emotion in emotions:
                values = [state.emotions[emotion] for state in history]
                plt.plot(timestamps, values, label=emotion, marker='.')

            plt.xlabel('Time (seconds)')
            plt.ylabel('Intensity')
            plt.title('Emotional State History')
            plt.legend()
            plt.grid(True)

            # Save or show
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib not installed, skipping plot")
        except Exception as e:
            logger.error(f"Error plotting emotional state: {e}")


# --------------------------
# Main Execution
# --------------------------

def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"emotional_llm_{timestamp}.log"

    logger = logging.getLogger("emotional_llm")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def run_interactive_session(chat: EmotionalChat) -> None:
    """Run interactive chat session"""
    logger.info("Starting interactive chat session")
    print("\nEmotional Chat Interface")
    print("------------------------")
    print("Enter 'quit' to exit, 'save' to save current session")
    print("Current emotional state:", chat.get_emotional_state())

    last_plot_time = time.time()
    plot_interval = 60  # Plot every minute

    try:
        while True:
            # Get user input
            user_input = input("\nUser: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                break

            if user_input.lower() == 'save':
                chat.save_conversation()
                print("Session saved!")
                continue

            try:
                # Generate response
                response = chat.generate_response(user_input)
                print(f"\nBot: {response}")

                # Show emotional state
                current_state = chat.get_emotional_state()
                print("\nEmotional state:", current_state)

                # Periodic state plotting
                current_time = time.time()
                if current_time - last_plot_time > plot_interval:
                    plot_path = Path("plots") / f"emotional_state_{time.strftime('%Y%m%d-%H%M%S')}.png"
                    chat.plot_emotional_state(plot_path)
                    last_plot_time = current_time

                # Log GPU stats if available
                if GPUMonitor.is_gpu_available():
                    GPUMonitor.log_gpu_stats()

            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"Error: {str(e)}")

    except KeyboardInterrupt:
        print("\nEnding chat session...")
    finally:
        # Save final state
        chat.save_conversation()
        logger.info("Chat session ended")


def main():
    """Main execution function"""
    try:
        # Setup logging
        global logger
        logger = setup_logging()
        logger.info("Starting Emotional LLM System")

        # Initialize trainer
        trainer = EmotionalTrainer()

        # Train model
        logger.info("Starting training pipeline...")
        trained_model = trainer.run_training_pipeline()
        logger.info("Training complete")

        # Initialize chat interface
        chat = EmotionalChat(
            model=trained_model,
            tokenizer=trainer.tokenizer,
            memory=trainer.memory
        )

        # Run interactive session
        run_interactive_session(chat)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        logger.info("System shutdown complete")


if __name__ == "__main__":
    main()
