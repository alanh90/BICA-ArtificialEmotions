import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional
import logging


class EmotionalDimensionExtractor:
    """Extract and analyze emotional dimensions from LLM"""

    def __init__(self, model_name: str = "gpt2-medium"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Store extracted dimensions
        self.emotional_subspace = None
        self.emotion_vectors = None
        self.pca = None  # Initialize PCA attribute

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def generate_emotional_probes(self) -> List[Dict[str, List[str]]]:
        """Generate structured probe sentences for emotional analysis"""
        emotion_probes = {
            "joy": [
                "I feel extremely happy because",
                "This brings me great joy since",
                "I'm absolutely delighted that",
                "This makes me feel wonderful because",
            ],
            "sadness": [
                "I feel very sad because",
                "This makes me quite depressed since",
                "I'm really down about",
                "It breaks my heart that",
            ],
            "anger": [
                "I'm absolutely furious that",
                "This makes me so angry because",
                "I'm really frustrated by",
                "It infuriates me when",
            ],
            "fear": [
                "I'm really scared about",
                "This terrifies me because",
                "I'm quite anxious regarding",
                "I'm deeply worried that",
            ],
            "surprise": [
                "I'm completely shocked that",
                "This really surprised me because",
                "I can't believe that",
                "I'm amazed to learn",
            ],
            "love": [
                "I deeply care about",
                "This fills me with love because",
                "I'm so attached to",
                "I feel such affection for",
            ],
            "neutral": [
                "I am thinking about",
                "I am considering",
                "I am analyzing",
                "I am evaluating",
            ]
        }
        return emotion_probes

    def get_activation_patterns(
        self,
        text: str,
        layer_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Extract activation patterns from specific layers for a given input."""

        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # If no specific layers selected, use last 4 layers
        if layer_indices is None:
            layer_indices = [-4, -3, -2, -1]

        activations = []

        # Get model activations (no grad needed for analysis)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True
            )
            hidden_states = outputs.hidden_states
            for layer_idx in layer_indices:
                layer_activations = hidden_states[layer_idx]
                # Average over sequence length
                mean_activation = layer_activations.mean(dim=1)
                activations.append(mean_activation)

        # Concatenate all layer activations
        combined_activation = torch.cat(activations, dim=-1)
        return combined_activation

    def extract_emotional_subspace(
        self,
        n_components: int = 7  # one PCA component per basic emotion
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract principal emotional dimensions and emotion vectors
        by running PCA on the hidden-state activations of various
        emotion-specific probes.
        """
        # Set print precision for debugging (optional)
        torch.set_printoptions(precision=8)
        np.set_printoptions(precision=8, suppress=True)

        self.logger.info("Extracting emotional subspace...")

        # Get probe sentences
        emotion_probes = self.generate_emotional_probes()

        # Collect activations for all probes
        all_activations = []
        emotion_labels = []

        for emotion, probes in emotion_probes.items():
            self.logger.info(f"Processing {emotion} probes...")
            for probe in probes:
                activation = self.get_activation_patterns(probe)
                all_activations.append(activation.cpu().numpy())
                emotion_labels.append(emotion)

        # Convert to numpy array
        activation_matrix = np.vstack(all_activations)

        # Perform PCA to find principal emotional dimensions
        pca = PCA(n_components=n_components)
        emotional_subspace = pca.fit_transform(activation_matrix)
        self.pca = pca  # <-- IMPORTANT: store the fitted PCA object

        # Calculate average emotion vectors in the reduced space
        emotion_vectors = {}
        for emotion in set(emotion_labels):
            emotion_indices = [i for i, e in enumerate(emotion_labels) if e == emotion]
            emotion_vectors[emotion] = np.mean(
                emotional_subspace[emotion_indices], axis=0
            )

        self.emotional_subspace = emotional_subspace
        self.emotion_vectors = emotion_vectors

        return emotional_subspace, emotion_vectors

    def analyze_text_emotions(self, text: str) -> Dict[str, float]:
        """Analyze emotional content of input text."""
        if self.emotional_subspace is None or self.pca is None:
            self.extract_emotional_subspace()

        # Get activation pattern for input text
        activation = self.get_activation_patterns(text)

        # Project into original dimension space
        text_emotions = activation.cpu().numpy().reshape(1, -1)

        self.logger.info(f"Analyzing text shape: {text_emotions.shape}")

        # Use the same PCA transformation as during extraction
        projected_emotions = self.pca.transform(text_emotions)[0]
        self.logger.info(f"Projected emotions shape: {projected_emotions.shape}")

        # Calculate similarity with each stored emotion vector
        emotion_scores = {}
        for emotion, vector in self.emotion_vectors.items():
            # Ensure vectors are normalized before dot product
            proj_norm = np.linalg.norm(projected_emotions)
            vec_norm = np.linalg.norm(vector)

            if proj_norm > 0 and vec_norm > 0:
                similarity = np.dot(projected_emotions, vector) / (proj_norm * vec_norm)
                # Scale to [0, 1] range
                similarity = (similarity + 1) / 2
            else:
                # Default to neutral if norms are zero
                similarity = 0.5

            emotion_scores[emotion] = float(similarity)

        return emotion_scores

    def generate_emotional_response(
        self,
        prompt: str,
        target_emotions: Dict[str, float],
        temperature: float = 0.7
    ) -> str:
        """
        Generate a response that attempts to embody the specified emotional
        qualities in `target_emotions`.
        """
        # Create emotional context
        emotion_prompt = "Respond with "
        emotion_descriptors = []
        for emotion, intensity in target_emotions.items():
            if intensity > 0.1:  # threshold for relevance
                emotion_descriptors.append(f"{emotion} ({intensity:.2f})")

        if emotion_descriptors:
            emotion_prompt += ", ".join(emotion_descriptors)
        else:
            emotion_prompt += "neutral"

        # Combine prompts
        full_prompt = f"{emotion_prompt}\nPrompt: {prompt}\nResponse:"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=150,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Verify emotional content
        response_emotions = self.analyze_text_emotions(response)
        self.logger.info("Response emotion analysis:")
        for emotion, score in response_emotions.items():
            self.logger.info(f"{emotion}: {score:.2f}")

        return response

    def create_emotion_lora_config(
        self,
        rank: int = 8,
        alpha: float = 32.0
    ) -> Dict:
        """
        Create a LoRA configuration that can be used for emotional fine-tuning.
        The idea is to use the principal emotional subspace to guide
        weight updates in the LoRA modules.
        """
        if self.emotional_subspace is None:
            self.extract_emotional_subspace()

        config = {
            "rank": rank,
            "alpha": alpha,
            "emotional_subspace": self.emotional_subspace,
            "emotion_vectors": self.emotion_vectors,
            "target_modules": [
                "q_proj",
                "v_proj",
                "k_proj",
                "out_proj"
            ]
        }
        return config


class EmotionalLoRALayer(nn.Module):
    """LoRA layer for emotional control"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        emotion_vectors: Dict[str, np.ndarray]
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.emotion_vectors = emotion_vectors

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.randn(rank, out_features) * 0.02)

        # Emotion mixing weights
        self.emotion_weights = nn.Parameter(
            torch.randn(len(emotion_vectors), rank) * 0.02
        )

    def forward(
        self,
        x: torch.Tensor,
        emotion_intensities: Dict[str, float]
    ) -> torch.Tensor:
        """Forward pass with emotional modulation."""
        # Create emotion intensity tensor
        emotions = list(self.emotion_vectors.keys())
        intensity_vector = torch.tensor(
            [emotion_intensities.get(e, 0.0) for e in emotions],
            dtype=x.dtype, device=x.device
        )

        # Mix emotions
        emotion_mix = torch.matmul(intensity_vector, self.emotion_weights)

        # Apply LoRA with emotional modulation
        lora_mid = torch.matmul(x, self.lora_A)
        lora_mid = lora_mid * emotion_mix  # elementwise scaling
        lora_output = torch.matmul(lora_mid, self.lora_B)

        return (self.alpha / self.rank) * lora_output


def visualize_emotional_space(subspace, vectors, emotion_labels):
    """Visualize the emotional subspace using PCA components."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Create 3D plot of first three components
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each emotion cluster
    for emotion in set(emotion_labels):
        indices = [i for i, e in enumerate(emotion_labels) if e == emotion]
        points = subspace[indices]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            label=emotion,
            alpha=0.6
        )

    # Plot emotion vectors
    for emotion, vector in vectors.items():
        ax.quiver(
            0, 0, 0,
            vector[0], vector[1], vector[2],
            color='black',
            alpha=0.5
        )

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()
    plt.title('Emotional Subspace (First 3 Principal Components)')
    plt.show()


def main():
    """Test the emotional dimension extraction system."""
    extractor = EmotionalDimensionExtractor()
    logging.info("Initialized EmotionalDimensionExtractor")

    # Extract emotional dimensions
    subspace, vectors = extractor.extract_emotional_subspace()
    logging.info(f"\nExtracted subspace shape: {subspace.shape}")
    logging.info(f"Number of emotion vectors: {len(vectors)}")

    # Visualize the emotional space
    emotion_labels = []
    for emotion, probes in extractor.generate_emotional_probes().items():
        emotion_labels.extend([emotion] * len(probes))

    try:
        visualize_emotional_space(subspace, vectors, emotion_labels)
        logging.info("Successfully visualized emotional space")
    except Exception as e:
        logging.error(f"Failed to visualize emotional space: {e}")

    # Test some example texts
    test_texts = [
        "I just won the lottery!",
        "My pet passed away yesterday.",
        "The weather is quite pleasant today.",
        "That driver just cut me off!",
    ]

    logging.info("\nAnalyzing test texts:")
    for text in test_texts:
        logging.info(f"\nAnalyzing: {text}")
        try:
            emotions = extractor.analyze_text_emotions(text)
            logging.info("Emotion analysis successful")
            logging.info("Results:")
            for emotion, score in emotions.items():
                logging.info(f"{emotion}: {score:.3f}")
        except Exception as e:
            logging.error(f"Error analyzing text: {e}")
            raise

    for text in test_texts:
        print(f"\nAnalyzing: {text}")
        emotions = extractor.analyze_text_emotions(text)
        for emotion, score in emotions.items():
            print(f"{emotion}: {score:.2f}")

        # Generate emotional response
        response = extractor.generate_emotional_response(
            text,
            target_emotions=emotions
        )
        print(f"\nGenerated response: {response}")


if __name__ == "__main__":
    main()
