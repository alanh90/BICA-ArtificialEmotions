# Artificial Emotion Framework (AEF)

*A potential integral component of the Bicameral AGI Project: Designing Affect for Intelligent Systems*

![GitHub commit activity](https://img.shields.io/github/commit-activity/m/alanh90/BICA-ArtificialEmotions)
![GitHub last commit](https://img.shields.io/github/last-commit/alanh90/BICA-ArtificialEmotions)
![GitHub License](https://img.shields.io/github/license/alanh90/BICA-ArtificialEmotions)

<div align="center"><img src="media/artificial_emotions.png" alt="Hungry Matrix Cover"></div>

This repository implements an artificial emotion and semantic memory system integrated into a Flask web application, utilizing OpenAI's API and sentence embeddings for semantic memory retrieval.

## Overview

The system consists of two main components:

### 1. Emotional System

Simulates an evolving emotional state with dynamic intensity influenced by user interaction, memories, and internal background thoughts. Emotion dynamics include:

- **Emotion dimensions:** joy, sadness, anger, fear, surprise, trust, disgust, anticipation
- **Time-based Decay:** Gradually reduces emotion intensity over time to simulate natural emotional transitions.
- **Micro-fluctuations and Noise:** Adds subtle, randomized variations for realistic emotional states.
- **Thought Generation:** Background generation of contextually influenced thoughts using the OpenAI API, which subtly affect emotional states.
- **Emotion-driven Responses:** User interactions affect and are influenced by the current emotional state, producing naturally varying responses.

### 2. Memory System

Stores and retrieves memories based on semantic similarity, with special handling for emotionally significant memories.

- **Semantic Memory:** Uses embeddings (via sentence-transformers) to store and retrieve semantically related memories.
- **Emotional Memories:** Associates memories with significant emotional responses.
- **Dynamic Memory Management:** Memories are maintained based on recency, relevance, and recall frequency, limiting memory size for efficient retrieval.

## Installation & Setup

Clone the repository:
```bash
git clone https://github.com/alanh90/BICA-ArtificialEmotions.git
cd BICA-ArtificialEmotions
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Configure environment variables by creating a `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Run the Flask server:
```bash
python app.py
```

## Future Directions & Potential Improvements

- **Emotion Vector Representation:** Implement emotions as vectors in latent emotional spaces for increased flexibility and realism.
- **Emotional Conflict Resolution:** Introduce methods for resolving conflicting emotions, such as weighted blending or context-based prioritization.
- **Emotion-Influenced LLM Attention:** Incorporate emotional states directly into attention mechanisms or prompts to more effectively guide AI responses.
- **Optimizations:** Cache embeddings for faster semantic searches, reduce computational overhead in background emotional updates.
- **Mutualistic Human-AI Interaction:** Further develop methods for adaptive emotional responses specifically tailored to improve human-AI interactions and empathy.

## Future Directions & Improvements

- Develop adaptive mechanisms allowing the AI to learn and modify its emotional responses based on experiences.
- Expand ethical considerations, ensuring transparency and interpretability of artificial emotional states.
- Implement dynamic emotion vector spaces for greater flexibility and realism.

## License

Distributed under the MIT License. See `LICENSE` for more details.

---

This implementation serves as a foundational structure, aligning closely with the theoretical "Latent Emotion Spaces" concept. Future iterations will deepen these capabilities to create more nuanced, adaptive, and realistic emotional interactions.

