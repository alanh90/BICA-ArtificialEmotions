#!/usr/bin/env python
import os
import json
import random
import argparse
import time
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Get API key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Error: GROQ_API_KEY not found in environment variables or .env file.")

client = Groq(api_key=GROQ_API_KEY)

# File where synthetic data will be saved
OUTPUT_FILE = "synthetic_emotion_data.jsonl"

# List of possible random topics
TOPIC_LIST = [
    "sunrise", "car", "toys", "hobby", "helping", "greeting", "saying goodbye",
    "saying something emotional", "assist", "talk", "storytelling", "questioning",
    "fighting", "interested", "ocean", "forest", "city", "mountain", "river",
    "rainbow", "desert", "flower", "star", "empathy"
]

# Function to choose a random word
def get_random_word():
    return random.choice(TOPIC_LIST)

# Function to randomly decide whether to generate a single message or a short back-and-forth
def generate_conversation():
    """
    Randomly decide between a single user input or a short back-and-forth conversation.
    """
    if random.random() < 0.5:  # 50% chance for a short conversation history
        return f"User: {random.choice(TOPIC_LIST)} is amazing! What do you think?\nAI: That's an interesting perspective! I believe {random.choice(TOPIC_LIST)} has a fascinating history.\nUser: Yeah! I've always wanted to learn more.\nAI:"
    else:
        return f"User: {random.choice(TOPIC_LIST)} makes me feel emotional.\nAI:"

# Function to build a structured prompt for Groq API
def build_generation_prompt():
    """
    Creates a structured prompt ensuring:
    - 'prompt' includes the user input and optionally conversation history.
    - 'response' contains only the AI's reply.
    - 'added_context' provides background from the AI's perspective before responding.
    - 'emotion_vector' represents the AI's emotional state.
    """
    conversation_prompt = generate_conversation()

    prompt = (
        f"Generate a synthetic training sample for an **emotion-aware AI**. "
        f"The conversation should feel **natural** and reflect a topic.\n"
        "- 'prompt' includes the user input (and optionally conversation history).\n"
        "- 'response' contains **only** the AI's reply (no user input included).\n"
        "- 'added_context' provides background from the AI's perspective before responding. "
        "This can be thoughts, memories, or related knowledge.\n"
        "- 'emotion_vector' represents the AI's emotions with **values between 0 and 1**.\n\n"
        "**Example format:**\n"
        "```json\n"
        "{\n"
        '  "prompt": "User: [User input or conversation history]\\nAI:",\n'
        '  "added_context": "[Context from AI perspective]",\n'
        '  "response": "[AI response only]",\n'
        '  "emotion_vector": { "joy": [0-1], "trust": [0-1], "fear": [0-1], "surprise": [0-1], "sadness": [0-1], "disgust": [0-1], "anger": [0-1], "anticipation": [0-1] }\n'
        "}\n"
        "```\n"
        "Generate the response **strictly in JSON format** with no extra text.\n\n"
        f"Here is the conversation setup:\n\n{conversation_prompt}"
    )
    return prompt

# Function to query Groq API and generate synthetic data
def generate_sample():
    prompt = build_generation_prompt()
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Adjust if needed
            messages=[
                {"role": "system", "content": "You are an AI that generates high-quality synthetic training data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=400,
        )
        generated_text = response.choices[0].message.content.strip()

        # Try to parse the JSON object.
        sample = json.loads(generated_text)
        return sample
    except Exception as e:
        print(f"❌ Error generating sample: {e}")
        return None

# Main function to generate synthetic data
def generate_synthetic_data(num_samples):
    samples = []
    for i in range(num_samples):
        print(f"🔹 [{i+1}/{num_samples}] Generating sample...")
        sample = generate_sample()
        if sample is not None:
            samples.append(sample)
        # Sleep to avoid rate limits
        time.sleep(1)
    return samples

# Save the samples to a JSONL file
def save_to_jsonl(samples, output_file):
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training data for emotion-aware AI.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate.")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output JSONL file.")
    args = parser.parse_args()

    synthetic_samples = generate_synthetic_data(args.num_samples)
    save_to_jsonl(synthetic_samples, args.output)
    print(f"✅ Generated {len(synthetic_samples)} samples and saved to {args.output}")
