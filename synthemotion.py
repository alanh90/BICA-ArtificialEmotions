#!/usr/bin/env python
import os
import json
import random
import argparse
import time
import urllib.request
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
DICTIONARY_FILE = "english_words.txt"  # Local dictionary file


# -------------------------------
# 🔹 Function: Load or Download a Dictionary
# -------------------------------
def load_or_download_dictionary():
    """
    Loads a dictionary from a local file or downloads one if missing.
    Returns a list of English words.
    """
    if not os.path.exists(DICTIONARY_FILE):
        print("🔹 Dictionary not found. Downloading...")
        try:
            # Download a word list from an online source
            url = "https://raw.githubusercontent.com/dwyl/english-words/master/words.txt"
            urllib.request.urlretrieve(url, DICTIONARY_FILE)
            print(f"✅ Dictionary downloaded and saved to {DICTIONARY_FILE}")
        except Exception as e:
            print(f"❌ Error downloading dictionary: {e}")
            return ["emotion", "happiness", "fear", "trust", "storytelling"]  # Fallback words

    # Load words from the dictionary file
    with open(DICTIONARY_FILE, "r") as f:
        words = [word.strip() for word in f.readlines() if word.strip().isalpha()]

    if not words:
        print("❌ Dictionary file is empty or corrupted. Using fallback words.")
        return ["emotion", "happiness", "fear", "trust", "storytelling"]

    return words


# -------------------------------
# 🔹 Function: Get a Random Word
# -------------------------------
def get_random_word(word_list):
    """Selects a random word from the dictionary."""
    return random.choice(word_list)


# -------------------------------
# 🔹 Function: Generate Conversation Prompt
# -------------------------------
def generate_conversation(word_list):
    """
    Randomly generates:
    - Either a single user input
    - Or a short back-and-forth conversation.
    """
    random_word = get_random_word(word_list)

    if random.random() < 0.5:  # 50% chance for short conversation history
        return (
            f"User: I've been thinking a lot about {random_word}. What do you think?\n"
            f"AI: {random_word} is quite fascinating! There's a lot to explore about it.\n"
            f"User: Yeah! I've always wanted to learn more.\n"
            f"AI:"
        )
    else:
        return f"User: {random_word} makes me feel something deep inside.\nAI:"


# -------------------------------
# 🔹 Function: Build Generation Prompt
# -------------------------------
def build_generation_prompt(word_list):
    """
    Creates a structured prompt ensuring:
    - 'prompt' includes either a single user input or conversation history.
    - 'response' contains **only** the AI's reply.
    - 'added_context' provides background from the AI's perspective before responding.
    - 'emotion_vector' represents the AI's emotions with values between 0 and 1.
    """
    conversation_prompt = generate_conversation(word_list)

    prompt = (
        f"Generate a synthetic training sample for an **emotion-aware AI**. "
        f"The conversation should feel **natural** and reflect a topic.\n"
        "- 'prompt' includes the user input (and optionally conversation history). Sometimes its just the user, other times it will be a short snippet of conversation history.\n"
        "- 'response' contains **only** the AI's reply (no user input included).\n"
        "- 'added_context' provides background from the AI's perspective before responding. Could be past related memories, could be thoughts, could be something random that popped in its head before responding.\n"
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


# -------------------------------
# 🔹 Function: Generate a Sample Using Groq API
# -------------------------------
def generate_sample(word_list):
    prompt = build_generation_prompt(word_list)
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
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


# -------------------------------
# 🔹 Function: Generate Multiple Samples
# -------------------------------
def generate_synthetic_data(num_samples, word_list):
    samples = []
    for i in range(num_samples):
        print(f"🔹 [{i + 1}/{num_samples}] Generating sample...")
        sample = generate_sample(word_list)
        if sample is not None:
            samples.append(sample)
        # Sleep to avoid rate limits
        time.sleep(1)
    return samples


# -------------------------------
# 🔹 Function: Save Samples to JSONL
# -------------------------------
def save_to_jsonl(samples, output_file):
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


# -------------------------------
# 🔹 Main Execution
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training data for emotion-aware AI.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate.")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output JSONL file.")
    args = parser.parse_args()

    word_list = load_or_download_dictionary()
    synthetic_samples = generate_synthetic_data(args.num_samples, word_list)
    save_to_jsonl(synthetic_samples, args.output)
    print(f"✅ Generated {len(synthetic_samples)} samples and saved to {args.output}")
