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
# Function: Load or Download a Dictionary
# -------------------------------
def load_or_download_dictionary():
    """
    Loads a dictionary from a local file or downloads one if missing.
    Returns a list of English words.
    """
    if not os.path.exists(DICTIONARY_FILE):
        print("🔹 Dictionary not found. Downloading...")
        try:
            url = "https://raw.githubusercontent.com/dwyl/english-words/master/words.txt"
            urllib.request.urlretrieve(url, DICTIONARY_FILE)
            print(f"✅ Dictionary downloaded and saved to {DICTIONARY_FILE}")
        except Exception as e:
            print(f"❌ Error downloading dictionary: {e}")
            return ["emotion", "happiness", "fear", "trust", "storytelling"]  # Fallback words

    with open(DICTIONARY_FILE, "r") as f:
        words = [word.strip() for word in f.readlines() if word.strip().isalpha()]
    return words if words else ["emotion", "happiness", "fear", "trust", "storytelling"]

# -------------------------------
# Function: Get a Random Word for Topic Influence
# -------------------------------
def generate_conversation_topic(word_list):
    """Selects a random word from the dictionary to influence the conversation topic."""
    return random.choice(word_list)

# -------------------------------
# Function: Build Generation Prompt
# -------------------------------
def build_generation_prompt(word_list):
    """
    Builds a prompt that instructs the AI to generate a synthetic training sample.
    The random word (topic) is used as an influence for the conversation.
    The prompt instructs the AI to output valid JSON strictly, with the following fields:
      - prompt: The conversation prompt (user input or conversation history).
      - added_context: Background from the AI's perspective (thoughts, memories, or additional cues).
      - response: The AI's reply only.
      - emotion_vector: A dictionary with emotion keys and values between 0 and 1.
    """
    topic = generate_conversation_topic(word_list)
    prompt = (
        f"Generate a synthetic training sample for an emotion-aware AI conversation. "
        f"Use the following topic as inspiration: **'{topic}'**. "
        "The output must be strictly in JSON format with no extra text. "
        "Ensure the JSON object includes the following fields:\n\n"
        "{\n"
        '  "prompt": "User: [User input or conversation history]\\nAI:",\n'
        '  "added_context": "[Context from AI\'s perspective]",\n'
        '  "response": "[AI response only]",\n'
        '  "emotion_vector": { "joy": [0-1], "trust": [0-1], "fear": [0-1], "surprise": [0-1], "sadness": [0-1], "disgust": [0-1], "anger": [0-1], "anticipation": [0-1] }\n'
        "}\n\n"
        f"Incorporate the topic '{topic}' naturally into the conversation."
    )
    return prompt

# -------------------------------
# Function: Extract Strict JSON from API Response
# -------------------------------
def extract_json(response_text):
    """Extracts the first valid JSON object from a string."""
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start == -1 or json_end == -1:
            print("❌ No valid JSON detected in API response.")
            return None
        json_data = response_text[json_start:json_end]
        return json.loads(json_data)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return None

# -------------------------------
# Function: Generate a Sample Using Groq API
# -------------------------------
def generate_sample(word_list):
    prompt = build_generation_prompt(word_list)
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Adjust as needed
            messages=[
                {"role": "system", "content": "You are an AI that generates high-quality synthetic training data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=400,
        )
        generated_text = response.choices[0].message.content.strip()
        print(f"🔹 Raw API Response:\n{generated_text}\n")
        if not generated_text:
            print("❌ Empty response received from Groq API.")
            return None
        sample = extract_json(generated_text)
        return sample
    except Exception as e:
        print(f"❌ Error generating sample: {e}")
        return None

# -------------------------------
# Function: Generate Multiple Samples
# -------------------------------
def generate_synthetic_data(num_samples, word_list):
    samples = []
    for i in range(num_samples):
        print(f"🔹 [{i + 1}/{num_samples}] Generating sample...")
        sample = generate_sample(word_list)
        if sample is not None:
            samples.append(sample)
        time.sleep(1)  # Avoid rate limits
    return samples

# -------------------------------
# Function: Save Samples to JSONL
# -------------------------------
def save_to_jsonl(samples, output_file):
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

# -------------------------------
# Main Execution
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
