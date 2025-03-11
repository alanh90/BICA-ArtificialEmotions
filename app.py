# app.py
from flask import Flask, render_template, jsonify, request
import time
import threading
import random
import json
import os
import numpy as np
from datetime import datetime
from openai import OpenAI

app = Flask(__name__)

# Initialize OpenAI client

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)






# ===========================================================================================
# EMOTIONAL SYSTEM
# ===========================================================================================

class EmotionalSystem:
    def __init__(self):
        # Emotion dimensions with intensity 0-1 (0 = none, 1 = maximum)
        self.emotions = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "trust": 0.0,
            "disgust": 0.0,
            "anticipation": 0.0
        }

        # Track thoughts, conversation, and emotion history
        self.thoughts = []
        self.conversation = []
        self.emotion_history = [self.emotions.copy()]

        # System parameters
        self.decay_rate = 0.98  # Natural emotional decay
        self.noise_magnitude = 0.005  # Background emotional noise
        self.thought_frequency = 10  # Seconds between thoughts (average)

        # Start background processing
        self.running = True
        self.last_thought_time = time.time()
        self.last_update_time = time.time()
        self.background_thread = threading.Thread(target=self._background_process)
        self.background_thread.daemon = True
        self.background_thread.start()

    def _background_process(self):
        """Run continuous background processing of emotions and thoughts"""
        while self.running:
            current_time = time.time()

            # Apply natural decay
            self._apply_decay(current_time - self.last_update_time)

            # Apply random noise to emotions
            self._apply_noise()

            # Generate periodic thoughts
            elapsed = current_time - self.last_thought_time
            thought_due = elapsed > random.uniform(
                self.thought_frequency * 0.5,
                self.thought_frequency * 1.5
            )

            if thought_due:
                self._generate_thought(current_time)
                self.last_thought_time = current_time

            # Record emotion history
            if current_time - self.last_update_time > 1.0:  # Every second
                self.emotion_history.append(self.emotions.copy())
                if len(self.emotion_history) > 300:  # 5 minutes worth
                    self.emotion_history = self.emotion_history[-300:]
                self.last_update_time = current_time

            # Sleep briefly
            time.sleep(0.1)

    def _apply_decay(self, elapsed_time):
        """Decay emotions gradually toward 0"""
        # Calculate decay factor based on elapsed time
        decay_factor = self.decay_rate ** elapsed_time

        for emotion in self.emotions:
            # More intense emotions decay more slowly
            intensity_factor = 0.2 + (self.emotions[emotion] * 0.8)  # 0.2-1.0
            adjusted_decay = decay_factor ** intensity_factor

            # Apply decay
            self.emotions[emotion] *= adjusted_decay

    def _apply_noise(self):
        """Apply small random changes to emotions"""
        for emotion in self.emotions:
            # Less noise for more intense emotions
            intensity = self.emotions[emotion]
            noise_scale = self.noise_magnitude * (1.0 - intensity * 0.8)

            # Apply noise
            noise = random.uniform(-noise_scale, noise_scale)
            self.emotions[emotion] = max(0.0, min(1.0, self.emotions[emotion] + noise))

    def _generate_thought(self, timestamp):
        """Generate a background thought using OpenAI API"""
        # Format current time
        time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')

        # Format emotions for the prompt
        emotion_text = ", ".join([f"{emotion}: {value:.2f}" for emotion, value in self.emotions.items()])

        # Get recent conversation for context
        recent_messages = self.conversation[-3:] if self.conversation else []
        context = "\n".join(recent_messages)

        try:
            # Call OpenAI API to generate a thought
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You generate brief, natural thoughts for an AI assistant based on its current emotional state and conversation context. Generate only the thought itself, no explanations or additional text."},
                    {"role": "user", "content": f"""
Current time: {time_str}
Emotional state: {emotion_text}

Recent conversation:
{context}

Generate a single brief, natural thought (1-2 sentences) that might occur to an AI assistant in this moment.
                    """}
                ],
                max_tokens=60,
                temperature=0.7
            )

            thought = response.choices[0].message.content.strip()

            # Record the thought
            thought_obj = {
                "text": thought,
                "time": timestamp,
                "emotions": self.emotions.copy()
            }

            self.thoughts.append(thought_obj)

            # Keep only recent thoughts
            if len(self.thoughts) > 10:
                self.thoughts = self.thoughts[-10:]

            # Update emotions based on thought (using OpenAI again)
            self._analyze_thought_impact(thought)

        except Exception as e:
            print(f"Error generating thought: {e}")

    def _analyze_thought_impact(self, thought):
        """Analyze emotional impact of a thought using OpenAI API"""
        try:
            # Call OpenAI to analyze emotional impact
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You analyze how a thought would impact emotions. You return a JSON object with emotion names as keys and values from -0.2 to 0.2 indicating how much each emotion should change."},
                    {"role": "user", "content": f"Analyze the emotional impact of this thought: '{thought}'. Return a JSON object with these emotions: joy, sadness, anger, fear, surprise, trust, disgust, anticipation. Values should be from -0.2 to 0.2."}
                ],
                max_tokens=150,
                temperature=0.3
            )

            # Extract JSON from response
            analysis_text = response.choices[0].message.content.strip()
            try:
                # Extract JSON if it's wrapped in backticks or has explanatory text
                if "```json" in analysis_text:
                    json_str = analysis_text.split("```json")[1].split("```")[0].strip()
                elif "```" in analysis_text:
                    json_str = analysis_text.split("```")[1].strip()
                else:
                    json_str = analysis_text

                impact = json.loads(json_str)

                # Update emotions based on the analysis
                for emotion, change in impact.items():
                    if emotion in self.emotions:
                        self.emotions[emotion] = max(0.0, min(1.0, self.emotions[emotion] + float(change)))

            except json.JSONDecodeError:
                print(f"Error parsing emotional impact: {analysis_text}")

        except Exception as e:
            print(f"Error analyzing thought impact: {e}")

    def process_message(self, message):
        """Process a user message, update emotions, and generate a response"""
        # Add to conversation history
        self.conversation.append(f"User: {message}")

        try:
            # Analyze emotional impact of message using OpenAI
            self._analyze_message_impact(message)

            # Generate response using OpenAI
            response = self._generate_response(message)

            # Add response to conversation
            self.conversation.append(f"AI: {response}")

            # Keep conversation history manageable
            if len(self.conversation) > 20:
                self.conversation = self.conversation[-20:]

            return response

        except Exception as e:
            print(f"Error processing message: {e}")
            return "I'm sorry, I encountered an error processing your message."

    def _analyze_message_impact(self, message):
        """Analyze emotional impact of user message using OpenAI API"""
        try:
            # Call OpenAI to analyze emotional impact
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You analyze how a message would impact emotions. You return a JSON object with emotion names as keys and values from -0.2 to 0.2 indicating how much each emotion should change."},
                    {"role": "user", "content": f"Analyze the emotional impact of this message: '{message}'. Return a JSON object with these emotions: joy, sadness, anger, fear, surprise, trust, disgust, anticipation. Values should be from -0.2 to 0.2."}
                ],
                max_tokens=150,
                temperature=0.3
            )

            # Extract JSON from response
            analysis_text = response.choices[0].message.content.strip()
            try:
                # Extract JSON if it's wrapped in backticks or has explanatory text
                if "```json" in analysis_text:
                    json_str = analysis_text.split("```json")[1].split("```")[0].strip()
                elif "```" in analysis_text:
                    json_str = analysis_text.split("```")[1].strip()
                else:
                    json_str = analysis_text

                impact = json.loads(json_str)

                # Update emotions based on the analysis
                for emotion, change in impact.items():
                    if emotion in self.emotions:
                        self.emotions[emotion] = max(0.0, min(1.0, self.emotions[emotion] + float(change)))

            except json.JSONDecodeError:
                print(f"Error parsing emotional impact: {analysis_text}")

        except Exception as e:
            print(f"Error analyzing message impact: {e}")

    def _generate_response(self, message):
        """Generate a response using OpenAI that takes into account emotional state"""
        # Format emotions for the prompt
        emotion_text = ", ".join([f"{emotion}: {value:.2f}" for emotion, value in self.emotions.items()])

        # Get recent thoughts
        recent_thoughts = [t["text"] for t in self.thoughts[-3:]] if self.thoughts else []
        thoughts_text = "\n".join([f"- {t}" for t in recent_thoughts])

        # Get recent conversation
        recent_messages = self.conversation[-6:] if self.conversation else []
        context = "\n".join(recent_messages)

        try:
            # Call OpenAI API to generate response
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"""You are an AI assistant with an emotional state that influences your responses.

Current emotional state:
{emotion_text}

Recent background thoughts:
{thoughts_text}

Your response should be influenced by your current emotional state, but you should not explicitly mention your emotions unless directly asked about them. The emotional influence should be subtle and natural, affecting your tone, word choice, and perspective.
                    """},
                    {"role": "user", "content": message}
                ],
                max_tokens=250,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error generating a response."

    def get_state(self):
        """Get the current state for display in UI"""
        return {
            "emotions": self.emotions,
            "thoughts": self.thoughts,
            "history": self.emotion_history
        }

    def stop(self):
        """Stop the background thread"""
        self.running = False
        if self.background_thread.is_alive():
            self.background_thread.join(timeout=1.0)


# Initialize the emotional system
emotion_system = EmotionalSystem()


# ===========================================================================================
# FLASK ROUTES
# ===========================================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/emotion/state')
def get_emotion_state():
    return jsonify(emotion_system.get_state())


@app.route('/api/chat/message', methods=['POST'])
def process_message():
    data = request.json
    message = data.get('message', '')

    if not message:
        return jsonify({
            "response": "I didn't receive any message.",
            "emotion_state": emotion_system.get_state()
        })

    # Process the message and get a response
    response = emotion_system.process_message(message)

    return jsonify({
        "response": response,
        "emotion_state": emotion_system.get_state()
    })


@app.route('/api/emotions/reset', methods=['POST'])
def reset_emotions():
    for emotion in emotion_system.emotions:
        emotion_system.emotions[emotion] = 0.0

    emotion_system.emotion_history.append(emotion_system.emotions.copy())

    return jsonify({
        "success": True,
        "emotion_state": emotion_system.get_state()
    })


# ===========================================================================================
# MAIN EXECUTION
# ===========================================================================================

if __name__ == '__main__':
    try:
        app.run(debug=True, use_reloader=False)
    finally:
        emotion_system.stop()