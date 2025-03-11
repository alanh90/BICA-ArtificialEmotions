# app.py
from flask import Flask, render_template, jsonify, request
import time
import threading
import random
import json
import os
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError
import numpy as np
from datetime import datetime

# Import for semantic memory
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers or scikit-learn not installed.")
    print("Install with: pip install sentence-transformers scikit-learn")
    print("Running without semantic memory search.")
    EMBEDDINGS_AVAILABLE = False

app = Flask(__name__)

# Initialize OpenAI client, could use any good AI honestly
load_dotenv()

# get api key from environment
api_key = os.getenv("OPENAI_API_KEY")


# create OpenAI client
def create_client(api_key):
    try:
        client = OpenAI(api_key=api_key)
        # Quick validation of client
        client.models.list()
        return client
    except AuthenticationError:
        print("Incorrect API")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


client = create_client(api_key)


# ===========================================================================================
# MEMORY SYSTEM
# ===========================================================================================

class MemorySystem:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        """Initialize the memory system with an embedding model for semantic search"""
        self.memories = []
        self.emotional_memories = []

        # Try to load embedding model if dependencies are available
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model_name)
                print(f"Loaded embedding model: {embedding_model_name}")
                self.embeddings_enabled = True
            except Exception as e:
                print(f"Could not load embedding model: {e}")
                print("Running without semantic memory search")
                self.embedding_model = None
                self.embeddings_enabled = False
        else:
            self.embedding_model = None
            self.embeddings_enabled = False

    def add_memory(self, text, source="conversation", metadata=None):
        """Add a regular memory without emotional association"""
        # Create embedding for semantic search
        embedding = None
        if self.embeddings_enabled:
            try:
                embedding = self.embedding_model.encode(text)
            except Exception as e:
                print(f"Error creating embedding: {e}")

        # Create memory entry
        memory = {
            "text": text,
            "source": source,
            "embedding": embedding,
            "timestamp": time.time(),
            "metadata": metadata or {},
            "recall_count": 0
        }

        # Add to memories
        self.memories.append(memory)

        # Keep memory size manageable
        if len(self.memories) > 100:
            # Sort by recency and recall count (keep frequently accessed)
            self.memories.sort(key=lambda x: x["timestamp"] + (x["recall_count"] * 86400))
            # Keep most recent/important
            self.memories = self.memories[-100:]

        return memory

    def add_emotional_memory(self, text, emotions, importance=1.0, metadata=None):
        """Add a memory with emotional associations"""
        # Only store if there are significant emotions
        significant = False
        for emotion, value in emotions.items():
            if abs(value - 0.5) > 0.2:  # Only store if emotion is significantly non-neutral
                significant = True
                break

        if not significant:
            return self.add_memory(text, "conversation", metadata)

        # Create embedding for semantic search
        embedding = None
        if self.embeddings_enabled:
            try:
                embedding = self.embedding_model.encode(text)
            except Exception as e:
                print(f"Error creating embedding: {e}")

        # Create emotional memory
        memory = {
            "text": text,
            "emotions": emotions.copy(),  # Store emotion state at time of memory
            "embedding": embedding,
            "importance": importance,
            "timestamp": time.time(),
            "metadata": metadata or {},
            "recall_count": 0
        }

        # Add to emotional memories
        self.emotional_memories.append(memory)

        # Keep memory size manageable
        if len(self.emotional_memories) > 50:
            # Sort by importance and recency
            self.emotional_memories.sort(key=lambda x: x["importance"] + (x["recall_count"] * 0.5))
            # Keep most important
            self.emotional_memories = self.emotional_memories[-50:]

        return memory

    def find_related_memories(self, text, threshold=0.6, max_results=3):
        """Find regular memories semantically related to the given text"""
        if not self.embeddings_enabled or not self.memories:
            return []

        # Create query embedding
        try:
            query_embedding = self.embedding_model.encode(text)

            # Compare with stored memories
            results = []
            for memory in self.memories:
                if memory["embedding"] is not None:
                    similarity = cosine_similarity(
                        [query_embedding],
                        [memory["embedding"]]
                    )[0][0]

                    if similarity >= threshold:
                        results.append((memory, similarity))

            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)

            # Update recall count for retrieved memories
            for memory, _ in results[:max_results]:
                memory["recall_count"] += 1

            return results[:max_results]

        except Exception as e:
            print(f"Error finding related memories: {e}")
            return []

    def find_emotional_memories(self, text, threshold=0.65, max_results=2):
        """Find emotional memories related to the given text"""
        if not self.embeddings_enabled or not self.emotional_memories:
            return []

        # Create query embedding
        try:
            query_embedding = self.embedding_model.encode(text)

            # Compare with stored emotional memories
            results = []
            for memory in self.emotional_memories:
                if memory["embedding"] is not None:
                    similarity = cosine_similarity(
                        [query_embedding],
                        [memory["embedding"]]
                    )[0][0]

                    if similarity >= threshold:
                        results.append((memory, similarity))

            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)

            # Update recall count for retrieved memories
            for memory, _ in results[:max_results]:
                memory["recall_count"] += 1

            return results[:max_results]

        except Exception as e:
            print(f"Error finding emotional memories: {e}")
            return []

    def get_emotional_influence(self, text):
        """Calculate emotional influence from similar emotional memories"""
        similar_memories = self.find_emotional_memories(text)

        if not similar_memories:
            return {}

        # Combine emotional influences from memories
        influences = {emotion: 0.0 for emotion in ["joy", "sadness", "anger", "fear",
                                                   "surprise", "trust", "disgust", "anticipation"]}

        for memory, similarity in similar_memories:
            # Scale influence by similarity, importance, and recency
            influence_strength = similarity * memory["importance"]

            # Time decay factor - memories weaken over time (half-strength after 7 days)
            days_old = (time.time() - memory["timestamp"]) / (3600 * 24)
            time_factor = 0.5 ** (days_old / 7)

            # More often recalled memories have less impact (diminishing returns)
            recall_factor = 0.8 ** memory["recall_count"]

            # Apply scaled influence from this memory's emotions
            for emotion, value in memory["emotions"].items():
                if emotion in influences:
                    # Calculate deviation from neutral (0.5)
                    deviation = (value - 0.5)

                    # Apply all factors to determine influence
                    influences[emotion] += deviation * influence_strength * time_factor * recall_factor

                    # Log significant emotional memory triggering
                    if abs(deviation * influence_strength) > 0.1:
                        print(f"Emotional memory triggered: {emotion} {deviation:.2f} * {influence_strength:.2f} = {deviation * influence_strength:.2f}")
                        print(f"Memory: {memory['text'][:100]}...")

        return influences


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

        # Initialize memory system
        self.memory = MemorySystem()

        # System parameters - UPDATED FOR MORE FLUIDITY
        self.decay_rate = 0.95  # Faster decay (was 0.98)
        self.noise_magnitude = 0.015  # More noise (was 0.005)
        self.thought_frequency = 7  # More frequent thoughts (was 10)
        self.update_frequency = 0.2  # Update emotion history more often

        # Start background processing
        self.running = True
        self.last_thought_time = time.time()
        self.last_update_time = time.time()
        self.last_micro_time = time.time()
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

            # Apply micro-fluctuations for more natural movement
            if current_time - self.last_micro_time > 0.5:  # Every half second
                self._apply_micro_fluctuations()
                self.last_micro_time = current_time

            # Generate periodic thoughts
            elapsed = current_time - self.last_thought_time
            thought_due = elapsed > random.uniform(
                self.thought_frequency * 0.5,
                self.thought_frequency * 1.5
            )

            if thought_due:
                self._generate_thought(current_time)
                self.last_thought_time = current_time

            # Record emotion history more frequently
            if current_time - self.last_update_time > self.update_frequency:
                self.emotion_history.append(self.emotions.copy())
                if len(self.emotion_history) > 300:  # 5 minutes worth
                    self.emotion_history = self.emotion_history[-300:]
                self.last_update_time = current_time

            # Sleep briefly - shorter sleep for more responsive updates
            time.sleep(0.05)

    def _apply_decay(self, elapsed_time):
        """Decay emotions gradually toward 0"""
        # Calculate decay factor based on elapsed time
        decay_factor = self.decay_rate ** elapsed_time

        for emotion in self.emotions:
            # More intense emotions decay more slowly
            intensity_factor = 0.2 + (self.emotions[emotion] * 0.8)
            adjusted_decay = decay_factor ** intensity_factor

            # Apply decay with small random variation for more natural movement
            variation = 1.0 + random.uniform(-0.05, 0.05)  # ±5% variation
            self.emotions[emotion] *= adjusted_decay * variation

    def _apply_noise(self):
        """Apply small random changes to emotions"""
        for emotion in self.emotions:
            # Dynamic noise scale - less predictable
            intensity = self.emotions[emotion]
            base_noise = self.noise_magnitude * (1.0 - intensity * 0.7)  # Less reduction at high intensity

            # Random fluctuation in noise amount
            noise_scale = base_noise * random.uniform(0.5, 1.5)

            # Apply noise
            noise = random.uniform(-noise_scale, noise_scale)
            self.emotions[emotion] = max(0.0, min(1.0, self.emotions[emotion] + noise))

    def _apply_micro_fluctuations(self):
        """Apply tiny fluctuations to create more natural emotion movement"""
        # Pick 2-3 random emotions to adjust
        emotions_to_adjust = random.sample(list(self.emotions.keys()), random.randint(2, 3))

        for emotion in emotions_to_adjust:
            # Very small adjustments
            micro_change = random.uniform(-0.03, 0.03)

            # Apply with preference toward the middle range (more movement in neutral states)
            distance_from_mid = abs(self.emotions[emotion] - 0.5)
            if distance_from_mid > 0.3:  # Emotions far from neutral move less
                micro_change *= (1.0 - distance_from_mid)

            # Apply the micro-fluctuation
            self.emotions[emotion] = max(0.0, min(1.0, self.emotions[emotion] + micro_change))

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
            # Check if we have related memories to include
            memory_context = ""
            if self.conversation:
                latest_conversation = self.conversation[-1]
                related_memories = self.memory.find_related_memories(latest_conversation)
                if related_memories:
                    memory_texts = [mem[0]['text'][:100] + "..." for mem in related_memories[:1]]
                    memory_context = "Related memory: " + memory_texts[0]

            # Call OpenAI API to generate a thought
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You generate brief, natural thoughts for an AI assistant based on its current emotional state and conversation context. Generate only the thought itself, no explanations or additional text."},
                    {"role": "user", "content": f"""
Current time: {time_str}
Emotional state: {emotion_text}
{memory_context}

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
            # Check for emotional memories that might be triggered
            memory_influences = self.memory.get_emotional_influence(message)

            # Analyze message for direct emotional impact
            direct_impacts = self._analyze_message_impact(message)

            # Apply combined emotional impacts (memory has 30% weight)
            combined_impacts = {}
            for emotion in self.emotions:
                direct = direct_impacts.get(emotion, 0) if isinstance(direct_impacts, dict) else 0
                memory = memory_influences.get(emotion, 0)
                combined_impacts[emotion] = (direct * 0.7) + (memory * 0.3)

                # Apply significant emotional changes
                if abs(combined_impacts[emotion]) > 0.01:
                    self.emotions[emotion] = max(0.0, min(1.0, self.emotions[emotion] + combined_impacts[emotion]))

            # Generate response using OpenAI
            response = self._generate_response(message)

            # Add response to conversation
            self.conversation.append(f"AI: {response}")

            # Store this interaction in memory
            conversation_text = f"User: {message}\nAI: {response}"
            self.memory.add_emotional_memory(conversation_text, self.emotions.copy())

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
                return impact

            except json.JSONDecodeError:
                print(f"Error parsing emotional impact: {analysis_text}")
                return {}

        except Exception as e:
            print(f"Error analyzing message impact: {e}")
            return {}

    def _generate_response(self, message):
        """Generate a response using OpenAI that takes into account emotional state"""
        # Format emotions for the prompt
        emotion_text = ", ".join([f"{emotion}: {value:.2f}" for emotion, value in self.emotions.items()])

        # Get recent thoughts
        recent_thoughts = [t["text"] for t in self.thoughts[-3:]] if self.thoughts else []
        thoughts_text = "\n".join([f"- {t}" for t in recent_thoughts])

        # Find related memories to include in context
        related_memories = self.memory.find_related_memories(message)
        memory_context = ""
        if related_memories:
            memory_texts = [f"- {mem[0]['text'][:100]}..." for mem in related_memories]
            memory_context = "Related memories:\n" + "\n".join(memory_texts)

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

{memory_context}

Your response should be influenced by your current emotional state and memories, but you should not explicitly mention your emotions unless directly asked about them. The emotional influence should be subtle and natural, affecting your tone, word choice, and perspective.
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


@app.route('/api/memories')
def get_memories():
    """Get recent memories for display"""
    regular_memories = emotion_system.memory.memories[-10:] if emotion_system.memory.memories else []
    emotional_memories = emotion_system.memory.emotional_memories[-10:] if emotion_system.memory.emotional_memories else []

    # Format for display
    formatted_regular = [{
        "text": mem["text"][:100] + "..." if len(mem["text"]) > 100 else mem["text"],
        "timestamp": mem["timestamp"],
        "recall_count": mem["recall_count"]
    } for mem in regular_memories]

    formatted_emotional = [{
        "text": mem["text"][:100] + "..." if len(mem["text"]) > 100 else mem["text"],
        "emotions": mem["emotions"],
        "importance": mem["importance"],
        "timestamp": mem["timestamp"]
    } for mem in emotional_memories]

    return jsonify({
        "regular": formatted_regular,
        "emotional": formatted_emotional
    })


# ===========================================================================================
# MAIN EXECUTION
# ===========================================================================================

if __name__ == '__main__':
    try:
        app.run(debug=True, use_reloader=False)
    finally:
        emotion_system.stop()