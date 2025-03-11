// static/js/app.js

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const emotionBarsContainer = document.getElementById('emotion-bars');
    const thoughtsContainer = document.getElementById('thoughts');
    const memoriesContainer = document.getElementById('memories');
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const resetButton = document.getElementById('reset-emotions');
    const emotionSpeedSelect = document.getElementById('emotion-speed');
    const emotionGraphCanvas = document.getElementById('emotion-graph');

    // Templates
    const thoughtTemplate = document.getElementById('thought-template');
    const memoryTemplate = document.getElementById('memory-template');
    const userMessageTemplate = document.getElementById('user-message-template');
    const assistantMessageTemplate = document.getElementById('assistant-message-template');

    // Configuration
    const emotions = [
        "joy", "sadness", "anger", "fear",
        "surprise", "trust", "disgust", "anticipation"
    ];

    const emotionColors = {
        "joy": "hsl(60, 100%, 50%)",
        "sadness": "hsl(240, 100%, 70%)",
        "anger": "hsl(0, 100%, 50%)",
        "fear": "hsl(270, 100%, 70%)",
        "surprise": "hsl(180, 100%, 50%)",
        "trust": "hsl(120, 100%, 40%)",
        "disgust": "hsl(300, 100%, 30%)",
        "anticipation": "hsl(40, 100%, 50%)"
    };

    // State
    let emotionState = {};
    let thoughtHistory = [];
    let memoryHistory = [];
    let emotionHistory = [];
    let emotionChart;
    let updateInterval;
    let animationSpeed = 1;

    // Initialize the application
    initEmotionBars();
    initEmotionGraph();
    fetchEmotionState();
    startUpdateInterval();

    // Event Listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    resetButton.addEventListener('click', resetEmotions);

    emotionSpeedSelect.addEventListener('change', (e) => {
        animationSpeed = parseFloat(e.target.value);
        updateEmotionBars(emotionState);
    });

    // Functions
    function initEmotionBars() {
        // Create emotion bars
        emotions.forEach(emotion => {
            const container = document.createElement('div');
            container.className = 'emotion-bar-container';
            container.dataset.emotion = emotion;

            const label = document.createElement('div');
            label.className = 'emotion-bar-label';

            const nameSpan = document.createElement('span');
            nameSpan.className = `color-${emotion}`;
            nameSpan.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);

            const valueSpan = document.createElement('span');
            valueSpan.className = 'emotion-value';
            valueSpan.textContent = '0';

            label.appendChild(nameSpan);
            label.appendChild(valueSpan);

            const track = document.createElement('div');
            track.className = 'emotion-bar-track';

            const progress = document.createElement('div');
            progress.className = `emotion-bar-progress bg-${emotion}`;
            progress.style.width = '50%'; // Start at neutral (50%)
            progress.style.opacity = '0.3'; // Low opacity for neutral

            track.appendChild(progress);

            container.appendChild(label);
            container.appendChild(track);

            emotionBarsContainer.appendChild(container);
        });
    }

    function initEmotionGraph() {
        // Create chart.js graph
        const ctx = emotionGraphCanvas.getContext('2d');

        emotionChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [], // Will be populated with time points
                datasets: emotions.map(emotion => ({
                    label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
                    data: [],
                    borderColor: emotionColors[emotion],
                    backgroundColor: `${emotionColors[emotion]}33`, // Add transparency
                    borderWidth: 2,
                    tension: 0.4,
                    fill: false
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        min: 0,
                        max: 1,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.7)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: 'rgba(255, 255, 255, 0.7)',
                            boxWidth: 12,
                            font: {
                                size: 10
                            }
                        }
                    }
                },
                animation: {
                    duration: 300
                }
            }
        });
    }

    function startUpdateInterval() {
        // Update emotion state and UI every 2 seconds
        updateInterval = setInterval(() => {
            fetchEmotionState();
        }, 2000);
    }

    async function fetchEmotionState() {
        try {
            const response = await fetch('/api/emotion/state');
            if (!response.ok) throw new Error('Failed to fetch emotion state');

            const data = await response.json();
            updateEmotionState(data);
        } catch (error) {
            console.error('Error fetching emotion state:', error);
        }
    }

    function updateEmotionState(data) {
        emotionState = data;

        // Update UI components
        updateEmotionBars(data);
        updateThoughts(data.thoughts);
        updateEmotionGraph(data.history);
    }

    function updateEmotionBars(data) {
        if (!data.emotions) return;

        emotions.forEach(emotion => {
            const value = data.emotions[emotion];
            const container = emotionBarsContainer.querySelector(`[data-emotion="${emotion}"]`);
            if (!container) return;

            const valueSpan = container.querySelector('.emotion-value');
            const progress = container.querySelector('.emotion-bar-progress');

            // Convert 0-1 value to display format (-5 to +5)
            const displayValue = Math.round((value - 0.5) * 10);
            valueSpan.textContent = displayValue > 0 ? `+${displayValue}` : displayValue;

            // Update progress bar width
            progress.style.width = `${value * 100}%`;

            // Update color intensity based on distance from neutral
            const neutralDistance = Math.abs(value - 0.5);
            const opacity = 0.3 + (neutralDistance * 1.4); // 0.3 to 1.0
            progress.style.opacity = opacity.toString();

            // Make the animation faster or slower based on user setting
            progress.style.transition = `width ${0.5 / animationSpeed}s ease, opacity ${0.5 / animationSpeed}s ease`;

            // Add visual pulse for significant changes
            if (neutralDistance > 0.15) {
                progress.classList.add('pulse');
                setTimeout(() => progress.classList.remove('pulse'), 1000);
            }
        });
    }

    function updateThoughts(thoughts) {
        if (!thoughts || !thoughts.length) return;

        // Check for new thoughts
        const newThoughts = thoughts.filter(t => {
            return !thoughtHistory.some(ht => ht.text === t.text && ht.time === t.time);
        });

        // Add new thoughts to history
        thoughtHistory = [...thoughtHistory, ...newThoughts];

        // Keep only recent thoughts
        if (thoughtHistory.length > 5) {
            thoughtHistory = thoughtHistory.slice(-5);
        }

        // Display new thoughts
        newThoughts.forEach(thought => {
            displayThought(thought);
        });
    }

    function displayThought(thought) {
        const clone = thoughtTemplate.content.cloneNode(true);
        const bubble = clone.querySelector('.thought-bubble');
        const text = clone.querySelector('.thought-text');

        text.textContent = thought.text;

        // Style based on dominant emotion
        let dominantEmotion = 'neutral';
        let maxDelta = 0;

        for (const emotion in thought.emotions) {
            const delta = Math.abs(thought.emotions[emotion] - 0.5);
            if (delta > maxDelta) {
                maxDelta = delta;
                dominantEmotion = emotion;
            }
        }

        if (dominantEmotion !== 'neutral') {
            bubble.style.borderLeft = `3px solid ${emotionColors[dominantEmotion]}`;
        }

        // Add to container with animation
        thoughtsContainer.appendChild(bubble);

        // Remove after animation completes
        setTimeout(() => {
            bubble.remove();
        }, 8000);
    }

    function updateEmotionGraph(history) {
        if (!history || !history.length) return;

        // Update history data
        emotionHistory = history;

        // Prepare data for chart
        const labels = Array.from({ length: history.length }, (_, i) => i);

        // Update each emotion dataset
        emotions.forEach((emotion, index) => {
            emotionChart.data.datasets[index].data = history.map(state => state[emotion]);
        });

        emotionChart.data.labels = labels;
        emotionChart.update();
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Display user message
        addMessage('user', message);

        // Clear input
        userInput.value = '';

        // Disable input while waiting for response
        userInput.disabled = true;
        sendButton.disabled = true;

        try {
            // Send to backend
            const response = await fetch('/api/chat/message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            });

            if (!response.ok) throw new Error('Failed to send message');

            const data = await response.json();

            // Display assistant response
            addMessage('assistant', data.response, data.emotion_state);

            // Update emotion state
            updateEmotionState(data.emotion_state);

        } catch (error) {
            console.error('Error sending message:', error);
            addMessage('assistant', 'Sorry, I encountered an error processing your message.');
        } finally {
            // Re-enable input
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
        }
    }

    function addMessage(role, content, emotionState = null) {
        const template = role === 'user' ? userMessageTemplate : assistantMessageTemplate;
        const clone = template.content.cloneNode(true);
        const message = clone.querySelector('.message');
        const messageContent = clone.querySelector('.message-content');

        messageContent.textContent = content;

        // Add emotion styling for assistant messages
        if (role === 'assistant' && emotionState) {
            const dominantEmotions = emotionState.dominant;
            if (dominantEmotions && dominantEmotions.length) {
                const [dominantEmotion, intensity] = dominantEmotions[0];
                const badge = message.querySelector('.message-emotion-badge');

                // Style badge with dominant emotion color
                badge.style.backgroundColor = emotionColors[dominantEmotion];
                badge.title = `${dominantEmotion}: ${intensity > 0 ? '+' : ''}${intensity}`;

                // Style message text
                messageContent.style.color = `${emotionColors[dominantEmotion]}aa`;

                // Adjust message styling based on emotion
                if (['joy', 'trust', 'anticipation'].includes(dominantEmotion)) {
                    message.style.borderLeft = `3px solid ${emotionColors[dominantEmotion]}`;
                    // More positive/warm styling
                } else if (['sadness', 'fear'].includes(dominantEmotion)) {
                    message.style.opacity = '0.85';
                    message.style.borderLeft = `3px solid ${emotionColors[dominantEmotion]}`;
                    // More subdued styling
                } else if (['anger', 'disgust'].includes(dominantEmotion)) {
                    message.style.borderLeft = `3px solid ${emotionColors[dominantEmotion]}`;
                    message.style.borderBottom = `1px solid ${emotionColors[dominantEmotion]}`;
                    // More intense styling
                }
            }
        }

        // Add message to chat
        chatMessages.appendChild(message);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function resetEmotions() {
        try {
            const response = await fetch('/api/emotions/reset', {
                method: 'POST'
            });

            if (!response.ok) throw new Error('Failed to reset emotions');

            const data = await response.json();
            updateEmotionState(data.emotion_state);

        } catch (error) {
            console.error('Error resetting emotions:', error);
        }
    }

    // Add visual effects for emotion bars
    document.head.insertAdjacentHTML('beforeend', `
        <style>
            @keyframes pulse {
                0% { transform: scaleY(1); }
                50% { transform: scaleY(1.1); }
                100% { transform: scaleY(1); }
            }

            .emotion-bar-progress.pulse {
                animation: pulse 0.5s ease;
            }
        </style>
    `);
});