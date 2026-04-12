// Global variables
        let isConnected = false;
        let ragConnected = false;
        let llmConnected = false;
        let isLoading = false;
        let sessionId = null;
        let conversationContext = {};
        let interactionCount = 0;
        
        const RAG_ENDPOINT = 'http://localhost:8002';
        const DIRECT_LLM_ENDPOINT = 'http://localhost:8001';

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            initializeSession();
            testConnection();
            document.getElementById('messageInput').focus();
            
            // Auto-resize textarea
            const textarea = document.getElementById('messageInput');
            textarea.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = this.scrollHeight + 'px';
            });
        });

        function initializeSession() {
            sessionId = localStorage.getItem('medical_rag_session_id');
            if (!sessionId) {
                sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                localStorage.setItem('medical_rag_session_id', sessionId);
            }
            document.getElementById('sessionId').textContent = sessionId.substring(0, 16) + '...';
            updateSessionStats();
            
            // Reconstruct chat UI bubbles if history exists in SQLite
            loadConversationHistory();
        }

        async function loadConversationHistory() {
            try {
                const response = await fetch(`${RAG_ENDPOINT}/conversation-history/${sessionId}`);
                if (response.ok) {
                    const data = await response.json();
                    if (data.conversation_context && data.conversation_context.conversation_history) {
                        for (const turn of data.conversation_context.conversation_history) {
                            addMessage(turn.user_input, true);
                            
                            // Reconstruct AI response visually
                            const msgData = {
                                response: turn.ai_response,
                                symptoms_detected: turn.extracted_symptoms,
                                extracted_entities: turn.extracted_entities,
                                confidence_score: turn.confidence_score,
                                conversation_context: data.conversation_context
                            };
                            
                            // Emulate streaming finish chunk
                            addEnhancedMessage(msgData, 0.0);
                        }
                        updateContextPanel(data);
                        interactionCount = data.total_interactions;
                        updateSessionStats();
                    }
                }
            } catch (e) {
                console.error("No history loaded", e);
            }
        }

        async function testConnection() {
            const statusElement = document.getElementById('connectionStatus');
            const testButton = document.getElementById('testButton');
            const ragStatus = document.getElementById('ragStatus');
            const llmStatus = document.getElementById('llmStatus');
            
            statusElement.className = 'connection-status connecting';
            testButton.disabled = true;
            testButton.textContent = 'Testing...';
            ragStatus.textContent = 'Testing...';
            llmStatus.textContent = 'Testing...';
            
            try {
                // Test RAG server
                const ragResponse = await fetch(`${RAG_ENDPOINT}/health`, {
                    method: 'GET',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                if (ragResponse.ok) {
                    ragStatus.textContent = '✅ Connected';
                    ragStatus.style.color = '#28a745';
                    ragConnected = true;
                } else {
                    ragStatus.textContent = '❌ Error';
                    ragStatus.style.color = '#dc3545';
                    ragConnected = false;
                }
                
                // Test LLM backend
                const llmResponse = await fetch(`${DIRECT_LLM_ENDPOINT}/health`, {
                    method: 'GET',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                if (llmResponse.ok) {
                    llmStatus.textContent = '✅ Connected';
                    llmStatus.style.color = '#28a745';
                    llmConnected = true;
                } else {
                    llmStatus.textContent = '❌ Error';
                    llmStatus.style.color = '#dc3545';
                    llmConnected = false;
                }
                
                // Update overall connection status
                isConnected = ragConnected && llmConnected;
                if (isConnected) {
                    statusElement.className = 'connection-status connected';
                    
                    addSystemMessage(`✅ Medical Assistant Ready!
                    
🧠 <strong>System Status:</strong>
• RAG Enhancement Engine: Active
• Medical Entity Recognition: Online  
• Conversation Memory: Initialized
• Symptom Analysis: Ready
• Context Building: Operational

🚀 Start describing your symptoms for intelligent medical guidance.`);
                    
                } else if (ragConnected || llmConnected) {
                    statusElement.className = 'connection-status partial';
                    addSystemMessage(`⚠️ Partial Connection Available
                    
🧠 <strong>System Status:</strong>
• RAG Server: ${ragConnected ? 'Connected ✅' : 'Disconnected ❌'}
• LLM Backend: ${llmConnected ? 'Connected ✅' : 'Disconnected ❌'}

${ragConnected ? '• RAG Mode available' : ''}
${llmConnected ? '• Direct LLM mode available' : ''}

Please use the appropriate mode based on available connections.`);
                } else {
                    throw new Error('Both servers not responding');
                }
                
            } catch (error) {
                // Handle connection errors - don't reset individual server status here
                // as they may have been set in the try block
                isConnected = ragConnected && llmConnected;
                statusElement.className = isConnected ? 'connection-status connected' : 'connection-status disconnected';
                
                addSystemMessage(`❌ Connection failed: ${error.message}
                
🔧 <strong>Setup Instructions:</strong>
1. Start RAG server: <code>python backend/medical_rag_server.py</code>
2. Start LLM backend: <code>python ../ssh-tunnel-chatbot-setup.md</code>
3. Verify endpoints:
   • RAG: ${RAG_ENDPOINT}
   • LLM: ${DIRECT_LLM_ENDPOINT}

💡 Make sure both servers are running before using the enhanced chatbot.`);
            }
            
            testButton.disabled = false;
            testButton.textContent = 'Test Connection';
        }

        function formatDirectLLMResponse(data) {
            // Format the direct LLM response with symptoms and illnesses
            try {
                if (data.symptoms && data.illnesses) {
                    let response = "Based on your symptoms, here's my analysis:\n\n";
                    
                    // Add symptoms section
                    if (data.symptoms.length > 0) {
                        response += "**Symptoms Identified:**\n";
                        data.symptoms.forEach((symptom, index) => {
                            response += `${index + 1}. ${symptom}\n`;
                        });
                        response += "\n";
                    }
                    
                    // Add illnesses section with coverage scores
                    if (data.illnesses.length > 0) {
                        response += "**Possible Conditions to Consider:**\n";
                        data.illnesses.slice(0, 5).forEach((illness, index) => {
                            const name = illness.name || illness;
                            const illnessCov = illness.illness_coverage || 0;
                            const conditionCov = illness.condition_coverage || 0;
                            
                            if (typeof illness === 'object' && (illnessCov || conditionCov)) {
                                response += `${index + 1}. ${name} (Illness: ${illnessCov}%, Condition: ${conditionCov}%)\n`;
                            } else {
                                response += `${index + 1}. ${name}\n`;
                            }
                        });
                        response += "\n";
                    }
                    
                    response += "**Important:** This is a preliminary assessment. Please consult with a healthcare professional for proper diagnosis and treatment.";
                    return response;
                } else {
                    // Fallback for other response formats
                    return data.response || data.answer || JSON.stringify(data, null, 2);
                }
            } catch (error) {
                console.error('Error formatting direct LLM response:', error);
                return 'Error formatting response. Please try again.';
            }
        }

        async function sendMessage() {
            const mode = document.getElementById('responseMode').value;
            
            // Check connection based on mode
            const requiredConnection = mode === 'enhanced' ? ragConnected : llmConnected;
            
            if (isLoading || !requiredConnection) {
                if (!requiredConnection) {
                    const serverName = mode === 'enhanced' ? 'RAG server' : 'LLM backend';
                    addSystemMessage(`❌ ${serverName} not connected. Please test connection first.`);
                }
                return;
            }

            const messageInput = document.getElementById('messageInput');
            const message = messageInput.value.trim();
            
            if (!message) return;

            const maxTokens = parseInt(document.getElementById('maxTokens').value);
            const temperature = parseFloat(document.getElementById('temperature').value);

            // Add user message
            addMessage(message, true);
            messageInput.value = '';
            messageInput.style.height = 'auto';

            // Show enhanced loading
            isLoading = true;
            document.getElementById('sendButton').disabled = true;
            addEnhancedLoadingMessage(mode);

            const startTime = Date.now();

            try {
                let response;
                
                if (mode === 'enhanced') {
                    // Use RAG enhancement STREAMING
                    response = await fetch(`${RAG_ENDPOINT}/enhanced-chat-stream`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: message,
                            session_id: sessionId,
                            max_tokens: maxTokens,
                            temperature: temperature
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Failed to get streaming response');
                    }
                    
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let textContainer = null;
                    let responseTime = 0;
                    let buffer = '';
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n\n');
                        buffer = lines.pop(); // Keep the last incomplete chunk in the buffer
                        
                        for (let block of lines) {
                            if (block.startsWith('data: ') && block !== 'data: [DONE]') {
                                try {
                                    const data = JSON.parse(block.substring(6));
                                    
                                    if (data.type === 'metadata') {
                                        removeLoadingMessage();
                                        const endTime = Date.now();
                                        responseTime = (endTime - startTime) / 1000;
                                        
                                        data.response = ''; // Empty initial response
                                        addEnhancedMessage(data, responseTime);
                                        updateContextPanel(data);
                                        conversationContext = data.conversation_context;
                                        
                                        // Grab the latest message content element we just added
                                        const contents = document.querySelectorAll('.enhanced-message .message-content');
                                        const latestContent = contents[contents.length - 1];
                                        
                                        // Create a text container specifically for the stream
                                        textContainer = document.createElement('span');
                                        latestContent.appendChild(textContainer);
                                    } else if (data.type === 'chunk' && textContainer) {
                                        textContainer.innerHTML += data.text.replace(/\n/g, '<br/>');
                                        const container = document.getElementById('chatMessages');
                                        container.scrollTop = container.scrollHeight;
                                    }
                                } catch (e) { }
                            }
                        }
                    }
                    interactionCount++;
                    updateSessionStats();
                    
                } else {
                    // Direct LLM call
                    response = await fetch(`${DIRECT_LLM_ENDPOINT}/diagnose`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            description: message,
                            max_tokens: maxTokens,
                            temperature: temperature
                        })
                    });
                    
                    removeLoadingMessage();
                    const endTime = Date.now();
                    const responseTime = (endTime - startTime) / 1000;

                    if (response.ok) {
                        const data = await response.json();
                        const formattedResponse = formatDirectLLMResponse(data);
                        addMessage(formattedResponse, false, responseTime);
                    } else {
                        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                        addMessage(`❌ Error: ${errorData.detail || 'Failed to get response'}`, false);
                    }
                }
            } catch (error) {
                removeLoadingMessage();
                addMessage(`🔌 Connection error: ${error.message}. Check server status.`, false);
            }

            isLoading = false;
            document.getElementById('sendButton').disabled = false;
            messageInput.focus();
        }

        function addEnhancedLoadingMessage(mode) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message bot-message';
            messageDiv.id = 'loadingMessage';
            
            const loadingText = mode === 'enhanced' 
                ? '🧠 AI is analyzing context, extracting medical entities, and building intelligent response...'
                : '🤖 AI is processing your message...';
            
            messageDiv.innerHTML = `
                <div class="message-content loading">
                    <span>${loadingText}</span>
                    <div class="loading-dots">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                </div>
            `;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function addEnhancedMessage(data, responseTime) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message bot-message enhanced-message';
            
            // Create context indicators
            let contextInfo = '';
            const symptoms = data.symptoms_detected || data.symptoms || [];
            const entities = data.extracted_entities || {};
            const confidence = data.confidence_score || 0;
            
            if (symptoms.length > 0 || Object.keys(entities).length > 0 || confidence > 0) {
                contextInfo += '<div class="context-indicators">';
                
                if (symptoms.length > 0) {
                    contextInfo += `<span class="context-tag symptoms">🔍 Symptoms: ${symptoms.length}</span>`;
                }
                
                if (entities && Object.keys(entities).length > 0) {
                    const entityCount = Object.values(entities).flat().filter(e => e && e.length > 0).length;
                    if (entityCount > 0) {
                        contextInfo += `<span class="context-tag entities">🏷️ Entities: ${entityCount}</span>`;
                    }
                }
                
                if (confidence > 0) {
                    contextInfo += `<span class="context-tag confidence">📊 Confidence: ${(confidence * 100).toFixed(0)}%</span>`;
                }
                
                if (data.conversation_context && data.conversation_context.urgency_level && data.conversation_context.urgency_level !== 'low') {
                    contextInfo += `<span class="context-tag urgency">⚡ ${data.conversation_context.urgency_level.toUpperCase()}</span>`;
                }
                
                if (data.conversation_context && data.conversation_context.conversation_state) {
                    contextInfo += `<span class="context-tag state">🎯 ${data.conversation_context.conversation_state.replace('_', ' ')}</span>`;
                }
                
                contextInfo += '</div>';
            }
            
            messageDiv.innerHTML = `
                <div class="message-content">
                    ${contextInfo}
                    ${data.response}
                </div>
                <div class="timestamp">
                    ${new Date().toLocaleTimeString()} 
                    <span class="processing-time">(${responseTime.toFixed(2)}s)</span>
                    <span class="enhanced-indicator">🧠 RAG Mode</span>
                </div>
            `;
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function addMessage(content, isUser = false, processingTime = null) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            
            let timestampHTML = `<div class="timestamp">${new Date().toLocaleTimeString()}`;
            if (processingTime) {
                timestampHTML += ` <span class="processing-time">(${processingTime.toFixed(2)}s)</span>`;
            }
            timestampHTML += `</div>`;
            
            messageDiv.innerHTML = `
                <div class="message-content">${content}</div>
                ${timestampHTML}
            `;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function addSystemMessage(message) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message system-message';
            messageDiv.innerHTML = `
                <div class="message-content">
                    ${message.replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/`(.*?)`/g, '<code>$1</code>')}
                </div>
                <div class="timestamp">${new Date().toLocaleTimeString()}</div>
            `;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function removeLoadingMessage() {
            const loadingMessage = document.getElementById('loadingMessage');
            if (loadingMessage) {
                loadingMessage.remove();
            }
        }

        function updateContextPanel(data) {
            // Update current symptoms
            const symptomsContainer = document.getElementById('currentSymptoms');
            const symptoms = data.symptoms_detected || data.symptoms || [];
            
            if (symptoms.length > 0) {
                symptomsContainer.innerHTML = symptoms.map(symptom => {
                    const name = typeof symptom === 'string' ? symptom : symptom.symptom || 'Unknown';
                    const confidence = typeof symptom === 'object' ? symptom.confidence || 0 : 0;
                    const urgency = typeof symptom === 'object' ? symptom.urgency || 'unknown' : 'unknown';
                    
                    return `
                        <div class="context-item">
                            <span>${name}</span>
                            <div>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${confidence * 100}%"></div>
                                </div>
                                <div class="urgency-indicator urgency-${urgency}">${urgency}</div>
                            </div>
                        </div>
                    `;
                }).join('');
            } else {
                symptomsContainer.innerHTML = '<div class="context-item"><span>No symptoms detected</span></div>';
            }
            
            // Update accumulated context
            const contextContainer = document.getElementById('accumulatedContext');
            const context = data.conversation_context || {};
            
            let contextHTML = '';
            if (context.accumulated_symptoms && context.accumulated_symptoms.length > 0) {
                contextHTML += `<div class="context-item"><span>Total Symptoms:</span><span>${context.accumulated_symptoms.length}</span></div>`;
            }
            if (context.accumulated_conditions && context.accumulated_conditions.length > 0) {
                contextHTML += `<div class="context-item"><span>Conditions:</span><span>${context.accumulated_conditions.length}</span></div>`;
            }
            if (context.conversation_summary) {
                contextHTML += `<div class="context-item" style="flex-direction: column; align-items: flex-start;"><span style="font-weight: bold;">Summary:</span><span style="font-size: 10px; margin-top: 4px;">${context.conversation_summary}</span></div>`;
            }
            
            if (contextHTML) {
                contextContainer.innerHTML = contextHTML;
            } else {
                contextContainer.innerHTML = '<div class="context-item"><span>Building context...</span></div>';
            }
        }

        function updateSessionStats() {
            document.getElementById('interactionCount').textContent = interactionCount;
            const state = conversationContext.conversation_state || 'initial';
            document.getElementById('conversationState').textContent = state.replace('_', ' ');
        }

        async function resetConversation() {
            if (confirm('Are you sure you want to reset the conversation? All context will be lost.')) {
                try {
                    await fetch(`${RAG_ENDPOINT}/reset-conversation/${sessionId}`, {
                        method: 'DELETE'
                    });
                    
                    // Clear local storage Session ID to start completely fresh
                    localStorage.removeItem('medical_rag_session_id');
                    
                    // Reset local state
                    document.getElementById('chatMessages').innerHTML = '';
                    conversationContext = {};
                    interactionCount = 0;
                    
                    // Auto-generate new fresh session ID
                    initializeSession();
                    updateSessionStats();
                    
                    // Clear messages except system message
                    const messagesContainer = document.getElementById('chatMessages');
                    messagesContainer.innerHTML = `
                        <div class="message system-message">
                            <div class="message-content">
                                🔄 <strong>Conversation Reset</strong><br>
                                Session has been reset. All previous context cleared.<br>
                                Ready for a new conversation.
                            </div>
                        </div>
                    `;
                    
                    // Reset context panel
                    document.getElementById('currentSymptoms').innerHTML = '<div class="context-item"><span>No symptoms detected</span></div>';
                    document.getElementById('accumulatedContext').innerHTML = '<div class="context-item"><span>Building conversation context...</span></div>';
                    
                } catch (error) {
                    addSystemMessage(`❌ Error resetting conversation: ${error.message}`);
                }
            }
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        // Export session data for debugging
        function exportSessionData() {
            const sessionData = {
                sessionId: sessionId,
                interactionCount: interactionCount,
                conversationContext: conversationContext,
                timestamp: new Date().toISOString()
            };
            
            const blob = new Blob([JSON.stringify(sessionData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `medical-chat-session-${sessionId}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }

        // Add keyboard shortcut for export (Ctrl+E)
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'e') {
                e.preventDefault();
                exportSessionData();
            }
        });