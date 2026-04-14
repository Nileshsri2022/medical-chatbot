// Global variables
        let isConnected = false;
        let ragConnected = false;
        let llmConnected = false;
        let isLoading = false;
        let sessionId = null;
        let conversationContext = {};
        let interactionCount = 0;
        let historyCollapsed = false;
        
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
            
            // Attach history toggle click handler
            const historyToggleBtn = document.getElementById('historyToggle');
            if (historyToggleBtn) {
                console.log('📜 [HISTORY] Attaching click listener to history toggle button');
                historyToggleBtn.addEventListener('click', toggleConversationHistory);
            } else {
                console.warn('📜 [HISTORY] historyToggle button not found in DOM');
            }
        });

        function initializeSession() {
            const storedSessionId = localStorage.getItem('medical_rag_session_id');
            
            if (storedSessionId) {
                sessionId = storedSessionId;
                console.log('💾 [SESSION] Restored persisted session from localStorage:', sessionId.substring(0, 16) + '...');
            } else {
                sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                localStorage.setItem('medical_rag_session_id', sessionId);
                console.log('💾 [SESSION] Created new session and saved to localStorage:', sessionId.substring(0, 16) + '...');
            }
            
            document.getElementById('sessionId').textContent = sessionId.substring(0, 16) + '...';
            updateSessionStats();
            
            // Reconstruct chat UI bubbles if history exists in SQLite
            loadConversationHistory();
        }

        async function loadConversationHistory() {
            console.log('💾 [SESSION] Loading conversation history for session:', sessionId.substring(0, 16) + '...');
            try {
                const response = await fetch(`${RAG_ENDPOINT}/api/v1/conversation-history/${sessionId}`);
                if (response.ok) {
                    const data = await response.json();
                    const history = data.conversation_context?.conversation_history || [];
                    console.log('💾 [SESSION] Loaded', history.length, 'conversation turns from backend');
                    renderConversationHistory(history);

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
                renderConversationHistory([]);
            }
        }

        function toggleConversationHistory() {
            console.log('📜 [HISTORY] Toggle clicked. Current state:', historyCollapsed);
            historyCollapsed = !historyCollapsed;
            console.log('📜 [HISTORY] New collapsed state:', historyCollapsed);
            
            const historyList = document.getElementById('conversationHistoryList');
            const icon = document.getElementById('historyToggleIcon');

            if (!historyList || !icon) {
                console.warn('📜 [HISTORY] Elements not found!');
                return;
            }

            historyList.classList.toggle('collapsed', historyCollapsed);
            icon.textContent = historyCollapsed ? '▶' : '▼';
            console.log('📜 [HISTORY] Toggled. Icon:', icon.textContent, 'Display:', window.getComputedStyle(historyList).display);
        }

        function escapeHtml(value) {
            return String(value || '')
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }

        function formatAssistantResponse(rawText) {
            const normalized = String(rawText || '')
                .replace(/\r/g, '')
                .replace(/(Possible symptoms identified:?)/gi, '\n$1')
                .replace(/(Possible conditions to discuss with a clinician:?)/gi, '\n$1')
                .replace(/(If symptoms are severe[^\n]*)/gi, '\n$1')
                .replace(/(🚨[^\n]*)/g, '\n$1')
                .replace(/(⚠️[^\n]*)/g, '\n$1')
                .replace(/(\d+)%\s*Symptom\s*fit/gi, '$1% | Symptom fit')
                .trim();
            if (!normalized) return '<div class="assistant-rich"><p>No response generated.</p></div>';

            const lines = normalized.split('\n').map((line) => line.trim());
            const parts = [];
            let listType = null;

            const closeList = () => {
                if (listType) {
                    parts.push(`</${listType}>`);
                    listType = null;
                }
            };

            const inlineFormat = (text) => {
                return escapeHtml(text)
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/`(.*?)`/g, '<code>$1</code>');
            };

            const parseMetrics = (text) => {
                const illnessMatch = text.match(/illness\s*(\d+)%/i);
                const symptomMatch = text.match(/symptom\s*fit\s*(\d+)%/i);
                return {
                    illnessPct: illnessMatch ? illnessMatch[1] : null,
                    symptomPct: symptomMatch ? symptomMatch[1] : null,
                };
            };

            const renderConditionCard = (rank, conditionText, fallbackMetricsLine = '') => {
                const fromText = parseMetrics(conditionText);
                const fromFallback = parseMetrics(fallbackMetricsLine);
                const illnessPct = fromText.illnessPct || fromFallback.illnessPct;
                const symptomPct = fromText.symptomPct || fromFallback.symptomPct;

                const cleanName = conditionText
                    .replace(/\(.*?\)/g, '')
                    .replace(/match\s*quality\s*:/gi, '')
                    .replace(/illness\s*\d+%/gi, '')
                    .replace(/symptom\s*fit\s*\d+%/gi, '')
                    .replace(/[|,]+/g, ' ')
                    .replace(/\s+/g, ' ')
                    .trim() || conditionText;

                let metrics = '';
                if (illnessPct || symptomPct) {
                    metrics = '<div class="condition-metrics">';
                    if (illnessPct) metrics += `<span class="condition-metric">Illness ${illnessPct}%</span>`;
                    if (symptomPct) metrics += `<span class="condition-metric">Symptom fit ${symptomPct}%</span>`;
                    metrics += '</div>';
                }

                return `
                    <div class="condition-item">
                        <span class="condition-rank">${rank}</span>
                        <div class="condition-body">
                            <div class="condition-text">${inlineFormat(cleanName)}</div>
                            ${metrics}
                        </div>
                    </div>
                `;
            };

            const extractInlineConditions = (text) => {
                const segments = [];
                const regex = /(\d+)[.)]\s*([\s\S]*?)(?=(?:\s+\d+[.)]\s*)|$)/g;
                let match;
                while ((match = regex.exec(text)) !== null) {
                    const rank = match[1];
                    const payload = (match[2] || '').trim();
                    if (payload) {
                        segments.push({ rank, payload });
                    }
                }
                return segments;
            };

            for (let i = 0; i < lines.length; i += 1) {
                const trimmed = lines[i];

                if (!trimmed) {
                    closeList();
                    continue;
                }

                // Handle split condition pattern: "1" then condition name then optional metrics line
                if (/^\d+$/.test(trimmed)) {
                    closeList();
                    const rank = trimmed;
                    const next = lines[i + 1] || '';
                    const next2 = lines[i + 2] || '';

                    if (next) {
                        const metricsLikely = /(illness\s*\d+%|symptom\s*fit\s*\d+%)/i.test(next2) ? next2 : '';
                        parts.push(renderConditionCard(rank, next, metricsLikely));
                        i += metricsLikely ? 2 : 1;
                        continue;
                    }
                }

                const ordered = trimmed.match(/^(\d+)[.)]\s*(.+)/);
                if (ordered) {
                    closeList();
                    const inlineConditions = extractInlineConditions(trimmed);
                    if (inlineConditions.length > 1) {
                        inlineConditions.forEach((entry) => {
                            parts.push(renderConditionCard(entry.rank, entry.payload));
                        });
                    } else {
                        const rank = ordered[1];
                        const conditionText = ordered[2];
                        parts.push(renderConditionCard(rank, conditionText));
                    }
                    continue;
                }

                const bullet = trimmed.match(/^[-*•]\s+(.+)/);
                if (bullet) {
                    if (listType !== 'ul') {
                        closeList();
                        parts.push('<ul class="assistant-bullet-list">');
                        listType = 'ul';
                    }
                    parts.push(`<li>${inlineFormat(bullet[1])}</li>`);
                    continue;
                }

                closeList();

                const labelLine = trimmed.match(/^(Possible symptoms identified|Possible conditions to discuss with a clinician|Important|Summary|Analysis):\s*(.*)$/i);
                if (labelLine) {
                    const label = inlineFormat(labelLine[1]);
                    const value = (labelLine[2] || '').trim();

                    if (/possible symptoms identified/i.test(labelLine[1])) {
                        const symptomParts = value
                            .split(',')
                            .map((item) => item.trim())
                            .filter(Boolean)
                            .slice(0, 12)
                            .map((item) => `<li class="symptom-chip">${inlineFormat(item)}</li>`)
                            .join('');

                        if (symptomParts) {
                            parts.push(`<div class="assistant-section"><h5>${label}</h5><ul class="symptom-chip-list">${symptomParts}</ul></div>`);
                        } else {
                            parts.push(`<div class="assistant-section"><h5>${label}</h5></div>`);
                        }
                    } else {
                        if (/possible conditions to discuss with a clinician/i.test(labelLine[1]) && value) {
                            parts.push(`<div class="assistant-section"><h5>${label}</h5></div>`);
                            const inlineConditions = extractInlineConditions(value);
                            if (inlineConditions.length > 0) {
                                inlineConditions.forEach((entry) => {
                                    parts.push(renderConditionCard(entry.rank, entry.payload));
                                });
                            } else {
                                parts.push(`<p>${inlineFormat(value)}</p>`);
                            }
                        } else if (value) {
                            parts.push(`<div class="assistant-section"><h5>${label}</h5><p>${inlineFormat(value)}</p></div>`);
                        } else {
                            parts.push(`<div class="assistant-section"><h5>${label}</h5></div>`);
                        }
                    }
                    continue;
                }

                if (/^(Possible symptoms identified|Possible conditions to discuss with a clinician|Important|Summary|Analysis)$/i.test(trimmed)) {
                    parts.push(`<h5>${inlineFormat(trimmed)}</h5>`);
                    continue;
                }

                if (/^[🚨⚠]/.test(trimmed)) {
                    parts.push(`<div class="assistant-alert">${inlineFormat(trimmed)}</div>`);
                    continue;
                }

                if (/^if symptoms are severe/i.test(trimmed)) {
                    parts.push(`<div class="assistant-alert assistant-alert-soft">${inlineFormat(trimmed)}</div>`);
                    continue;
                }

                if (/^(analysis|summary|important|possible conditions to consider|symptoms identified|recommended next steps|when to seek emergency care)[:]?$/i.test(trimmed)) {
                    parts.push(`<h5>${inlineFormat(trimmed.replace(/:$/, ''))}</h5>`);
                    continue;
                }

                parts.push(`<p>${inlineFormat(trimmed)}</p>`);
            }

            closeList();
            return `<div class="assistant-rich">${parts.join('')}</div>`;
        }

        function resolveDisplaySymptoms(data) {
            const direct = data?.symptoms_detected || data?.symptoms || [];
            if (Array.isArray(direct) && direct.length > 0) {
                return direct;
            }

            const accumulated = data?.conversation_context?.accumulated_symptoms || [];
            if (Array.isArray(accumulated) && accumulated.length > 0) {
                return accumulated.map((name) => ({
                    symptom: name,
                    confidence: data?.confidence_score || 0,
                    urgency: data?.conversation_context?.urgency_level || 'unknown',
                }));
            }

            return [];
        }

        function toSnippet(value, maxLen = 500) {
            const cleaned = String(value || '').replace(/\s+/g, ' ').trim();
            if (cleaned.length <= maxLen) return cleaned;
            return `${cleaned.slice(0, maxLen)}...`;
        }

        function renderConversationHistory(historyTurns = []) {
            const list = document.getElementById('conversationHistoryList');
            const count = document.getElementById('historyCount');
            if (!list || !count) return;

            count.textContent = historyTurns.length;

            if (!historyTurns.length) {
                list.innerHTML = '<div class="history-empty">No previous exchanges yet.</div>';
                return;
            }

            const sorted = [...historyTurns].reverse();
            list.innerHTML = sorted.map((turn, index) => {
                const turnNumber = historyTurns.length - index;
                const userText = escapeHtml(toSnippet(turn.user_input || ''));
                const aiText = escapeHtml(toSnippet(turn.ai_response || ''));
                const confidence = Number(turn.confidence_score || 0);

                return `
                    <div class="history-entry">
                        <div class="history-entry-header">
                            <span class="history-turn">Turn ${turnNumber}</span>
                            <span>${(confidence * 100).toFixed(0)}% confidence</span>
                        </div>
                        <div class="history-user"><span class="history-label">You:</span> ${userText || 'N/A'}</div>
                        <div class="history-assistant"><span class="history-label">AI:</span> ${aiText || 'N/A'}</div>
                    </div>
                `;
            }).join('');
        }

        async function refreshConversationHistoryPanel() {
            if (!sessionId) return;
            try {
                const response = await fetch(`${RAG_ENDPOINT}/api/v1/conversation-history/${sessionId}`);
                if (!response.ok) return;
                const data = await response.json();
                const history = data.conversation_context?.conversation_history || [];
                renderConversationHistory(history);
            } catch (e) {
                console.error('Failed to refresh history panel:', e);
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
                const ragResponse = await fetch(`${RAG_ENDPOINT}/api/v1/health`, {
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
                    // Try streaming first, fallback to non-streaming
                    try {
                        console.log('🚀 [STREAMING] Initiating streaming request to /api/v1/chat/stream');
                        response = await fetch(`${RAG_ENDPOINT}/api/v1/chat/stream`, {
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
                            let streamErrorBody = '';
                            try {
                                streamErrorBody = await response.text();
                            } catch (readErr) {
                                console.error('❌ [STREAMING] Failed to read error body:', readErr);
                            }
                            console.warn('⚠️ [STREAMING] Stream endpoint error body:', streamErrorBody);
                            console.warn('⚠️ [STREAMING] Streaming endpoint returned:', response.status);
                            throw new Error(`Streaming not available (HTTP ${response.status}) ${streamErrorBody}`);
                        }

                        console.log('✅ [STREAMING] Connection established with status:', response.status);
                        const reader = response.body.getReader();
                        const decoder = new TextDecoder();
                        let responseData = null;
                        let fullText = '';
                        let chunkCount = 0;

                        removeLoadingMessage();
                        const streamMessageDiv = createStreamingMessageElement();
                        const messagesContainer = document.getElementById('chatMessages');
                        messagesContainer.appendChild(streamMessageDiv);
                        const contentElement = streamMessageDiv.querySelector('.streaming-content');
                        console.log('📝 [STREAMING] Streaming message element created and added to DOM');

                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) {
                                console.log('🏁 [STREAMING] Stream completed. Total chunks received:', chunkCount);
                                break;
                            }

                            const chunk = decoder.decode(value, { stream: true });
                            const lines = chunk.split('\n');

                            for (const line of lines) {
                                if (line.startsWith('data: ')) {
                                    try {
                                        const jsonData = JSON.parse(line.slice(6));
                                        if (jsonData.error) {
                                            console.error('❌ [STREAMING] Stream error:', jsonData.error);
                                            contentElement.innerHTML += `<p style="color: #dc3545;">❌ ${jsonData.error}</p>`;
                                        } else if (jsonData.type === 'start') {
                                            console.log('▶️ [STREAMING] Stream started');
                                            contentElement.innerHTML = `<p>${jsonData.text}</p>`;
                                        } else if (jsonData.type === 'chunk') {
                                            chunkCount++;
                                            fullText += jsonData.text;
                                            contentElement.lastChild.textContent = fullText;
                                            if (chunkCount % 10 === 0) {
                                                console.log(`📊 [STREAMING] Received ${chunkCount} chunks, text length: ${fullText.length}`);
                                            }
                                        } else if (jsonData.type === 'complete') {
                                            console.log('✅ [STREAMING] Received complete response with data:', jsonData.data);
                                            responseData = jsonData.data;
                                        }
                                        messagesContainer.scrollTop = messagesContainer.scrollHeight;
                                    } catch (e) {
                                        console.error('❌ [STREAMING] JSON parse error:', e, 'Line:', line);
                                    }
                                }
                            }
                        }

                        if (responseData) {
                            const endTime = Date.now();
                            const responseTime = (endTime - startTime) / 1000;
                            
                            streamMessageDiv.innerHTML = `
                                <div class="message-content">
                                    <div class="context-indicators">
                                        <span class="context-tag symptoms">🔍 Symptoms: ${resolveDisplaySymptoms(responseData).length}</span>
                                        <span class="context-tag confidence">📊 Confidence: ${(responseData.confidence_score * 100).toFixed(0)}%</span>
                                        <span class="context-tag urgency">⚡ ${(responseData.conversation_context?.urgency_level || 'low').toUpperCase()}</span>
                                    </div>
                                    ${formatAssistantResponse(fullText || responseData.response)}
                                </div>
                                <div class="timestamp">
                                    ${new Date().toLocaleTimeString()} 
                                    <span class="processing-time">(${responseTime.toFixed(2)}s)</span>
                                    <span class="enhanced-indicator">🧠 RAG Streaming</span>
                                </div>
                            `;
                            
                            updateContextPanel(responseData);
                            conversationContext = responseData.conversation_context;
                            interactionCount++;
                            updateSessionStats();
                            await refreshConversationHistoryPanel();
                        }
                    } catch (streamError) {
                        // Fallback to non-streaming
                        console.log('Streaming unavailable, using standard mode:', streamError);
                        addSystemMessage(`⚠️ Streaming unavailable, fallback to standard response. Reason: ${streamError.message}`);
                        response = await fetch(`${RAG_ENDPOINT}/api/v1/chat`, {
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
                            throw new Error('Failed to get response');
                        }
                        
                        removeLoadingMessage();
                        const data = await response.json();
                        const endTime = Date.now();
                        const responseTime = (endTime - startTime) / 1000;
                        
                        addEnhancedMessage(data, responseTime);
                        updateContextPanel(data);
                        conversationContext = data.conversation_context;
                        
                        interactionCount++;
                        updateSessionStats();
                        await refreshConversationHistoryPanel();
                    }
                    
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
            const symptoms = resolveDisplaySymptoms(data);
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
                    ${formatAssistantResponse(data.response)}
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

        function createStreamingMessageElement() {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message bot-message streaming-message';
            messageDiv.innerHTML = `
                <div class="message-content">
                    <div class="streaming-content">
                        <p>🧠 Analyzing your symptoms...</p>
                    </div>
                </div>
                <div class="timestamp">
                    ${new Date().toLocaleTimeString()} 
                    <span class="enhanced-indicator">🧠 RAG Streaming</span>
                </div>
            `;
            return messageDiv;
        }

        function addMessage(content, isUser = false, processingTime = null) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;

            const renderedContent = isUser
                ? escapeHtml(content).replace(/\n/g, '<br>')
                : formatAssistantResponse(content);
            
            let timestampHTML = `<div class="timestamp">${new Date().toLocaleTimeString()}`;
            if (processingTime) {
                timestampHTML += ` <span class="processing-time">(${processingTime.toFixed(2)}s)</span>`;
            }
            timestampHTML += `</div>`;
            
            messageDiv.innerHTML = `
                <div class="message-content">${renderedContent}</div>
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
            const symptoms = resolveDisplaySymptoms(data);
            
            console.log('🏥 [SYMPTOMS] Updating symptoms panel with', symptoms.length, 'detected symptoms');
            
            if (symptoms.length > 0) {
                symptomsContainer.innerHTML = symptoms.map(symptom => {
                    const name = typeof symptom === 'string' ? symptom : symptom.symptom || 'Unknown';
                    const confidence = typeof symptom === 'object' ? symptom.confidence || 0 : 0;
                    const urgency = typeof symptom === 'object' ? symptom.urgency || 'unknown' : 'unknown';
                    const confidencePercent = Math.round(confidence * 100);
                    
                    // Determine confidence level color
                    let confidenceLevel = 'low';
                    if (confidencePercent >= 70) {
                        confidenceLevel = 'high';
                    } else if (confidencePercent >= 40) {
                        confidenceLevel = 'medium';
                    }
                    
                    console.log(`  🏥 ${name}: ${confidencePercent}% confidence (${confidenceLevel}), urgency: ${urgency}`);
                    
                    return `
                        <div class="symptom-item">
                            <div class="symptom-name">${escapeHtml(name)}</div>
                            <div class="symptom-details">
                                <div class="confidence-container">
                                    <div class="confidence-bar">
                                        <div class="confidence-fill confidence-${confidenceLevel}" style="width: ${confidencePercent}%"></div>
                                    </div>
                                    <span class="confidence-label">${confidencePercent}%</span>
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
                    console.log('💾 [SESSION] Resetting session:', sessionId.substring(0, 16) + '...');
                    await fetch(`${RAG_ENDPOINT}/api/v1/conversation/${sessionId}`, {
                        method: 'DELETE'
                    });
                    
                    // Clear local storage Session ID to start completely fresh
                    localStorage.removeItem('medical_rag_session_id');
                    console.log('💾 [SESSION] Cleared localStorage, generating new session...');
                    
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
                    renderConversationHistory([]);
                    
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
