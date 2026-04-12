# Medical RAG Chatbot System
## Healthcare - Advanced Conversational AI

![RAG System](https://img.shields.io/badge/RAG-Enhanced-blue) ![Medical AI](https://img.shields.io/badge/Medical-AI-green) ![FastAPI](https://img.shields.io/badge/FastAPI-Framework-red) ![Python](https://img.shields.io/badge/Python-3.7+-blue)

### 🧠 Intelligent Medical Conversation System with Memory & Context

This advanced RAG (Retrieval-Augmented Generation) system transforms a basic medical chatbot into an intelligent conversational assistant that:
- **Remembers** previous conversations and builds context over time
- **Extracts** medical entities, symptoms, and urgency indicators  
- **Analyzes** conversation flow and provides contextual responses
- **Maintains** conversation state and patient interaction history
- **Provides** real-time context visualization and confidence scoring

---
<img width="1220" height="826" alt="Medical AI Assistant" src="https://github.com/user-attachments/assets/d84a40e2-39e4-4bf1-9bb5-bff15e1b67ae" />


## 🚀 Architecture

<img width="892" height="1621" alt="oci-enhanced-architecture" src="https://github.com/user-attachments/assets/fdede137-7bf8-49d9-871c-e7f4199174c0" />


## 🚀 Quick Start Guide

### Prerequisites
- Python 3.7+
- Original medical LLM backend running on port 8000
- 4GB+ RAM recommended for optimal performance

### 1. Install Dependencies
```bash
cd RAG/
./scripts/install_dependencies.sh
```

### 2. Start the Enhanced System
```bash
./scripts/start_rag_system.sh
```

### 3. Access the Medical Chatbot
Open your browser to: **http://localhost:3000/enhanced-medical-chatbot.html**

### 4. Stop the System
```bash
./scripts/stop_rag_system.sh
```

---

## 🏗️ System Architecture

### Enhanced Conversation Flow
```
User Input → Medical NER → Symptom Extraction → Context Building → RAG Enhancement → LLM → Intelligent Response
     ↓             ↓              ↓                    ↓                ↓         ↓            ↓
 Raw Text → Medical Entities → Symptom Analysis → Conversation Memory → Enhanced Prompt → Context-Aware Response
```

### Service Architecture
```
Frontend (Port 3000)
    ↓
RAG Enhancement Server (Port 8001)
    ↓
Original LLM Backend (Port 8000)
```

### Component Overview

#### 🧠 **Medical RAG Engine** (`backend/medical_rag_engine.py`)
- **MedicalEntityRecognizer**: Extracts symptoms, conditions, medications, temporal info
- **SymptomExtractor**: Advanced symptom analysis with confidence scoring
- **ConversationMemory**: Maintains conversation state and patient interaction history
- **ContextBuilder**: Creates enriched context for intelligent prompt generation

#### 🔗 **RAG Server** (`backend/medical_rag_server.py`) 
- FastAPI-based middleware server
- Coordinates RAG components
- Manages conversation sessions
- Provides comprehensive APIs

#### 🖥️ **Medical Frontend** (`frontend/enhanced-medical-chatbot.html`)
- Real-time context visualization
- Conversation state tracking
- Symptom and entity display
- Confidence and urgency indicators

---

## 📊 Key Features & Benefits

### 🎯 **Conversation Continuity**
- **Before RAG**: Each message treated independently
- **After RAG**: Full conversation memory with context building

**Example:**
```
User: "I have chest pain"
RAG Analysis: Symptom=chest_pain, Urgency=critical, Context=new_conversation
AI Response: "I understand you're experiencing chest pain. This is concerning. When did this start and do you have any other symptoms like shortness of breath or sweating?"

User: "It started an hour ago and I feel nauseous"  
RAG Analysis: Symptom=nausea, Duration=acute, Previous_context=chest_pain
AI Response: "The combination of chest pain that started an hour ago with nausea is very concerning and could indicate a cardiac event. Please call emergency services (999) immediately or go to the nearest emergency department. While waiting, try to stay calm and avoid physical exertion."
```

### 🔍 **Medical Entity Recognition**
Automatically extracts and categorizes:
- **Symptoms**: Pain patterns, fever, gastrointestinal issues, neurological symptoms
- **Body Parts**: Anatomical locations and organ systems
- **Conditions**: Known medical conditions and diseases
- **Medications**: Current medications and treatments
- **Temporal Information**: Duration, onset, frequency
- **Severity Indicators**: Mild, moderate, severe, critical

### 📈 **Intelligent Context Building**
- **Conversation State Tracking**: Initial → Symptom Gathering → Analysis → Treatment Discussion
- **Urgency Assessment**: Low → Moderate → High → Critical
- **Follow-up Suggestion**: Context-aware next questions
- **Information Gap Analysis**: Identifies missing critical information

### 🎛️ **Real-time Visualization**
The enhanced frontend provides:
- **Context Panel**: Live display of extracted entities and conversation state
- **Confidence Indicators**: Visual confidence bars for symptom detection
- **Urgency Alerts**: Color-coded urgency levels with visual indicators
- **Session Statistics**: Interaction count, conversation duration, accumulated context

---

## 🛠️ Technical Implementation

### Directory Structure
```
RAG/
├── backend/
│   ├── medical_rag_engine.py      # Core RAG components
│   └── medical_rag_server.py      # FastAPI server
├── frontend/
│   └── enhanced-medical-chatbot.html  # Medical UI
├── scripts/
│   ├── install_dependencies.sh    # Dependency installer
│   ├── start_rag_system.sh       # System startup
│   └── stop_rag_system.sh        # System shutdown
├── docs/
│   ├── API_Documentation.md       # API reference
│   ├── Architecture_Guide.md      # Technical architecture
│   └── Troubleshooting.md        # Common issues
└── README.md                      # This file
```

### Core Components Deep Dive

#### 1. **Medical Entity Recognizer**
```python
# Extracts medical entities using pattern matching
entities = {
    "symptoms": ["chest pain", "nausea"],
    "body_parts": ["chest", "arm"], 
    "severity": "severe",
    "duration": "acute",
    "urgency_indicators": ["emergency", "immediate"]
}
```

#### 2. **Conversation Memory System**
```python
# Maintains conversation state and context
conversation_context = {
    "accumulated_symptoms": ["chest pain", "nausea", "sweating"],
    "conversation_state": "symptom_analysis", 
    "urgency_level": "critical",
    "total_interactions": 3,
    "conversation_summary": "Patient reporting acute chest pain with associated symptoms"
}
```

#### 3. **Context-Aware Prompt Building**
```python
# Creates enriched prompts for LLM
enriched_prompt = f"""
SYSTEM: You are an empathetic AI medical assistant. URGENT SITUATION DETECTED.

CONVERSATION CONTEXT:
- Total interactions: 3
- Previous symptoms: chest pain, nausea
- Current urgency: CRITICAL

MEDICAL CONTEXT:
- Current symptoms: sweating (confidence: 0.8, urgency: high)
- Severity: severe
- Duration: acute (started 1 hour ago)

CURRENT USER INPUT: "I'm also sweating a lot now"

Please provide immediate guidance while recommending emergency care.
"""
```

---

## 📡 API Documentation

### Enhanced Chat Endpoint
**POST** `/enhanced-chat`

#### Request
```json
{
    "message": "I have chest pain and feel nauseous",
    "session_id": "session_12345",
    "max_tokens": 200,
    "temperature": 0.7
}
```

#### Response
```json
{
    "response": "I understand you're experiencing chest pain with nausea...",
    "conversation_context": {
        "conversation_state": "symptom_analysis",
        "accumulated_symptoms": ["chest pain", "nausea"],
        "urgency_level": "critical",
        "total_interactions": 2
    },
    "extracted_entities": {
        "symptoms": ["chest pain", "nausea"],
        "severity": "unspecified",
        "duration": "unspecified"
    },
    "symptoms_detected": [
        {
            "symptom": "Chest Pain",
            "confidence": 0.9,
            "urgency": "critical",
            "possible_causes": ["heart attack", "angina", "anxiety"]
        }
    ],
    "confidence_score": 0.85,
    "processing_time": 1.23,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Additional Endpoints

- **GET** `/health` - System health check
- **GET** `/conversation-history/{session_id}` - Get conversation history
- **DELETE** `/reset-conversation/{session_id}` - Reset conversation memory
- **GET** `/session-stats` - Server and session statistics
- **GET** `/dev/test-rag` - Test RAG engine functionality

---

## 🔧 Configuration & Customization

### Environment Variables
```bash
# Server configuration
RAG_PORT=8001
FRONTEND_PORT=3000
ORIGINAL_LLM_PORT=8000
REQUEST_TIMEOUT=30.0

# RAG engine parameters
MAX_CONVERSATION_HISTORY=10
SYMPTOM_CONFIDENCE_THRESHOLD=0.6
ENTITY_EXTRACTION_TIMEOUT=5.0
```

### Customizing Medical Patterns
Edit `medical_rag_engine.py` to add new symptom patterns:

```python
self.symptom_database = {
    "your_new_symptom": {
        "aliases": ["symptom name", "alternative name"],
        "related_symptoms": ["related symptom 1", "related symptom 2"],
        "urgency": "moderate",  # low, moderate, high, critical
        "common_causes": ["cause 1", "cause 2"],
        "follow_up_questions": ["question 1", "question 2"]
    }
}
```

### Frontend Customization
The enhanced chatbot interface can be customized by modifying:
- **Colors & Themes**: Update CSS variables in the `<style>` section
- **Context Panel**: Modify the context display components
- **Message Format**: Customize message rendering functions

---

## 🐛 Troubleshooting

### Common Issues

#### 1. **RAG Server Won't Start**
```bash
# Check if port is in use
lsof -i :8001

# Check logs
tail -f backend/rag_server.log

# Kill existing process
./scripts/stop_rag_system.sh
```

#### 2. **Original LLM Not Accessible**
```bash
# Test original LLM
curl http://localhost:8000/health

# Check if SSH tunnel is running (if applicable)
ssh -L 8000:remote-server:8000 user@bastion-host
```

#### 3. **Frontend Connection Errors**
- Ensure both RAG server (8001) and original LLM (8000) are running
- Check browser console for CORS errors
- Verify frontend is accessing correct endpoints

#### 4. **Memory Issues**
- Monitor memory usage: `ps aux | grep python`
- Reduce `MAX_CONVERSATION_HISTORY` if needed
- Clear old sessions: `DELETE /reset-conversation/{session_id}`

### Debug Commands
```bash
# Check all services status
curl http://localhost:8001/health
curl http://localhost:8000/health

# Test RAG functionality
curl http://localhost:8001/dev/test-rag

# Get system statistics
curl http://localhost:8001/session-stats

# Monitor logs in real-time
tail -f backend/rag_server.log
tail -f frontend/frontend_server.log
```

---

## 🔐 Security Considerations

### Current Security Measures
- **No Authentication**: Development system - implement authentication for production
- **CORS Enabled**: Currently allows all origins - restrict for production
- **Local Network Only**: System binds to localhost by default

### Production Security Recommendations
1. **Add Authentication**: Implement JWT or session-based authentication
2. **Enable HTTPS**: Use SSL/TLS for all communications
3. **Restrict CORS**: Limit allowed origins to known domains
4. **Input Validation**: Enhanced validation for all user inputs
5. **Rate Limiting**: Implement rate limiting to prevent abuse
6. **Audit Logging**: Log all medical interactions for compliance

---

## 📈 Performance Optimization

### Current Performance
- **Average Response Time**: 1-2 seconds
- **Memory Usage**: ~100MB per active session
- **Concurrent Sessions**: Supports 50+ concurrent users
- **Entity Extraction**: <100ms for typical inputs

### Optimization Strategies
1. **Caching**: Implement Redis for conversation state caching
2. **Async Processing**: Use async/await for all I/O operations
3. **Database**: Move from in-memory to persistent storage for large deployments
4. **Load Balancing**: Use multiple RAG server instances for high traffic
5. **GPU Acceleration**: Consider GPU-based entity recognition for large scale

---

## 🚀 Future Enhancements

### Planned Features
1. **Vector Database Integration**: Add semantic search for medical knowledge
2. **Multi-language Support**: Multi-language support for healthcare
3. **Integration APIs**: FHIR, HL7 integration for EMR systems
4. **Advanced Analytics**: Conversation analytics and insights dashboard
5. **Mobile App**: React Native app for mobile healthcare access

### Research Areas
- **Advanced NLP**: Integration with medical NLP models (BioBERT, ClinicalBERT)
- **Knowledge Graphs**: Medical knowledge graph integration
- **Federated Learning**: Privacy-preserving model updates
- **Multimodal AI**: Support for medical image analysis

---

## 📞 Support & Contributing

### Getting Help
- **Documentation**: Check `docs/` directory for detailed guides
- **Issues**: Report issues via GitHub or internal ticketing system
- **Support**: Contact Healthcare technical team

### Contributing
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

---

## 📄 License & Compliance

### License
Proprietary software for Healthcare Organization. All rights reserved.

### Medical Compliance
- **HIPAA Compliance**: Implement additional security measures for PHI
- **Data Privacy**: Follow local data protection regulations
- **Medical Disclaimers**: Add appropriate medical disclaimers for production use
- **Audit Trail**: Maintain comprehensive audit logs for regulatory compliance

---

## 📊 Version History

### v1.0.0 (Current)
- ✅ Core RAG engine implementation
- ✅ Medical entity recognition and symptom extraction
- ✅ Conversation memory and context building
- ✅ Enhanced frontend with real-time visualization
- ✅ Comprehensive API and documentation

### Planned v1.1.0
- 🔄 Vector database integration
- 🔄 Enhanced medical knowledge base
- 🔄 Advanced conversation analytics
- 🔄 Multi-language support preparation

---

**🏥 Medical RAG Chatbot System**  
*Transforming Medical Conversations with AI Intelligence*

**Healthcare Organization** | **Advanced Conversational AI** | **Built with ❤️ for Better Healthcare**
