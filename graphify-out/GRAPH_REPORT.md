# Medical RAG Chatbot - Knowledge Graph Report

## Graph Statistics
- **Nodes**: 37 (code components + services)
- **Edges**: 45 relationships
- **Communities**: 7 architectural clusters

## God Nodes (Core Hub Components)

These are the most connected nodes - central to the system architecture:

1. **FastAPI App** (15 connections)
   - Central to request handling and orchestration
2. **MedicalRAGEnrichmentEngine** (9 connections)
   - Central to request handling and orchestration
3. **ConversationMemory** (6 connections)
   - Central to request handling and orchestration
4. **RAG Processing Pipeline** (5 connections)
   - Central to request handling and orchestration
5. **SymptomExtractor** (4 connections)
   - Central to request handling and orchestration

## Surprising Connections (Architectural Bridges)

1. **Frontend Application** <-> **FastAPI App**
   - Cross-cutting connection bridging different concerns
2. **CORS Proxy** <-> **FastAPI App**
   - Cross-cutting connection bridging different concerns
3. **API Integration Tests** <-> **MedicalRAGEnrichmentEngine**
   - Cross-cutting connection bridging different concerns
4. **RAG Engine Tests** <-> **ConversationMemory**
   - Cross-cutting connection bridging different concerns
5. **FastAPI App** <-> **InputSanitizer**
   - Cross-cutting connection bridging different concerns

## Suggested Questions

Use the graph to answer:
1. What is the dependency chain from Frontend -> RAG Engine?
2. How does data flow from extraction to response generation?
3. Which components could be independently deployed/scaled?
