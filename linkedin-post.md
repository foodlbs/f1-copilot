# LinkedIn Post: F1 RAG Microservices Platform

---

## Post Content

**Excited to share my latest project: An AI-powered Formula 1 Knowledge Assistant built with RAG (Retrieval-Augmented Generation)!**

After months of development, I've built a production-ready microservices platform that combines:

**What it does:**
- Natural language Q&A about F1 history (1950-present)
- Real-time race strategy predictions based on historical patterns
- Fantasy F1 driver recommendations with value metrics
- Lap-by-lap race simulation with tire physics modeling

**Tech Stack:**
- **Frontend:** Next.js 14 + TypeScript + Tailwind CSS
- **Backend:** FastAPI + LangChain for RAG orchestration
- **Vector DB:** Pinecone with OpenAI text-embedding-3-large
- **LLM:** GPT-4o-mini (with Ollama fallback for local deployment)
- **Infrastructure:** Docker + Kong API Gateway + Redis

**Key Technical Highlights:**

1. **Multi-Strategy RAG System** - Implemented 4 retrieval strategies (Similarity, MMR, Multi-Query, Compression) to optimize for different query types

2. **Intelligent Web Search Fallback** - When the vector DB lacks information, the system automatically searches the web, ingests results into Pinecone, and regenerates answers

3. **Real-Time Streaming** - Token-by-token SSE streaming for responsive UX

4. **Multi-Source Data Pipeline** - Unified ingestion from FastF1 API, OpenF1, Ergast, and CSV archives

The most interesting challenge? Building a retrieval system that understands F1-specific context - pit stop strategies, tire compounds, circuit characteristics - while maintaining general conversational ability.

**Architecture diagram and code snippets in the comments!**

#AI #MachineLearning #RAG #LangChain #FastAPI #NextJS #Formula1 #VectorDatabase #Pinecone #OpenAI #Microservices #Python #TypeScript

---

## Architecture Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           F1 RAG Platform                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Next.js Frontend (:3000)                      │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │    │
│  │  │ F1-Themed   │  │  Strategy   │  │   Real-Time Streaming   │  │    │
│  │  │  Chat UI    │  │  Selector   │  │   Token Display         │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │    │
│  └──────────────────────────┬──────────────────────────────────────┘    │
│                             │                                            │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │               Kong API Gateway (:8000)                           │    │
│  │  ┌──────────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────┐  │    │
│  │  │ Rate Limiting│ │   CORS   │ │Health Check│ │  Prometheus  │  │    │
│  │  │  10/s 100/m  │ │ Handling │ │  Routing   │ │   Metrics    │  │    │
│  │  └──────────────┘ └──────────┘ └────────────┘ └──────────────┘  │    │
│  └──────────────────────────┬──────────────────────────────────────┘    │
│                             │                                            │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              FastAPI RAG Service (:8001)                         │    │
│  │                                                                   │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │              Retrieval Strategies                        │    │    │
│  │  │  ┌──────────┐ ┌──────┐ ┌─────────────┐ ┌─────────────┐  │    │    │
│  │  │  │Similarity│ │ MMR  │ │ Multi-Query │ │ Compression │  │    │    │
│  │  │  └──────────┘ └──────┘ └─────────────┘ └─────────────┘  │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  │                                                                   │    │
│  │  ┌──────────────────┐  ┌────────────────────────────────────┐   │    │
│  │  │ LangChain RAG    │  │    Web Search Fallback             │   │    │
│  │  │ Conversational   │  │  ┌────────────┐ ┌───────────────┐  │   │    │
│  │  │ Retrieval Chain  │  │  │ DuckDuckGo │→│ Auto-Ingest   │  │   │    │
│  │  └──────────────────┘  │  │   Search   │ │ to Pinecone   │  │   │    │
│  │                        │  └────────────┘ └───────────────┘  │   │    │
│  │                        └────────────────────────────────────┘   │    │
│  └───────┬───────────────────────────────┬─────────────────────────┘    │
│          │                               │                               │
│          ▼                               ▼                               │
│  ┌───────────────────┐          ┌───────────────────┐                   │
│  │  Pinecone Cloud   │          │   Redis (:6379)   │                   │
│  │   Vector DB       │          │                   │                   │
│  │ ┌───────────────┐ │          │ ┌───────────────┐ │                   │
│  │ │ F1 Embeddings │ │          │ │ Chat History  │ │                   │
│  │ │ 3072-dim      │ │          │ │ Session Mgmt  │ │                   │
│  │ │ text-emb-3    │ │          │ │ Web Doc Cache │ │                   │
│  │ └───────────────┘ │          │ └───────────────┘ │                   │
│  └───────────────────┘          └───────────────────┘                   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Data Ingestion Pipeline                       │    │
│  │                                                                   │    │
│  │   CSV Archives    FastF1 API     OpenF1 API     Ergast API       │    │
│  │   (1950-2017)     (2018-2024)    (2023-2024)    (Fallback)       │    │
│  │       │               │              │              │             │    │
│  │       └───────────────┴──────────────┴──────────────┘             │    │
│  │                           │                                       │    │
│  │                           ▼                                       │    │
│  │              ┌─────────────────────────┐                          │    │
│  │              │  Unified Data Ingestion │                          │    │
│  │              │  - Merge & Deduplicate  │                          │    │
│  │              │  - Fantasy Metrics      │                          │    │
│  │              │  - Generate Embeddings  │                          │    │
│  │              └─────────────────────────┘                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Code Snippets for Comments

### Snippet 1: Multi-Strategy RAG Retriever

```python
def get_retriever(strategy: str, k: int = 5):
    """Select retrieval strategy based on query type"""

    if strategy == "similarity":
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    elif strategy == "mmr":  # Maximal Marginal Relevance
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": k * 3,
                "lambda_mult": 0.5  # Balance relevance & diversity
            }
        )

    elif strategy == "multi_query":
        # Expands single query into multiple perspectives
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm_no_stream,
            include_original=True
        )

    elif strategy == "compression":
        # LLM extracts only relevant content from retrieved docs
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": k * 2})
        compressor = LLMChainExtractor.from_llm(llm_no_stream)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
```

### Snippet 2: Intelligent Web Search Fallback

```python
async def ingest_search_results(query: str, search_results: List[dict]) -> List[Document]:
    """Auto-ingest web results into vector DB for future queries"""
    vectors_to_upsert = []

    for result in search_results:
        content = f"{result.get('title', '')}\n\n{result.get('body', '')}"
        doc_id = generate_doc_id(content)

        # Skip if already cached (7-day dedup window)
        cache_key = f"web_doc:{doc_id}"
        if redis_client.exists(cache_key):
            continue

        metadata = {
            "source": "web_search",
            "url": result.get("href", ""),
            "title": result.get("title", ""),
            "query": query,
            "ingested_at": datetime.now().isoformat(),
            "text": content
        }

        # Generate embedding and batch for upsert
        embedding = embeddings.embed_query(content)
        vectors_to_upsert.append({
            "id": doc_id,
            "values": embedding,
            "metadata": metadata
        })

        # Mark as ingested in Redis (TTL: 7 days)
        redis_client.setex(cache_key, 604800, "1")

    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)

    return documents
```

### Snippet 3: Real-Time Streaming Handler (TypeScript)

```typescript
const sendStreamingMessage = async (query: string) => {
  const response = await fetch('http://localhost:8000/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      session_id: sessionId,
      retrieval_strategy: strategy,
      stream: true,
      top_k: 5
    }),
  });

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));

        if (data.type === 'token') {
          // Update UI with streaming token
          assistantMessage += data.token;
        } else if (data.type === 'web_search_start') {
          // Show "Searching the web..." indicator
        } else if (data.type === 'sources') {
          // Display retrieved document sources
        }
      }
    }
  }
};
```

### Snippet 4: F1 Race Strategy Predictor

```python
@dataclass
class StrategyRecommendation:
    """AI-generated race strategy"""
    strategy_type: str  # '1-stop', '2-stop', '3-stop'
    stints: List[Dict[str, Any]]
    confidence: float
    reasoning: str
    similar_races: List[Dict[str, Any]]
    expected_position: Optional[int] = None

class F1StrategyPredictor:
    def predict_optimal_strategy(
        self,
        session_data: SessionData,
        driver_position: Optional[int] = None,
        num_similar_races: int = 10
    ) -> StrategyRecommendation:
        """
        Predict optimal race strategy using RAG

        Queries vector DB for historically similar races:
        - Same circuit characteristics
        - Similar weather conditions
        - Comparable tire degradation patterns

        Returns strategy with confidence score and reasoning
        """
        # Find similar historical races via semantic search
        similar_races = self.vector_db.search(
            query=f"Race strategy {session_data.circuit} {session_data.weather}",
            top_k=num_similar_races,
            filter={"data_type": "race_strategy"}
        )

        # Analyze patterns and generate recommendation
        return self._generate_recommendation(session_data, similar_races)
```

---

## Suggested Comment Strategy

**Comment 1:** Post the architecture diagram

**Comment 2:** "Here's how the multi-strategy RAG system works..." + Snippet 1

**Comment 3:** "The web search fallback is my favorite feature..." + Snippet 2

**Comment 4:** "Real-time streaming makes the UX feel responsive..." + Snippet 3

**Comment 5:** "The strategy predictor uses RAG to find historically similar races..." + Snippet 4

---

## Hashtag Recommendations

Primary: #AI #MachineLearning #RAG #LangChain #GenAI
Technical: #FastAPI #NextJS #Python #TypeScript #Microservices
Domain: #Formula1 #F1 #Motorsport
Infrastructure: #VectorDatabase #Pinecone #Docker #Kubernetes

---

## Image Suggestions for LinkedIn

1. **Architecture diagram** (use the ASCII above or create in Figma/draw.io)
2. **Screenshot of the chat interface** showing an F1 query and response
3. **Code editor screenshot** with the RAG strategy selector highlighted
4. **Terminal showing Docker compose up** with all services healthy
