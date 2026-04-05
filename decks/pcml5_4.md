---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 5.4: RAG Systems

### Module 5: LLMs and Agents

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain Retrieval-Augmented Generation and why it beats pure LLMs
- Build vector embeddings and similarity search indices
- Use `RAGResearchAgent` for knowledge-grounded Q&A
- Design chunking and retrieval strategies for domain documents

---

## Recap: Lesson 5.3

- ReAct agents interleave reasoning with tool use
- Tools give agents access to databases, calculators, and APIs
- The Thought-Action-Observation loop grounds answers in real data
- Tool design requires clear descriptions and bounded scope

---

## The Knowledge Problem

```
LLM alone:
  Q: "What are the HDB cooling measures announced in Dec 2024?"
  A: "I don't have specific information about Dec 2024 policies..."
     → Training data cutoff. Hallucination risk.

RAG:
  1. Search your document collection for "cooling measures 2024"
  2. Retrieve relevant policy documents
  3. Feed retrieved text to the LLM as context
  4. LLM answers based on YOUR documents
  → Grounded, accurate, up-to-date.
```

---

## RAG Architecture

```
User question
    ↓
┌──────────┐     ┌─────────────────┐
│  Embed   │────→│  Vector Store   │
│  query   │     │  (your docs)    │
└──────────┘     └────────┬────────┘
                          │ Top-K similar chunks
                          ↓
                 ┌─────────────────┐
                 │  LLM generates  │
                 │  answer using   │
                 │  retrieved text │
                 └─────────────────┘
                          ↓
                    Grounded answer
```

---

## Step 1: Document Chunking

```python
# Raw documents → overlapping chunks
documents = [
    "HDB resale prices policy update Dec 2024...",
    "Cooling measures include additional buyer stamp duty...",
]

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 50:  # skip tiny trailing chunks
            chunks.append(chunk)
    return chunks

all_chunks = []
for doc in documents:
    all_chunks.extend(chunk_text(doc))
```

---

## Chunking Strategy

```
Too small (100 chars):
  "HDB resale prices"  → Lost context, incomplete meaning

Too large (5000 chars):
  Entire document       → Dilutes relevant section with noise

Sweet spot (300-800 chars):
  Complete paragraph    → Enough context, focused content
  with overlap          → No information lost at boundaries

  ┌──────────────────────────────┐
  │     Chunk 1          │       │
  │              ┌───────┤───────┤───────┐
  │              │overlap│ Chunk 2        │
  └──────────────┴───────┘───────┴───────┘
```

---

## Step 2: Embedding Chunks

```python
from kailash_kaizen import Embedder

embedder = Embedder()
embedder.configure(model="text-embedding-3-small")

# Convert text chunks to vectors
embeddings = embedder.embed(all_chunks)

# Each chunk becomes a vector of ~1536 dimensions
# Similar text → similar vectors → close in vector space
print(f"Chunks: {len(all_chunks)}")
print(f"Embedding dimension: {embeddings.shape[1]}")
```

---

## Step 3: Vector Store

```python
from kailash_kaizen import VectorStore

store = VectorStore()
store.configure(storage_path="./vector_store")

# Index the chunks
store.add(
    texts=all_chunks,
    embeddings=embeddings,
    metadata=[{"source": "policy_doc", "page": i} for i in range(len(all_chunks))],
)

# Search by similarity
results = store.search(
    query="What are the latest cooling measures?",
    top_k=5,
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.text[:100]}...")
```

---

## Step 4: RAGResearchAgent

```python
from kailash_kaizen import RAGResearchAgent, Signature

agent = RAGResearchAgent()
agent.configure(
    model="claude-sonnet",
    vector_store=store,
    top_k=5,
    temperature=0.0,
)

sig = Signature(
    input_fields={"question": "User's question about HDB policy"},
    output_fields={
        "answer": "Detailed answer based on retrieved documents",
        "sources": "List of source documents used",
        "confidence": "How well the sources support the answer (high/medium/low)",
    },
)

result = agent.execute(sig, inputs={
    "question": "What cooling measures affect HDB resale prices?"
})
```

---

## Source Attribution

```python
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"\nSources:")
for source in result.sources:
    print(f"  - {source}")
```

```
Answer: The key cooling measures affecting HDB resale prices include:
  1. Additional Buyer's Stamp Duty (ABSD) of 20% for...
  2. Loan-to-value limits capped at 75% for...
  3. Wait-out period of 15 months for...

Confidence: high

Sources:
  - policy_doc (page 3): "ABSD rates were revised..."
  - policy_doc (page 7): "LTV limits for HDB loans..."
```

---

## RAG for ML Documentation

```python
# Build a knowledge base from Kailash SDK docs
import os

doc_chunks = []
for doc_file in os.listdir("./kailash_docs/"):
    with open(f"./kailash_docs/{doc_file}") as f:
        text = f.read()
    doc_chunks.extend([
        {"text": chunk, "source": doc_file}
        for chunk in chunk_text(text)
    ])

# Index SDK documentation
store.add(
    texts=[c["text"] for c in doc_chunks],
    metadata=[{"source": c["source"]} for c in doc_chunks],
)

# Now agents can answer "How do I use TrainingPipeline?"
# based on actual SDK documentation
```

---

## Retrieval Quality Matters

```
Good retrieval → Good answer
Bad retrieval  → Hallucinated answer citing wrong context

Improve retrieval:
  1. Better chunking (respect paragraph boundaries)
  2. Hybrid search (keyword + semantic)
  3. Reranking (score top results more carefully)
  4. Query expansion (rephrase the question)
  5. Metadata filtering (limit to relevant sources)
```

---

## Hybrid Search

```python
agent.configure(
    model="claude-sonnet",
    vector_store=store,

    # Combine semantic and keyword search
    search_strategy="hybrid",
    semantic_weight=0.7,
    keyword_weight=0.3,

    top_k=5,
)
```

Hybrid search catches both semantically similar and keyword-matching documents.

---

## Exercise Preview

**Exercise 5.4: HDB Policy Knowledge Base**

You will:

1. Chunk and embed HDB policy documents into a vector store
2. Build a `RAGResearchAgent` for policy Q&A
3. Evaluate retrieval quality with precision and recall
4. Compare RAG answers to pure LLM answers

Scaffolding level: **Light (~30% code provided)**

---

## Common Pitfalls

| Mistake                                 | Fix                                    |
| --------------------------------------- | -------------------------------------- |
| Chunks too large or too small           | 300-800 characters with overlap        |
| No source attribution                   | Always return sources with answers     |
| Trusting RAG without checking retrieval | Inspect retrieved chunks for relevance |
| Single search strategy                  | Use hybrid (semantic + keyword)        |
| Stale vector store                      | Re-index when documents change         |

---

## Summary

- RAG retrieves relevant documents before generating answers
- Chunking, embedding, and vector search form the retrieval pipeline
- `RAGResearchAgent` combines retrieval with LLM generation
- Source attribution makes answers verifiable
- Retrieval quality is the bottleneck: hybrid search and good chunking are critical

---

## Next Lesson

**Lesson 5.5: MCP Servers**

We will learn:

- Model Context Protocol for tool registration
- Building MCP servers with transports
- Connecting agents to standardised tool interfaces
