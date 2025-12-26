# ⚡ Barq (برق) | The Saudi Search Engine

> **"Building an Intelligent Search Engine for Saudi Arabia 🇸🇦."**

---

## 📖 The Story

### 1. The "Why" (Problem Statement)

> "Saudi Arabia's government portals contain valuable information, but finding specific details can be challenging for citizens. Traditional search often returns entire documents when users just need key answers. I wanted to build something that could _understand_ Arabic content at scale and provide precise, summarized answers."

### 2. The "Journey" (The Solution)

This project represents a deep technical exploration where I built a complete pipeline from scratch:

-   **🧠 The Brain (QFS Nano Model):** Instead of using a massive, expensive API, I fine-tuned a **0.6B parameter model** specifically for _Query-Focused Summarization (QFS)_. It is trained to read large documents and extract only the relevant answer.
-   **🧭 The Navigator (Search Engine):** A hybrid search orchestrator. I built a factory system that tests multiple algorithms (BM25, Vector, and RRF Fusion) to ensure the "Brain" gets the right documents to read.
-   **🕷️ The Gatherer (Crawler):** A custom Python bot that indexes `my.gov.sa`, cleaning and structuring regulatory text into our searchable knowledge base.
-   **⚡ The App (Inference):** A high-performance FastAPI server running on CPU.

### 3. The "Innovation"

> "What makes this project unique is its focus on speed and resource efficiency. By fine-tuning a compact 0.6B parameter model, the system performs specialized tasks effectively while running entirely on CPU in $5 VPS setup. This approach makes the technology more accessible and sustainable, especially for people with limited computational resources, Furthermore, the search engine is lightning-fast, responding in less than 50ms for most queries while delivering precise, concise answers."

---

## 🏗️ Architecture

```
┌─────────┐  crawl   ┌──────────────┐  index   ┌─────────────┐
│ mygov.sa│ ────────▶│   Crawler    │ ────────▶│ Search Index│
└─────────┘          └──────────────┘          └─────────────┘
                                                      │
┌─────────┐          ┌──────────────┐          ┌─────▼────┐
│  Query  │ ────────▶│   FastAPI    │ ────────▶│ Search   │
│         │          │    Server    │          │ Algorithm│
└─────────┘          └──────────────┘          └─────┬────┘
                                                      │
                                               ┌─────▼────┐
                                               │Summarizer│
                                               │ (QFS)    │
                                               └─────┬────┘
                                                     │
                                               ┌─────▼────┐
                                               │ Response │
                                               │ (Summary)│
                                               └──────────┘
```

---

## 🔮 Roadmap

-   [x] **Phase 1: The Foundation** – Search Factory, Nano Model, and Basic Crawler.
-   [ ] **Phase 2: The Expansion** – Indexing more official sites.
