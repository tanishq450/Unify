<div align="center">
  <img src="unify_logo_1777839039166.png" width="200" alt="Unify Logo">
  <h1>🏦 Unify — The AI Financial Truth Engine</h1>
  <p><i>Ensuring 100% accuracy in financial document analysis through specialized verification modes.</i></p>

  [![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Security: FinGround](https://img.shields.io/badge/Security-FinGround-green.svg)](https://github.com/tanishq450/Unify)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
</div>

---

## 🌟 Why Unify?
Most standard AI models can "hallucinate"—confidently stating incorrect numbers or dates. In finance, a single wrong digit can be catastrophic. 

**Unify solves this through:**
*   🔍 **Human-Like Reading**: Distinguishes between general queries, complex tables, and corporate relationships.
*   🧠 **Multi-Expert Search**: Utilizes three specialized "brains" to extract data from messy PDFs and complex structures.
*   🛡️ **Atomic Verification**: An automated Auditor breaks down answers and cross-references every single data point against the source.

---

## 🎬 Demo Video
Experience Unify in action: **[Watch the Demo Video](https://drive.google.com/file/d/17pUyqtszgPQlldslHN7WFAUKSjljfBPP/view?usp=sharing)**

---

## 🧭 System Architecture
Imagine Unify as a high-end Research Team working in concert:

1.  **The Receptionist (Intent Classifier)**: Routes your question to the correct expert.
2.  **The Experts (Search Engines)**: 
    *   **The Table Expert**: Specialized in PDF grids and spreadsheets.
    *   **The Librarian**: Expert at semantic search and document retrieval.
    *   **The Relationship Expert**: Maps connections between entities (CEO histories, corporate ties).
3.  **The Drafter (LLM)**: Synthesizes the expert findings into a readable answer.
4.  **The Auditor (Hallucination Guardrail)**: The final gatekeeper. Verifies every number, name, and date before release.

### 📊 Process Flow
```mermaid
graph LR
    User([Your Question]) --> Receptionist{Analyze Intent}
    
    subgraph Experts [The Expert Searchers]
    Receptionist -->|Table Data| TableExpert[Table Expert]
    Receptionist -->|General Info| Librarian[General Librarian]
    Receptionist -->|Relationships| NetworkExpert[Network Expert]
    end
    
    Experts --> Drafter[Draft Answer]
    Drafter --> Auditor{Fact-Check Everything}
    
    Auditor -->|Verified| FinalResult([Safe Final Answer])
    Auditor -->|Mistake Found!| Drafter
```

---

## 🚀 Getting Started

### 1️⃣ Installation
```bash
# Clone the repository
git clone https://github.com/tanishq450/Unify.git
cd Unify

# Set up environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Configuration
Create a `.env` file in the root directory and add your API keys (see `.env.example` for reference).
```bash
cp .env.example .env
# Edit .env with your LLM provider and database keys
```

### 3️⃣ Usage Modes
| Mode | Command | Description |
| :--- | :--- | :--- |
| **Web Dashboard** | `python3 api.py` | Launches the interactive web interface. |
| **Interactive Chat** | `python3 main.py interactive [folder]` | Chat directly with your local financial documents. |
| **Safety Evaluation** | `python3 evaluation.py` | Runs the accuracy report and verification benchmarks. |

---

## 📁 Project Structure
```text
.
├── api.py                   # Web interface engine
├── main.py                  # Main control center
├── evaluation.py            # Testing and accuracy lab
├── implementations/         # Core logic (Search & Fact-checking)
│   ├── Graph_rag.py         # Relationship mapping
│   ├── hybrid_retriever.py  # Advanced search
│   └── hallucination_verifier.py # The Auditor
├── utils/                   # PDF processing and data cleaning
├── Model_loader/            # LLM initialization
└── qdrant/                  # Vector database integration
```

---

## 🛠️ Tech Stack
*   **Intelligence**: GPT-4o / Claude 3.5
*   **Memory**: [Qdrant](https://qdrant.tech/) (Vector DB) & [Neo4j](https://neo4j.com/) (Graph DB)
*   **Language**: Python 3.10+
*   **Framework**: FinGround Atomic Verification

---

## ⚠️ Current Limitations
*   **Graph Extraction**: Can fail on malformed structured outputs.
*   **Table Extraction**: Heavily dependent on the quality of the source PDF.
*   **Intent Classification**: Currently utilizes regex-based logic which may miss edge cases.
*   **Latency**: The multi-step verification process increases overall response time.
*   **Ingestion Speed**: Processing large financial reports significantly increases ingestion time.

---

## 🔮 Future Scope
*   🎯 **Specialized Classifier**: Fine-tuned finance-specific query classifier for better routing.
*   📉 **Reliability**: Enhanced graph extraction reliability and robustness.
*   🌐 **Real-time Data**: Integration with live financial market data feeds.
*   📊 **Comparative Analysis**: Multi-document comparative analysis across multiple reporting periods.
*   🖥️ **Dashboards**: Rich financial dashboard visualizations for data insights.
*   🤖 **Agentic Reporting**: Automated, agent-driven financial report generation.
*   💎 **Portfolio Intelligence**: Advanced intelligence layer for portfolio-wide insights.

---

<div align="center">
  <sub>Built for Financial Accuracy</sub>
</div>