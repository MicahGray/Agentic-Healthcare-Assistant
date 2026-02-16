# 🏥 HealthAgent AI – Agentic Healthcare Assistant for Medical Task Automation

> Capstone project for the **Purdue Generative AI Specialist** program.

HealthAgent AI is a multi-agent GenAI system that automates healthcare administration tasks including appointment scheduling, patient record management, medical history retrieval (RAG), and disease information search.

---

## ✨ Features

| Category | Details |
|----------|---------|
| **Multi-Agent System** | 5 agents: Planner, Appointment, Records, History (RAG), Disease Search |
| **LangGraph Workflow** | Conditional routing based on intent classification |
| **RAG Pipeline** | FAISS vector store over PDF medical reports + patient DB |
| **LLM Integration** | GPT-4o-mini with LCEL chain prompts |
| **Database** | SQLite with patients, doctors, and appointments tables |
| **Model Evaluation** | 10 test cases with LLM-based QA grading |
| **Streamlit UI** | 6-tab dashboard: Chat, Patients, Appointments, History, Evaluation, Logs |

---

## 🧠 Agent Architecture

```
Patient Query
     ↓
🧠 Planner Agent (Intent Classification)
     ↓
┌────────┼────────┼────────┐
↓        ↓        ↓        ↓
📅       📋       🔍       🌐
Book     Update   RAG      Disease
Appt     Record   History  Search
↓        ↓        ↓        ↓
└────────┴────────┴────────┘
     ↓
Response to Patient
```

---

## 🛠️ Tech Stack

- **Multi-Agent Framework:** LangGraph
- **LLM:** OpenAI GPT-4o-mini via LangChain (LCEL)
- **RAG:** FAISS + OpenAI Embeddings (text-embedding-3-small)
- **Database:** SQLite (auto-initialized with seed data)
- **UI:** Streamlit (dark-themed, 6-tab interface)

---

## 📁 Project Structure

```
├── Agentic_Healthcare_Assistant.py        # Main application
├── requirements.txt                        # Dependencies
├── Reference Materials/                    # PDF medical reports for RAG
│   └── Agentic Healthcare Assistant.../
│       ├── sample_report_anjali.pdf
│       ├── sample_report_david.pdf
│       ├── sample_report_ramesh.pdf
│       └── records.xlsx
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key

### Local Setup

```bash
git clone https://github.com/MicahGray/Agentic-Healthcare-Assistant.git
cd Agentic-Healthcare-Assistant
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
streamlit run Agentic_Healthcare_Assistant.py
```

### Streamlit Community Cloud

Add your `OPENAI_API_KEY` in **Settings → Secrets**:
```toml
OPENAI_API_KEY = "sk-proj-..."
```

---

## 💬 Sample Use Cases

| Input | Agent Path | Behavior |
|-------|------------|----------|
| "Book a nephrologist for Ramesh Kulkarni" | Planner → Appointment | Finds available nephrologist, books slot |
| "Summarize David Thompson's medical history" | Planner → History (RAG) | Retrieves from FAISS, generates summary |
| "Update Anjali's record: moved to Mumbai" | Planner → Records | Updates patient address in SQLite |
| "Tell me about chronic kidney disease treatments" | Planner → Disease Search | Provides CKD overview with disclaimer |

---

## 📑 Capstone Requirements Mapping

| Step | Requirement | Implementation |
|------|-------------|----------------|
| 1 | Agent Planning & Goal Decomposition | Planner agent with 4-way intent classification |
| 2 | Tool & Memory Setup | SQLite DB, FAISS vector store, session memory |
| 3 | Prompt Engineering & Task Chaining | LCEL chains with patient context |
| 4 | Agent Execution Flow | LangGraph conditional routing (5 nodes) |
| 6 | Model Evaluation | 10 test cases, LLM QA grading, routing metrics |
| 7 | Data Visualization & UI | 6-tab Streamlit dashboard |
| 8 | Memory & Logs Interface | Agent traces, RAG chunks, tool usage logs |

---

## 📜 License

Created as part of the Purdue University Generative AI Specialist program.
