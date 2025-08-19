


# 🔬 Research-Agent

**Research-Agent** is a terminal-based AI companion that helps you **discover, summarize, and interact with research papers** from [arXiv](https://arxiv.org).  
It combines fast retrieval, scoring, and OpenAI-powered summarization with a live, streaming chat experience — all in a **single Python file**.

---

## ✨ Features
- 📚 **Fetch recent arXiv papers** by query and filter by date
- ⚖️ **Relevance scoring** to rank papers
- 📝 **Summarization**
  - OpenAI (`gpt-4o-mini` by default)
  - Extractive fallback if API unavailable
- 📑 **Digest mode** → generates `digest.md` + updates a local knowledge base (`kb.jsonl`)
- 💬 **Chat mode** → interactive Q&A with:
  - Streaming “typing” effect
  - Contextual answers using stored papers
  - Quick commands (`:list`, `:sum`, `:open`, `:speed`, `:exit`)
- 🗂️ **Knowledge persistence** — keeps track of what you’ve already read

---

## ⚡ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/Esmail-ibraheem/Research-Agent.git
cd Research-Agent
pip install requests
````

> 💡 No external DB or frameworks required. Just Python 3.8+ and `requests`.

---

## 🚀 Usage

### 🔎 Search Mode

Fetch and rank papers quickly:

```bash
python Research-agent.py search --query "graph neural networks drug discovery" --max 25 --days 14
```

---

### 📑 Digest Mode

Generate a Markdown digest with AI summaries:

```bash
python Research-agent.py digest \
  --query "transformers diffusion" \
  --days 365 --max 50 --top 10 --min-score 0.0 --out digest.md
```

✅ Produces `digest.md` and updates `kb.jsonl`.

```markdown
# Research Digest — Graph Neural Networks in Drug Discovery
_Generated: 2025-08-18T10:45:00+00:00_

---

## 1. Graph Neural Networks for Drug-Target Interaction Prediction
- **Score**: 0.842  
- **Authors**: Alice Zhang, Marco Rossi, Daniel Li  
- **Published**: 2025-07-28T00:00:00+00:00  
- **Categories**: cs.LG, q-bio.BM  
- **Link**: https://arxiv.org/abs/2507.12345  

**AI Summary**  
- **Problem**: Existing drug-target interaction models struggle with sparse biochemical data.  
- **Method**: Proposes a GNN with attention-based pooling for drug–protein binding prediction.  
- **Results**: Achieves state-of-the-art AUC (0.91) on multiple public datasets.  
- **Limitations**: Tested only on small molecules; generalization to peptides not validated.  

---

## 2. Multi-Scale Graph Transformers for Drug Discovery Pipelines
- **Score**: 0.791  
- **Authors**: Priya Gupta, Henrik Müller, John Smith  
- **Published**: 2025-06-15T00:00:00+00:00  
- **Categories**: cs.AI, chem-ph  
- **Link**: https://arxiv.org/abs/2506.09876  

**AI Summary**  
- **Problem**: Traditional GNNs capture local but not global chemical structure.  
- **Method**: Introduces hierarchical graph transformer layers with multi-scale attention.  
- **Results**: Outperforms baselines on QM9 and MoleculeNet benchmarks.  
- **Limitations**: Computational cost higher; requires GPUs for scalability.  

---

```

---

### 💬 Chat Mode

Interactive terminal chat:

```bash
python Research-agent.py chat --k 5 --cps 80
```

#### Example:

```
PRC> What are the latest uses of GNNs in drug discovery?
[typing… streaming answer]

PRC> :list diffusion
PRC> :sum 2
PRC> :open 3
PRC> :speed 120
PRC> :exit
```

---

## ⚙️ CLI Options

| Command  | Description                       |
| -------- | --------------------------------- |
| `search` | Fetch and rank papers             |
| `digest` | Generate digest.md with summaries |
| `chat`   | Start interactive chat session    |
| `watch`  | Auto-refresh digest periodically  |

---

## 🛠 Configuration

* API key is embedded directly in the script (`OPENAI_API_KEY` variable).
* Default model: `gpt-4o-mini` (can be changed inside file).
* Knowledge base stored in `kb.jsonl`.
* Digests saved to `digest.md`.

---

## 📸 Demo

![Research-agent-1 py-Agent-Swarm-Visual-Studio-Code-2025-08-19-18-36-12](https://github.com/user-attachments/assets/14bc12af-9b1d-43ab-9bca-3242f43242ac)


---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Fork the repo, open a PR, or suggest new features.

---

## 📜 License

MIT License © 2025 [Esmail Ibraheem](https://github.com/Esmail-ibraheem)


