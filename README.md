Perfect 🚀 Let me draft you a polished **README.md** for your **Research-Agent** project.
This version is professional, detailed, and GitHub-ready:

---

````markdown
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

## 📸 Demo (Optional)

*Add a GIF or screenshot here for extra clarity!*

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Fork the repo, open a PR, or suggest new features.

---

## 📜 License

MIT License © 2025 [Esmail Ibraheem](https://github.com/Esmail-ibraheem)


