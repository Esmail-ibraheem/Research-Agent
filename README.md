# Research-Agent
An agent powered by OpenAI that searches and explains papers from arXiv and summarizes each paper with the specific topic the user asks 

## How to use

- install your KB
```
python Research-agent.py digest --query "transformers diffusion" --days 730 --max 100 --top 25 --min-score 0.0 --out digest.md

```

- Launch chat with streaming at 80 chars/sec:
```
python Research-agent.py chat --k 5 --cps 80

```

- in chat
```
PRC> How do diffusion models relate to transformers in continuous-time settings?
# → you’ll see the answer “typing out” live

PRC> :speed 120
PRC> :sum 12

```

- **output is inside the file digest.md**
- **and the chatting system in the terminal: where the knowledge base is the papers searched from the beginning**
