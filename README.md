# GarGPT 🤖

A lightweight GPT-4o-powered Discord bot with:

- 🧠 Custom personalities (saved per server)
- 🔍 Brave-powered web search
- 🔎 Message history search + summarization
- 📊 `/status` command for uptime, latency, and current personality

## Setup

1. Clone the repo
2. Create a `.env` file based on `.env.example`
3. Run:

```
pip install -r requirements.txt
python main.py
```

## Commands

- `/ask [prompt]` — Ask GPT a question
- `/setpersonality [name] [prompt]` — Save and use a new personality
- `/usepersonality [name]` — Switch to a saved personality
- `/listpersonalities` — Show all personalities
- `/deletepersonality [name]` — Delete a saved personality
- `/web [query]` — Search the web and summarize results
- `/search [text]` — Find messages in history
- `/summarize [limit]` — Summarize recent messages
- `/status` — See uptime, latency, and active personality
