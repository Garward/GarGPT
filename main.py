import discord
from discord.ext import commands
import openai
import sqlite3
import os
import requests

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)
bot_name = "GarGPT"

# Load keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ALLOWED_ROLES = os.getenv("ALLOWED_ROLES", "").split(",")  # Comma-separated list
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
openai.api_key = OPENAI_API_KEY

# SQLite setup for caching messages and settings
conn = sqlite3.connect("message_cache.db")
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS messages (channel TEXT, author TEXT, content TEXT, timestamp TEXT)")
c.execute("CREATE TABLE IF NOT EXISTS personality (guild_id TEXT, name TEXT, system_prompt TEXT, PRIMARY KEY (guild_id, name))")
c.execute("CREATE TABLE IF NOT EXISTS personality_active (guild_id TEXT PRIMARY KEY, active_name TEXT)")
conn.commit()

def is_allowed(ctx):
    if ctx.guild is None:
        return True  # Allow in DMs
    role_names = [role.name for role in ctx.author.roles]
    return any(allowed.strip() in role_names for allowed in ALLOWED_ROLES if allowed.strip())

def get_personality_prompt(guild_id):
    c.execute("SELECT active_name FROM personality_active WHERE guild_id = ?", (str(guild_id),))
    row = c.fetchone()
    active_name = row[0] if row else None
    if not active_name:
        return "You are GarGPT, a helpful assistant with a bit of sass and a love for the word 'Gar'."
    c.execute("SELECT system_prompt FROM personality WHERE guild_id = ? AND name = ?", (str(guild_id), active_name))
    row = c.fetchone()
    return row[0] if row else "You are GarGPT, a helpful assistant with a bit of sass and a love for the word 'Gar'."

@bot.event
async def on_ready():
    print(f"{bot_name} connected as {bot.user}")

@bot.event
async def on_message(message):
    if message.author.bot:
        return
    # Cache messages
    c.execute("INSERT INTO messages VALUES (?, ?, ?, ?)",
              (str(message.channel.id), str(message.author), message.content, str(message.created_at)))
    conn.commit()
    await bot.process_commands(message)

@bot.command()
async def web(ctx, *, query):
    """Search the web and get summarized results using GPT-4o"""
    if not is_allowed(ctx):
        await ctx.reply("You don't have permission to use this command.")
        return
    headers = {"X-Subscription-Token": BRAVE_API_KEY}
    params = {"q": query, "count": 3}
    r = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
    if r.status_code != 200:
        await ctx.reply("Web search failed. Check your Brave API key or try again later.")
        return
    results = r.json().get("web", {}).get("results", [])
    if not results:
        await ctx.reply("No web results found.")
        return
    snippets = "\n\n".join([f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['description']}" for r in results])
    system_prompt = get_personality_prompt(ctx.guild.id if ctx.guild else "dm")
    await ctx.trigger_typing()
    summary = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Summarize these search results and answer the query: {query}\n\n{snippets}"}
        ],
        max_tokens=400
    )
    await ctx.reply(summary.choices[0].message.content[:2000])

@bot.command()
async def status(ctx):
    """Show current bot status and runtime info"""
    import time
    import datetime
    uptime = time.time() - bot.start_time
    uptime_str = str(datetime.timedelta(seconds=int(uptime)))

    guild_id = str(ctx.guild.id) if ctx.guild else "dm"
    c.execute("SELECT active_name FROM personality_active WHERE guild_id = ?", (guild_id,))
    row = c.fetchone()
    personality = row[0] if row else "default"

    latency_ms = round(bot.latency * 1000)

    status_msg = f"""
**GarGPT Status**
Model: GPT-4o (May 2024)
Active Personality: `{personality}`
Uptime: `{uptime_str}`
Latency: `{latency_ms} ms`
"""
    await ctx.reply(status_msg.strip())

@bot.command()
async def deletepersonality(ctx, name: str):
    """Delete a saved personality"""
    if not is_allowed(ctx):
        await ctx.reply("You don't have permission to delete personalities.")
        return
    guild_id = str(ctx.guild.id)
    # Check if it's the active one
    c.execute("SELECT active_name FROM personality_active WHERE guild_id = ?", (guild_id,))
    row = c.fetchone()
    if row and row[0] == name:
        c.execute("DELETE FROM personality_active WHERE guild_id = ?", (guild_id,))
    c.execute("DELETE FROM personality WHERE guild_id = ? AND name = ?", (guild_id, name))
    conn.commit()
    await ctx.reply(f"Deleted personality '{name}' for this server.")

# (Other commands remain unchanged)

import time
bot.start_time = time.time()

if __name__ == "__main__":
    print("Launching bot...")
    bot.run(DISCORD_BOT_TOKEN)
