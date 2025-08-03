import discord
from discord.ext import commands
import openai
import sqlite3
import os
import requests
import asyncio
from discord import app_commands

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)
bot_name = "GarGPT"

tree = bot.tree

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

def is_allowed(interaction):
    if interaction.guild is None:
        return True  # Allow in DMs
    role_names = [role.name for role in interaction.user.roles]
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
    await tree.sync()
    print(f"{bot_name} connected as {bot.user}")

@bot.event
async def on_message(message):
    if message.author.bot:
        return
    # Cache messages
    c.execute("INSERT INTO messages VALUES (?, ?, ?, ?)",
              (str(message.channel.id), str(message.author), message.content, str(message.created_at)))
    conn.commit()

@tree.command(name="ask", description="Ask GPT-4o a question")
async def ask(interaction: discord.Interaction, prompt: str):
    if not is_allowed(interaction):
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return
    await interaction.response.defer()
    system_prompt = get_personality_prompt(interaction.guild.id if interaction.guild else "dm")
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    await interaction.followup.send(response.choices[0].message.content[:2000])

@tree.command(name="web", description="Search the web and summarize with GPT-4o")
async def web(interaction: discord.Interaction, query: str):
    if not is_allowed(interaction):
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return
    await interaction.response.defer()
    headers = {"X-Subscription-Token": BRAVE_API_KEY}
    params = {"q": query, "count": 3}
    r = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
    if r.status_code != 200:
        await interaction.followup.send("Web search failed. Check your Brave API key or try again later.")
        return
    results = r.json().get("web", {}).get("results", [])
    if not results:
        await interaction.followup.send("No web results found.")
        return
    snippets = "\n\n".join([f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['description']}" for r in results])
    system_prompt = get_personality_prompt(interaction.guild.id if interaction.guild else "dm")
    summary = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Summarize these search results and answer the query: {query}\n\n{snippets}"}],
        max_tokens=400
    )
    await interaction.followup.send(summary.choices[0].message.content[:2000])

@tree.command(name="status", description="Show current bot status and runtime info")
async def status(interaction: discord.Interaction):
    import time
    import datetime
    uptime = time.time() - bot.start_time
    uptime_str = str(datetime.timedelta(seconds=int(uptime)))
    guild_id = str(interaction.guild.id) if interaction.guild else "dm"
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
    await interaction.response.send_message(status_msg.strip())

@tree.command(name="deletepersonality", description="Delete a saved personality")
@app_commands.describe(name="The personality name to delete")
async def deletepersonality(interaction: discord.Interaction, name: str):
    if not is_allowed(interaction):
        await interaction.response.send_message("You don't have permission to delete personalities.", ephemeral=True)
        return
    guild_id = str(interaction.guild.id)
    c.execute("SELECT active_name FROM personality_active WHERE guild_id = ?", (guild_id,))
    row = c.fetchone()
    if row and row[0] == name:
        c.execute("DELETE FROM personality_active WHERE guild_id = ?", (guild_id,))
    c.execute("DELETE FROM personality WHERE guild_id = ? AND name = ?", (guild_id, name))
    conn.commit()
    await interaction.response.send_message(f"Deleted personality '{name}' for this server.")

import time
bot.start_time = time.time()

if __name__ == "__main__":
    print("Launching bot...")
    bot.run(DISCORD_BOT_TOKEN)
