import discord
from discord.ext import commands
import openai
from openai import OpenAI
import sqlite3
import os
import requests
import asyncio
import logging
import time
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from contextlib import contextmanager
from discord import app_commands

# Try to import PostgreSQL support
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None
    RealDictCursor = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gargpt.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment variable validation
def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key for GPT-4o access',
        'DISCORD_BOT_TOKEN': 'Discord bot token',
        'BRAVE_API_KEY': 'Brave Search API key for web search'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")
    
    if missing_vars:
        logger.error("Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"  - {var}")
        raise ValueError("Required environment variables are missing. Please set them before starting the bot.")
    
    # Check for PostgreSQL configuration
    postgres_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
    if postgres_url and POSTGRES_AVAILABLE:
        logger.info("PostgreSQL configuration detected and available.")
    elif postgres_url and not POSTGRES_AVAILABLE:
        logger.warning("PostgreSQL URL provided but psycopg2 not installed. Falling back to SQLite.")
    else:
        logger.info("No PostgreSQL configuration found. Using SQLite.")
    
    logger.info("All required environment variables are set.")

# Validate environment on startup
validate_environment()

# Load configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ALLOWED_ROLES = [role.strip() for role in os.getenv("ALLOWED_ROLES", "").split(",") if role.strip()]
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "2000"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
USE_POSTGRES = DATABASE_URL and POSTGRES_AVAILABLE

# Input validation and sanitization
def sanitize_input(text: str, max_length: int = MAX_PROMPT_LENGTH) -> str:
    """Sanitize and validate user input."""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Remove potentially harmful characters
    text = re.sub(r'[^\w\s\-.,!?;:()\[\]{}"\'/\\@#$%^&*+=<>|`~]', '', text)
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
        logger.warning(f"Input truncated to {max_length} characters")
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        raise ValueError("Input cannot be empty after sanitization")
    
    return text

# Rate limiting system
class RateLimiter:
    def __init__(self):
        self.user_requests: Dict[int, list] = {}
    
    def is_rate_limited(self, user_id: int) -> bool:
        """Check if user is rate limited."""
        now = time.time()
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        
        # Clean old requests
        self.user_requests[user_id] = [
            req_time for req_time in self.user_requests[user_id]
            if now - req_time < RATE_LIMIT_WINDOW
        ]
        
        # Check if user has exceeded rate limit
        if len(self.user_requests[user_id]) >= RATE_LIMIT_REQUESTS:
            return True
        
        # Add current request
        self.user_requests[user_id].append(now)
        return False
    
    def get_reset_time(self, user_id: int) -> int:
        """Get seconds until rate limit resets."""
        if user_id not in self.user_requests or not self.user_requests[user_id]:
            return 0
        
        oldest_request = min(self.user_requests[user_id])
        return max(0, int(RATE_LIMIT_WINDOW - (time.time() - oldest_request)))

rate_limiter = RateLimiter()

# Database connection management
class DatabaseManager:
    def __init__(self, db_path: str = "message_cache.db"):
        self.db_path = db_path
        self.use_postgres = USE_POSTGRES
        self.database_url = DATABASE_URL
        
        if self.use_postgres:
            logger.info("Using PostgreSQL database")
        else:
            logger.info("Using SQLite database")
        
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with PostgreSQL/SQLite fallback."""
        conn = None
        try:
            if self.use_postgres and POSTGRES_AVAILABLE:
                try:
                    conn = psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
                    conn.autocommit = False
                    yield conn
                except Exception as e:
                    logger.error(f"PostgreSQL connection failed: {e}. Falling back to SQLite.")
                    self.use_postgres = False
                    # Fallback to SQLite
                    if conn:
                        conn.close()
                    conn = sqlite3.connect(self.db_path, timeout=30.0)
                    conn.row_factory = sqlite3.Row
                    yield conn
            else:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.row_factory = sqlite3.Row
                yield conn
        except Exception as e:
            logger.error(f"Database error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def init_database(self):
        """Initialize database tables for PostgreSQL or SQLite."""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                
                if self.use_postgres:
                    # PostgreSQL table creation
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS messages (
                            id SERIAL PRIMARY KEY,
                            channel TEXT NOT NULL,
                            author TEXT NOT NULL,
                            content TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS personality (
                            guild_id TEXT NOT NULL,
                            name TEXT NOT NULL,
                            system_prompt TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (guild_id, name)
                        )
                    """)
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS personality_active (
                            guild_id TEXT PRIMARY KEY,
                            active_name TEXT NOT NULL,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Create indexes for better performance
                    c.execute("CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_personality_guild ON personality(guild_id)")
                    
                else:
                    # SQLite table creation
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS messages (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            channel TEXT NOT NULL,
                            author TEXT NOT NULL,
                            content TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS personality (
                            guild_id TEXT NOT NULL,
                            name TEXT NOT NULL,
                            system_prompt TEXT NOT NULL,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (guild_id, name)
                        )
                    """)
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS personality_active (
                            guild_id TEXT PRIMARY KEY,
                            active_name TEXT NOT NULL,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Create indexes for better performance
                    c.execute("CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_personality_guild ON personality(guild_id)")
                
                conn.commit()
                db_type = "PostgreSQL" if self.use_postgres else "SQLite"
                logger.info(f"{db_type} database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            if self.use_postgres:
                logger.info("Falling back to SQLite due to PostgreSQL initialization failure")
                self.use_postgres = False
                self.init_database()  # Retry with SQLite
            else:
                raise
    
    def execute_query(self, query: str, params: tuple = (), fetch_one: bool = False, fetch_all: bool = False):
        """Execute a database query with automatic PostgreSQL/SQLite syntax handling."""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                
                # Convert PostgreSQL-style parameters to SQLite if needed
                if not self.use_postgres and '%s' in query:
                    # Convert %s to ? for SQLite
                    query = query.replace('%s', '?')
                
                c.execute(query, params)
                
                if fetch_one:
                    result = c.fetchone()
                    return dict(result) if result else None
                elif fetch_all:
                    results = c.fetchall()
                    return [dict(row) for row in results] if results else []
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Database query error: {e}")
            if fetch_one or fetch_all:
                return None if fetch_one else []
            return False

db_manager = DatabaseManager()

# Response chunking utility
async def send_chunked_response(interaction: discord.Interaction, content: str, ephemeral: bool = False):
    """Send response in chunks if it exceeds Discord's character limit."""
    max_length = 2000
    
    if len(content) <= max_length:
        if interaction.response.is_done():
            await interaction.followup.send(content, ephemeral=ephemeral)
        else:
            await interaction.response.send_message(content, ephemeral=ephemeral)
        return
    
    # Split content into chunks
    chunks = []
    current_chunk = ""
    
    for line in content.split('\n'):
        if len(current_chunk) + len(line) + 1 <= max_length:
            current_chunk += line + '\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + '\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Send first chunk as response
    if not interaction.response.is_done():
        await interaction.response.send_message(chunks[0], ephemeral=ephemeral)
        chunks = chunks[1:]
    
    # Send remaining chunks as followups
    for chunk in chunks:
        await interaction.followup.send(chunk, ephemeral=ephemeral)

# OpenAI API wrapper with error handling
class OpenAIManager:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    async def create_completion(self, messages: list, max_tokens: int = 300) -> Optional[str]:
        """Create OpenAI completion with error handling."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except openai.RateLimitError:
            logger.warning("OpenAI rate limit exceeded")
            return "I'm currently experiencing high demand. Please try again in a few moments."
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return "I'm experiencing technical difficulties. Please try again later."
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI completion: {e}")
            return "An unexpected error occurred. Please try again."

openai_manager = OpenAIManager()

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)
bot_name = "GarGPT"
tree = bot.tree

def is_allowed(interaction: discord.Interaction) -> bool:
    """Check if user has permission to use bot commands."""
    if interaction.guild is None:
        return True  # Allow in DMs
    
    if not ALLOWED_ROLES:
        return True  # No role restrictions
    
    role_names = [role.name for role in interaction.user.roles]
    return any(allowed in role_names for allowed in ALLOWED_ROLES)

def get_personality_prompt(guild_id: str) -> str:
    """Get the active personality prompt for a guild."""
    try:
        # Get active personality name
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            if db_manager.use_postgres:
                c.execute("SELECT active_name FROM personality_active WHERE guild_id = %s", (guild_id,))
            else:
                c.execute("SELECT active_name FROM personality_active WHERE guild_id = ?", (guild_id,))
            row = c.fetchone()
            active_name = row[0] if row else None
            
            if not active_name:
                return "You are GarGPT, a helpful assistant with a bit of sass and a love for the word 'Gar'."
            
            # Get personality prompt
            if db_manager.use_postgres:
                c.execute("SELECT system_prompt FROM personality WHERE guild_id = %s AND name = %s", (guild_id, active_name))
            else:
                c.execute("SELECT system_prompt FROM personality WHERE guild_id = ? AND name = ?", (guild_id, active_name))
            row = c.fetchone()
            return row[0] if row else "You are GarGPT, a helpful assistant with a bit of sass and a love for the word 'Gar'."
    except Exception as e:
        logger.error(f"Error getting personality prompt: {e}")
        return "You are GarGPT, a helpful assistant with a bit of sass and a love for the word 'Gar'."

@bot.event
async def on_ready():
    """Bot ready event handler."""
    try:
        await tree.sync()
        logger.info(f"{bot_name} connected as {bot.user}")
        print(f"{bot_name} connected as {bot.user}")
    except Exception as e:
        logger.error(f"Error syncing commands: {e}")

@bot.event
async def on_message(message):
    """Message event handler for caching."""
    if message.author.bot:
        return
    
    try:
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            if db_manager.use_postgres:
                c.execute("INSERT INTO messages (channel, author, content, timestamp) VALUES (%s, %s, %s, %s)",
                         (str(message.channel.id), str(message.author), message.content, str(message.created_at)))
            else:
                c.execute("INSERT INTO messages (channel, author, content, timestamp) VALUES (?, ?, ?, ?)",
                         (str(message.channel.id), str(message.author), message.content, str(message.created_at)))
            conn.commit()
    except Exception as e:
        logger.error(f"Error caching message: {e}")

@tree.command(name="ask", description="Ask GPT-4o a question")
async def ask(interaction: discord.Interaction, prompt: str):
    """Ask GPT-4o a question."""
    if not is_allowed(interaction):
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return
    
    # Rate limiting check
    if rate_limiter.is_rate_limited(interaction.user.id):
        reset_time = rate_limiter.get_reset_time(interaction.user.id)
        await interaction.response.send_message(
            f"Rate limit exceeded. Please wait {reset_time} seconds before trying again.", 
            ephemeral=True
        )
        return
    
    await interaction.response.defer()
    
    try:
        # Sanitize input
        sanitized_prompt = sanitize_input(prompt)
        
        # Get personality and create completion
        guild_id = str(interaction.guild.id) if interaction.guild else "dm"
        system_prompt = get_personality_prompt(guild_id)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sanitized_prompt}
        ]
        
        answer = await openai_manager.create_completion(messages, max_tokens=300)
        
        if answer:
            await send_chunked_response(interaction, answer)
            logger.info(f"Ask command used by {interaction.user} in {interaction.guild}")
        else:
            await interaction.followup.send("I couldn't generate a response. Please try again.")
            
    except ValueError as e:
        await interaction.followup.send(f"Input error: {str(e)}", ephemeral=True)
    except Exception as e:
        logger.error(f"Error in ask command: {e}")
        await interaction.followup.send("An error occurred while processing your request.", ephemeral=True)

@tree.command(name="web", description="Search the web and summarize with GPT-4o")
async def web(interaction: discord.Interaction, query: str):
    """Search the web and summarize results with GPT-4o."""
    if not is_allowed(interaction):
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return
    
    # Rate limiting check
    if rate_limiter.is_rate_limited(interaction.user.id):
        reset_time = rate_limiter.get_reset_time(interaction.user.id)
        await interaction.response.send_message(
            f"Rate limit exceeded. Please wait {reset_time} seconds before trying again.", 
            ephemeral=True
        )
        return
    
    await interaction.response.defer()
    
    try:
        # Sanitize input
        sanitized_query = sanitize_input(query)
        
        # Perform web search
        headers = {"X-Subscription-Token": BRAVE_API_KEY}
        params = {"q": sanitized_query, "count": 3}
        
        r = requests.get("https://api.search.brave.com/res/v1/web/search", 
                        headers=headers, params=params, timeout=10)
        
        if r.status_code != 200:
            await interaction.followup.send("Web search failed. Please try again later.")
            logger.warning(f"Brave API returned status code: {r.status_code}")
            return
        
        results = r.json().get("web", {}).get("results", [])
        if not results:
            await interaction.followup.send("No web results found.")
            return
        
        # Format search results
        snippets = "\n\n".join([
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['description']}" 
            for r in results
        ])
        
        # Get personality and create summary
        guild_id = str(interaction.guild.id) if interaction.guild else "dm"
        system_prompt = get_personality_prompt(guild_id)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Summarize these search results and answer the query: {sanitized_query}\n\n{snippets}"}
        ]
        
        answer = await openai_manager.create_completion(messages, max_tokens=400)
        
        if answer:
            await send_chunked_response(interaction, answer)
            logger.info(f"Web command used by {interaction.user} in {interaction.guild}")
        else:
            await interaction.followup.send("I couldn't generate a summary. Please try again.")
            
    except ValueError as e:
        await interaction.followup.send(f"Input error: {str(e)}", ephemeral=True)
    except requests.RequestException as e:
        logger.error(f"Web search request error: {e}")
        await interaction.followup.send("Web search failed due to network issues. Please try again.")
    except Exception as e:
        logger.error(f"Error in web command: {e}")
        await interaction.followup.send("An error occurred while processing your request.", ephemeral=True)

@tree.command(name="status", description="Show current bot status and runtime info")
async def status(interaction: discord.Interaction):
    """Show bot status information."""
    try:
        uptime = time.time() - bot.start_time
        uptime_str = str(timedelta(seconds=int(uptime)))
        guild_id = str(interaction.guild.id) if interaction.guild else "dm"
        
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            if db_manager.use_postgres:
                c.execute("SELECT active_name FROM personality_active WHERE guild_id = %s", (guild_id,))
            else:
                c.execute("SELECT active_name FROM personality_active WHERE guild_id = ?", (guild_id,))
            row = c.fetchone()
            personality = row[0] if row else "default"
        
        latency_ms = round(bot.latency * 1000)
        
        status_msg = f"""**GarGPT Status**
Model: GPT-4o
Active Personality: `{personality}`
Uptime: `{uptime_str}`
Latency: `{latency_ms} ms`
Rate Limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s"""
        
        await interaction.response.send_message(status_msg.strip())
        
    except Exception as e:
        logger.error(f"Error in status command: {e}")
        await interaction.response.send_message("Error retrieving status information.", ephemeral=True)

@tree.command(name="setpersonality", description="Save and set a new personality")
@app_commands.describe(name="The personality name", prompt="The system prompt for this personality")
async def setpersonality(interaction: discord.Interaction, name: str, prompt: str):
    """Save and set a new personality."""
    if not is_allowed(interaction):
        await interaction.response.send_message("You don't have permission to set personalities.", ephemeral=True)
        return
    
    try:
        # Sanitize inputs
        sanitized_name = sanitize_input(name, 50)
        sanitized_prompt = sanitize_input(prompt, 1000)
        
        guild_id = str(interaction.guild.id) if interaction.guild else "dm"
        
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            if db_manager.use_postgres:
                c.execute("INSERT INTO personality (guild_id, name, system_prompt) VALUES (%s, %s, %s) ON CONFLICT (guild_id, name) DO UPDATE SET system_prompt = EXCLUDED.system_prompt, created_at = CURRENT_TIMESTAMP",
                         (guild_id, sanitized_name, sanitized_prompt))
                c.execute("INSERT INTO personality_active (guild_id, active_name) VALUES (%s, %s) ON CONFLICT (guild_id) DO UPDATE SET active_name = EXCLUDED.active_name, updated_at = CURRENT_TIMESTAMP",
                         (guild_id, sanitized_name))
            else:
                c.execute("INSERT OR REPLACE INTO personality VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                         (guild_id, sanitized_name, sanitized_prompt))
                c.execute("INSERT OR REPLACE INTO personality_active VALUES (?, ?, CURRENT_TIMESTAMP)",
                         (guild_id, sanitized_name))
            conn.commit()
        
        await interaction.response.send_message(f"Set personality '{sanitized_name}' and made it active for this server.")
        logger.info(f"Personality '{sanitized_name}' set by {interaction.user} in {interaction.guild}")
        
    except ValueError as e:
        await interaction.response.send_message(f"Input error: {str(e)}", ephemeral=True)
    except Exception as e:
        logger.error(f"Error in setpersonality command: {e}")
        await interaction.response.send_message("Error setting personality.", ephemeral=True)

@tree.command(name="usepersonality", description="Switch to a saved personality")
@app_commands.describe(name="The personality name to use")
async def usepersonality(interaction: discord.Interaction, name: str):
    """Switch to a saved personality."""
    if not is_allowed(interaction):
        await interaction.response.send_message("You don't have permission to use personalities.", ephemeral=True)
        return
    
    try:
        sanitized_name = sanitize_input(name, 50)
        guild_id = str(interaction.guild.id) if interaction.guild else "dm"
        
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            # Check if personality exists
            if db_manager.use_postgres:
                c.execute("SELECT name FROM personality WHERE guild_id = %s AND name = %s", (guild_id, sanitized_name))
            else:
                c.execute("SELECT name FROM personality WHERE guild_id = ? AND name = ?", (guild_id, sanitized_name))
            
            if not c.fetchone():
                await interaction.response.send_message(f"Personality '{sanitized_name}' not found for this server.", ephemeral=True)
                return
            
            # Set active personality
            if db_manager.use_postgres:
                c.execute("INSERT INTO personality_active (guild_id, active_name) VALUES (%s, %s) ON CONFLICT (guild_id) DO UPDATE SET active_name = EXCLUDED.active_name, updated_at = CURRENT_TIMESTAMP",
                         (guild_id, sanitized_name))
            else:
                c.execute("INSERT OR REPLACE INTO personality_active VALUES (?, ?, CURRENT_TIMESTAMP)",
                         (guild_id, sanitized_name))
            conn.commit()
        
        await interaction.response.send_message(f"Switched to personality '{sanitized_name}' for this server.")
        logger.info(f"Switched to personality '{sanitized_name}' by {interaction.user} in {interaction.guild}")
        
    except ValueError as e:
        await interaction.response.send_message(f"Input error: {str(e)}", ephemeral=True)
    except Exception as e:
        logger.error(f"Error in usepersonality command: {e}")
        await interaction.response.send_message("Error switching personality.", ephemeral=True)

@tree.command(name="listpersonalities", description="View all saved personalities")
async def listpersonalities(interaction: discord.Interaction):
    """List all saved personalities."""
    try:
        guild_id = str(interaction.guild.id) if interaction.guild else "dm"
        
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            # Get all personalities
            if db_manager.use_postgres:
                c.execute("SELECT name, system_prompt FROM personality WHERE guild_id = %s", (guild_id,))
            else:
                c.execute("SELECT name, system_prompt FROM personality WHERE guild_id = ?", (guild_id,))
            personalities = c.fetchall()
            
            if not personalities:
                await interaction.response.send_message("No personalities saved for this server.", ephemeral=True)
                return
            
            # Get active personality
            if db_manager.use_postgres:
                c.execute("SELECT active_name FROM personality_active WHERE guild_id = %s", (guild_id,))
            else:
                c.execute("SELECT active_name FROM personality_active WHERE guild_id = ?", (guild_id,))
            active_row = c.fetchone()
            active_name = active_row[0] if active_row else None
        
        personality_list = "**Saved Personalities:**\n\n"
        for row in personalities:
            name, prompt = row[0], row[1]
            status = " ✅ (Active)" if name == active_name else ""
            personality_list += f"**{name}**{status}\n{prompt[:100]}{'...' if len(prompt) > 100 else ''}\n\n"
        
        await send_chunked_response(interaction, personality_list, ephemeral=True)
        
    except Exception as e:
        logger.error(f"Error in listpersonalities command: {e}")
        await interaction.response.send_message("Error retrieving personalities.", ephemeral=True)

@tree.command(name="deletepersonality", description="Delete a saved personality")
@app_commands.describe(name="The personality name to delete")
async def deletepersonality(interaction: discord.Interaction, name: str):
    """Delete a saved personality."""
    if not is_allowed(interaction):
        await interaction.response.send_message("You don't have permission to delete personalities.", ephemeral=True)
        return
    
    try:
        sanitized_name = sanitize_input(name, 50)
        guild_id = str(interaction.guild.id) if interaction.guild else "dm"
        
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            # Check if deleting active personality
            if db_manager.use_postgres:
                c.execute("SELECT active_name FROM personality_active WHERE guild_id = %s", (guild_id,))
            else:
                c.execute("SELECT active_name FROM personality_active WHERE guild_id = ?", (guild_id,))
            row = c.fetchone()
            
            if row and row[0] == sanitized_name:
                if db_manager.use_postgres:
                    c.execute("DELETE FROM personality_active WHERE guild_id = %s", (guild_id,))
                else:
                    c.execute("DELETE FROM personality_active WHERE guild_id = ?", (guild_id,))
            
            # Delete personality
            if db_manager.use_postgres:
                c.execute("DELETE FROM personality WHERE guild_id = %s AND name = %s", (guild_id, sanitized_name))
            else:
                c.execute("DELETE FROM personality WHERE guild_id = ? AND name = ?", (guild_id, sanitized_name))
            conn.commit()
        
        await interaction.response.send_message(f"Deleted personality '{sanitized_name}' for this server.")
        logger.info(f"Personality '{sanitized_name}' deleted by {interaction.user} in {interaction.guild}")
        
    except ValueError as e:
        await interaction.response.send_message(f"Input error: {str(e)}", ephemeral=True)
    except Exception as e:
        logger.error(f"Error in deletepersonality command: {e}")
        await interaction.response.send_message("Error deleting personality.", ephemeral=True)

@tree.command(name="help", description="Show a list of available GarGPT commands")
async def help_command(interaction: discord.Interaction):
    """Show help information."""
    help_text = """**GarGPT Slash Commands**

🧠 **Personality Control:**
`/setpersonality [name] [prompt]` — Save and set a new personality  
`/usepersonality [name]` — Switch to a saved personality  
`/listpersonalities` — View all saved personalities  
`/deletepersonality [name]` — Delete a saved personality  

💬 **Chat + Search:**
`/ask [prompt]` — Ask GPT-4o a question  
`/web [query]` — Search the web and summarize results  

📊 **Status:**
`/status` — Show uptime, latency, and active personality  

⚡ **Rate Limits:**
{RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds per user"""
    
    await interaction.response.send_message(help_text.strip(), ephemeral=True)

# Error handling for the bot
@bot.event
async def on_error(event, *args, **kwargs):
    """Global error handler."""
    logger.error(f"Bot error in {event}: {args}, {kwargs}")

@bot.event
async def on_command_error(ctx, error):
    """Command error handler."""
    logger.error(f"Command error: {error}")

# Initialize bot start time
bot.start_time = time.time()

if __name__ == "__main__":
    logger.info("Starting GarGPT bot...")
    print("Launching bot...")
    try:
        bot.run(DISCORD_BOT_TOKEN)
    except Exception as e:
        logger.critical(f"Failed to start bot: {e}")
        raise
