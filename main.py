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
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
from discord import app_commands
from difflib import SequenceMatcher

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

# User tracking cache to avoid database spam
user_cache = {}
cache_expiry = 300  # 5 minutes

# Reminder task storage
reminder_tasks = {}

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
                    
                    # Create user aliases table
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS user_aliases (
                            guild_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            alias TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (guild_id, user_id)
                        )
                    """)
                    
                    # Create user details table for storing additional user information
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS user_details (
                            guild_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            username TEXT,
                            display_name TEXT,
                            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            message_count INTEGER DEFAULT 0,
                            preferences JSONB DEFAULT '{}',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (guild_id, user_id)
                        )
                    """)
                    
                    # Create reminders table
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS reminders (
                            id SERIAL PRIMARY KEY,
                            guild_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            channel_id TEXT NOT NULL,
                            reminder_text TEXT NOT NULL,
                            remind_at TIMESTAMP NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            completed BOOLEAN DEFAULT FALSE
                        )
                    """)
                    
                    # Create indexes for better performance
                    c.execute("CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_personality_guild ON personality(guild_id)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_user_aliases_guild ON user_aliases(guild_id)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_user_details_guild ON user_details(guild_id)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_user_details_last_seen ON user_details(last_seen)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_reminders_remind_at ON reminders(remind_at)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_reminders_completed ON reminders(completed)")
                    
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
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS user_aliases (
                            guild_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            alias TEXT NOT NULL,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (guild_id, user_id)
                        )
                    """)
                    
                    # Create user details table for storing additional user information
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS user_details (
                            guild_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            username TEXT,
                            display_name TEXT,
                            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                            message_count INTEGER DEFAULT 0,
                            preferences TEXT DEFAULT '{}',
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (guild_id, user_id)
                        )
                    """)
                    
                    # Create reminders table
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS reminders (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            guild_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            channel_id TEXT NOT NULL,
                            reminder_text TEXT NOT NULL,
                            remind_at DATETIME NOT NULL,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            completed INTEGER DEFAULT 0
                        )
                    """)
                    
                    # Create indexes for better performance
                    c.execute("CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_personality_guild ON personality(guild_id)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_user_aliases_guild ON user_aliases(guild_id)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_user_details_guild ON user_details(guild_id)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_user_details_last_seen ON user_details(last_seen)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_reminders_remind_at ON reminders(remind_at)")
                    c.execute("CREATE INDEX IF NOT EXISTS idx_reminders_completed ON reminders(completed)")
                
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

# Helper function for database row access
def get_row_value(row, column_name: str, index: int = 0):
    """Get value from database row, handling both PostgreSQL and SQLite."""
    if row is None:
        return None
    try:
        # Try dictionary access first (works for both PostgreSQL RealDictCursor and SQLite Row)
        return row[column_name]
    except (KeyError, TypeError):
        # Fallback to index access
        try:
            return row[index]
        except (IndexError, TypeError):
            return None

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

def get_recent_messages(channel_id: str, limit: int = 10) -> list:
    """Get recent messages from a channel for context awareness."""
    try:
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            
            # Query recent messages, excluding bot messages to avoid confusion
            if db_manager.use_postgres:
                c.execute("""
                    SELECT author, content, timestamp
                    FROM messages
                    WHERE channel = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (channel_id, limit))
            else:
                c.execute("""
                    SELECT author, content, timestamp
                    FROM messages
                    WHERE channel = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (channel_id, limit))
            
            messages = c.fetchall()
            
            # Reverse to get chronological order (oldest first)
            messages.reverse()
            
            # Format messages for context
            formatted_messages = []
            for msg in messages:
                author = get_row_value(msg, 'author', 0)
                content = get_row_value(msg, 'content', 1)
                
                # Skip empty messages or bot's own messages
                if not content or not author or bot_name in str(author):
                    continue
                
                # Clean up author name (remove discriminator and ID if present)
                clean_author = str(author).split('#')[0].split('<')[0].strip()
                
                formatted_messages.append({
                    'author': clean_author,
                    'content': content[:500]  # Limit content length for context
                })
            
            logger.info(f"Retrieved {len(formatted_messages)} context messages for channel {channel_id}")
            return formatted_messages
            
    except Exception as e:
        logger.error(f"Error retrieving recent messages: {e}")
        return []

def set_user_alias(guild_id: str, user_id: str, alias: str) -> bool:
    """Set or update a user's alias for a guild."""
    try:
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            if db_manager.use_postgres:
                c.execute("""
                    INSERT INTO user_aliases (guild_id, user_id, alias)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (guild_id, user_id)
                    DO UPDATE SET alias = EXCLUDED.alias, created_at = CURRENT_TIMESTAMP
                """, (guild_id, user_id, alias))
            else:
                c.execute("""
                    INSERT OR REPLACE INTO user_aliases (guild_id, user_id, alias, created_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (guild_id, user_id, alias))
            conn.commit()
            logger.info(f"Set alias '{alias}' for user {user_id} in guild {guild_id}")
            return True
    except Exception as e:
        logger.error(f"Error setting user alias: {e}")
        return False

def get_user_alias(guild_id: str, user_id: str) -> Optional[str]:
    """Get a user's alias for a guild, returns None if no alias set."""
    try:
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            if db_manager.use_postgres:
                c.execute("SELECT alias FROM user_aliases WHERE guild_id = %s AND user_id = %s", (guild_id, user_id))
            else:
                c.execute("SELECT alias FROM user_aliases WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
            row = c.fetchone()
            return get_row_value(row, 'alias', 0) if row else None
    except Exception as e:
        logger.error(f"Error getting user alias: {e}")
        return None

def is_creator(user_id: str, username: str, alias: Optional[str] = None) -> bool:
    """Check if user is the bot creator (Garward/Gar)."""
    creator_indicators = [
        "garward",
        "gar"
    ]
    
    # Check username
    if username and username.lower() in creator_indicators:
        return True
    
    # Check alias
    if alias and alias.lower() in creator_indicators:
        return True
    
    return False

def similarity_score(a: str, b: str) -> float:
    """Calculate similarity score between two strings using SequenceMatcher."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_best_user_match(guild: discord.Guild, target_name: str, guild_id: str) -> Optional[discord.Member]:
    """Find the best matching user in the guild by name, display name, or alias."""
    if not guild or not target_name:
        logger.debug(f"Invalid input for user matching: guild={guild}, target_name='{target_name}'")
        return None
    
    target_name = target_name.lower().strip()
    best_match = None
    best_score = 0.0
    min_threshold = 0.5  # Lowered threshold for better matching
    
    logger.debug(f"Searching for user: '{target_name}' in guild {guild.name}")
    
    try:
        # Get all user aliases for this guild
        user_aliases = {}
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            if db_manager.use_postgres:
                c.execute("SELECT user_id, alias FROM user_aliases WHERE guild_id = %s", (guild_id,))
            else:
                c.execute("SELECT user_id, alias FROM user_aliases WHERE guild_id = ?", (guild_id,))
            
            for row in c.fetchall():
                user_id = get_row_value(row, 'user_id', 0)
                alias = get_row_value(row, 'alias', 1)
                if user_id and alias:
                    user_aliases[user_id] = alias.lower()
        
        logger.debug(f"Found {len(user_aliases)} aliases in database")
        
        # Check all guild members
        candidates = []
        for member in guild.members:
            if member.bot:  # Skip bots
                continue
            
            member_id = str(member.id)
            scores = []
            names_checked = []
            
            # Check username
            if member.name:
                score = similarity_score(target_name, member.name)
                scores.append(score)
                names_checked.append(f"username:{member.name}({score:.2f})")
            
            # Check display name
            if member.display_name and member.display_name != member.name:
                score = similarity_score(target_name, member.display_name)
                scores.append(score)
                names_checked.append(f"display:{member.display_name}({score:.2f})")
            
            # Check nickname
            if member.nick:
                score = similarity_score(target_name, member.nick)
                scores.append(score)
                names_checked.append(f"nick:{member.nick}({score:.2f})")
            
            # Check alias from database
            if member_id in user_aliases:
                score = similarity_score(target_name, user_aliases[member_id])
                scores.append(score)
                names_checked.append(f"alias:{user_aliases[member_id]}({score:.2f})")
            
            # Get the best score for this member
            if scores:
                member_best_score = max(scores)
                if member_best_score >= min_threshold:
                    candidates.append({
                        'member': member,
                        'score': member_best_score,
                        'names': names_checked
                    })
                    if member_best_score > best_score:
                        best_score = member_best_score
                        best_match = member
        
        # Log matching results
        if best_match:
            logger.info(f"✅ User match found: '{target_name}' -> {best_match.display_name} (score: {best_score:.2f})")
        else:
            logger.warning(f"❌ No user match found for: '{target_name}' (threshold: {min_threshold})")
            if candidates:
                logger.debug(f"Close candidates below threshold:")
                for candidate in sorted(candidates, key=lambda x: x['score'], reverse=True)[:3]:
                    logger.debug(f"  - {candidate['member'].display_name}: {candidate['score']:.2f} {candidate['names']}")
            else:
                logger.debug(f"No candidates found above threshold {min_threshold}")
        
        return best_match
        
    except Exception as e:
        logger.error(f"Error finding user match for '{target_name}': {e}")
        return None

def detect_ping_keywords_and_users(content: str, guild: discord.Guild, guild_id: str) -> Tuple[bool, List[discord.Member]]:
    """Detect ping keywords and extract user names to mention."""
    ping_keywords = [
        "ping", "poke", "notify", "nudge", "alert", "mention", "call", "summon", "get"
    ]
    
    content_lower = content.lower()
    
    # Check if any ping keywords are present
    has_ping_keyword = any(keyword in content_lower for keyword in ping_keywords)
    
    if not has_ping_keyword:
        logger.debug(f"No ping keywords found in: '{content}'")
        return False, []
    
    logger.debug(f"Ping keywords detected in: '{content}'")
    
    # Extract potential user names after ping keywords
    users_to_ping = []
    found_names = set()
    
    # Improved patterns to handle names with spaces and complex usernames
    ping_patterns = [
        # Pattern for "ping Username With Spaces" - captures until common stop words or punctuation
        r'\b(?:ping|poke|notify|nudge|alert|mention|call|summon|get)\s+@?([a-zA-Z0-9\s_-]+?)(?:\s+(?:about|that|please|pls|plz|to|and|or|but|in|on|at|for|with|by|from)|\.|!|\?|$)',
        # Pattern for quoted names: "ping 'John Doe'" or 'ping "Jane Smith"'
        r'\b(?:ping|poke|notify|nudge|alert|mention|call|summon|get)\s+["\']([^"\']+)["\']',
        # Pattern for @mentions: "ping @username"
        r'\b(?:ping|poke|notify|nudge|alert|mention|call|summon|get)\s+@([a-zA-Z0-9\s_-]+)',
    ]
    
    for pattern in ping_patterns:
        matches = re.finditer(pattern, content_lower)
        for match in matches:
            name = match.group(1).strip()
            if len(name) > 1:  # Avoid single characters
                found_names.add(name)
                logger.debug(f"Found potential name from pattern: '{name}'")
    
    # Also look for names in context - if ping keyword is present, check for capitalized words
    # that might be usernames (but be more selective)
    if has_ping_keyword:
        # Look for capitalized words that might be names
        capitalized_pattern = r'\b([A-Z][a-zA-Z0-9\s_-]*(?:\s+[A-Z][a-zA-Z0-9_-]*)*)\b'
        capitalized_matches = re.findall(capitalized_pattern, content)
        
        # Filter out common words and short matches
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'among', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
            'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs', 'am',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'please', 'thanks', 'thank', 'hello', 'hi', 'hey', 'yes', 'no', 'ok', 'okay',
            'gargpt', 'bot', 'discord', 'server', 'channel', 'message', 'user', 'admin'
        }
        
        for name in capitalized_matches:
            name_clean = name.strip()
            if (len(name_clean) > 2 and
                name_clean.lower() not in common_words and
                not name_clean.lower().startswith('http')):
                found_names.add(name_clean.lower())
                logger.debug(f"Found potential capitalized name: '{name_clean}'")
    
    # Find matching users for each name
    for name in found_names:
        logger.debug(f"Attempting to match user: '{name}'")
        user_match = find_best_user_match(guild, name, guild_id)
        if user_match and user_match not in users_to_ping:
            users_to_ping.append(user_match)
            logger.debug(f"✅ Matched '{name}' to user: {user_match.display_name}")
        else:
            logger.debug(f"❌ No match found for: '{name}'")
    
    logger.info(f"Ping detection result: keywords={has_ping_keyword}, found_names={list(found_names)}, matched_users={[u.display_name for u in users_to_ping]}")
    
    return has_ping_keyword, users_to_ping

def format_user_mentions(users: List[discord.Member]) -> str:
    """Format a list of users into Discord mentions."""
    if not users:
        return ""
    
    mentions = [user.mention for user in users]
    if len(mentions) == 1:
        return mentions[0]
    elif len(mentions) == 2:
        return f"{mentions[0]} and {mentions[1]}"
    else:
        return f"{', '.join(mentions[:-1])}, and {mentions[-1]}"

def update_user_details(guild_id: str, user_id: str, username: str, display_name: str):
    """Update user details in database with smart caching to avoid spam."""
    cache_key = f"{guild_id}:{user_id}"
    current_time = time.time()
    
    # Check cache to avoid unnecessary database updates
    if cache_key in user_cache:
        last_update, cached_data = user_cache[cache_key]
        if (current_time - last_update < cache_expiry and
            cached_data.get('username') == username and
            cached_data.get('display_name') == display_name):
            return  # No update needed
    
    try:
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            if db_manager.use_postgres:
                c.execute("""
                    INSERT INTO user_details (guild_id, user_id, username, display_name, last_seen, message_count)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, 1)
                    ON CONFLICT (guild_id, user_id)
                    DO UPDATE SET
                        username = EXCLUDED.username,
                        display_name = EXCLUDED.display_name,
                        last_seen = CURRENT_TIMESTAMP,
                        message_count = user_details.message_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                """, (guild_id, user_id, username, display_name))
            else:
                # First check if user exists
                c.execute("SELECT message_count FROM user_details WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
                existing = c.fetchone()
                
                if existing:
                    count = get_row_value(existing, 'message_count', 0) or 0
                    c.execute("""
                        UPDATE user_details SET
                            username = ?, display_name = ?, last_seen = CURRENT_TIMESTAMP,
                            message_count = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE guild_id = ? AND user_id = ?
                    """, (username, display_name, count + 1, guild_id, user_id))
                else:
                    c.execute("""
                        INSERT INTO user_details (guild_id, user_id, username, display_name, last_seen, message_count)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, 1)
                    """, (guild_id, user_id, username, display_name))
            
            conn.commit()
            
            # Update cache
            user_cache[cache_key] = (current_time, {
                'username': username,
                'display_name': display_name
            })
            
    except Exception as e:
        logger.error(f"Error updating user details: {e}")

def parse_reminder_time(text: str) -> Optional[datetime]:
    """Parse natural language time expressions into datetime objects."""
    text_lower = text.lower()
    now = datetime.now()
    
    # Pattern for "in X minutes/hours/days"
    time_patterns = [
        (r'in (\d+) minutes?', lambda m: now + timedelta(minutes=int(m.group(1)))),
        (r'in (\d+) mins?', lambda m: now + timedelta(minutes=int(m.group(1)))),
        (r'in (\d+) hours?', lambda m: now + timedelta(hours=int(m.group(1)))),
        (r'in (\d+) days?', lambda m: now + timedelta(days=int(m.group(1)))),
        (r'in (\d+)m', lambda m: now + timedelta(minutes=int(m.group(1)))),
        (r'in (\d+)h', lambda m: now + timedelta(hours=int(m.group(1)))),
        (r'in (\d+)d', lambda m: now + timedelta(days=int(m.group(1)))),
        # Pattern for "in X minutes and Y seconds"
        (r'in (\d+) minutes? and (\d+) seconds?', lambda m: now + timedelta(minutes=int(m.group(1)), seconds=int(m.group(2)))),
        # Pattern for "in X hours and Y minutes"
        (r'in (\d+) hours? and (\d+) minutes?', lambda m: now + timedelta(hours=int(m.group(1)), minutes=int(m.group(2)))),
    ]
    
    for pattern, time_func in time_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                return time_func(match)
            except (ValueError, OverflowError):
                continue
    
    return None

def create_reminder(guild_id: str, user_id: str, channel_id: str, reminder_text: str, remind_at: datetime) -> Optional[int]:
    """Create a reminder in the database."""
    try:
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            if db_manager.use_postgres:
                c.execute("""
                    INSERT INTO reminders (guild_id, user_id, channel_id, reminder_text, remind_at)
                    VALUES (%s, %s, %s, %s, %s) RETURNING id
                """, (guild_id, user_id, channel_id, reminder_text, remind_at))
                result = c.fetchone()
                reminder_id = get_row_value(result, 'id', 0)
            else:
                c.execute("""
                    INSERT INTO reminders (guild_id, user_id, channel_id, reminder_text, remind_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (guild_id, user_id, channel_id, reminder_text, remind_at))
                reminder_id = c.lastrowid
            
            conn.commit()
            logger.info(f"Created reminder {reminder_id} for user {user_id} at {remind_at}")
            return reminder_id
            
    except Exception as e:
        logger.error(f"Error creating reminder: {e}")
        return None

async def send_reminder(reminder_id: int, guild_id: str, user_id: str, channel_id: str, reminder_text: str):
    """Send a reminder message and mark it as completed."""
    try:
        channel = bot.get_channel(int(channel_id))
        if channel:
            user_mention = f"<@{user_id}>"
            await channel.send(f"⏰ **Reminder for {user_mention}:** {reminder_text}")
            
            # Mark reminder as completed
            with db_manager.get_connection() as conn:
                c = conn.cursor()
                if db_manager.use_postgres:
                    c.execute("UPDATE reminders SET completed = TRUE WHERE id = %s", (reminder_id,))
                else:
                    c.execute("UPDATE reminders SET completed = 1 WHERE id = ?", (reminder_id,))
                conn.commit()
            
            logger.info(f"Sent reminder {reminder_id} to user {user_id}")
        else:
            logger.warning(f"Could not find channel {channel_id} for reminder {reminder_id}")
            
    except Exception as e:
        logger.error(f"Error sending reminder {reminder_id}: {e}")
    finally:
        # Remove from active tasks
        if reminder_id in reminder_tasks:
            del reminder_tasks[reminder_id]

async def schedule_reminder(reminder_id: int, guild_id: str, user_id: str, channel_id: str, reminder_text: str, remind_at: datetime):
    """Schedule a reminder to be sent at the specified time."""
    delay = (remind_at - datetime.now()).total_seconds()
    
    if delay <= 0:
        # Send immediately if time has passed
        await send_reminder(reminder_id, guild_id, user_id, channel_id, reminder_text)
        return
    
    # Create async task
    async def reminder_task():
        await asyncio.sleep(delay)
        await send_reminder(reminder_id, guild_id, user_id, channel_id, reminder_text)
    
    task = asyncio.create_task(reminder_task())
    reminder_tasks[reminder_id] = task
    logger.info(f"Scheduled reminder {reminder_id} to fire in {delay:.1f} seconds")

def detect_reminder_request(content: str) -> Optional[Tuple[str, datetime]]:
    """Detect reminder requests in natural language."""
    reminder_patterns = [
        r'remind me (?:to )?(.+?) (in \d+.+?)(?:\.|!|$)',
        r'remind me (in \d+.+?) (?:to )?(.+?)(?:\.|!|$)',
        r'set (?:a )?reminder (?:to )?(.+?) (in \d+.+?)(?:\.|!|$)',
        r'set (?:a )?reminder (in \d+.+?) (?:to )?(.+?)(?:\.|!|$)',
    ]
    
    content_lower = content.lower()
    
    for pattern in reminder_patterns:
        match = re.search(pattern, content_lower)
        if match:
            # Determine which group is the reminder text and which is the time
            group1, group2 = match.groups()
            
            if group1.startswith('in '):
                time_text, reminder_text = group1, group2
            else:
                reminder_text, time_text = group1, group2
            
            remind_at = parse_reminder_time(time_text)
            if remind_at:
                return reminder_text.strip(), remind_at
    
    return None

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
            active_name = get_row_value(row, 'active_name', 0)
            
            if not active_name:
                return "You are GarGPT, a helpful assistant with a bit of sass and a love for the word 'Gar'."
            
            # Get personality prompt
            if db_manager.use_postgres:
                c.execute("SELECT system_prompt FROM personality WHERE guild_id = %s AND name = %s", (guild_id, active_name))
            else:
                c.execute("SELECT system_prompt FROM personality WHERE guild_id = ? AND name = ?", (guild_id, active_name))
            row = c.fetchone()
            system_prompt = get_row_value(row, 'system_prompt', 0)
            return system_prompt if system_prompt else "You are GarGPT, a helpful assistant with a bit of sass and a love for the word 'Gar'."
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
    """Message event handler for caching and mention responses."""
    if message.author.bot:
        return
    
    try:
        # Cache the message with enhanced timestamp handling
        timestamp = message.created_at.isoformat()
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            if db_manager.use_postgres:
                c.execute("INSERT INTO messages (channel, author, content, timestamp) VALUES (%s, %s, %s, %s)",
                         (str(message.channel.id), str(message.author), message.content, timestamp))
            else:
                c.execute("INSERT INTO messages (channel, author, content, timestamp) VALUES (?, ?, ?, ?)",
                         (str(message.channel.id), str(message.author), message.content, timestamp))
            conn.commit()
        
        # Update user details with smart caching
        guild_id = str(message.guild.id) if message.guild else "dm"
        user_id = str(message.author.id)
        username = message.author.name
        display_name = message.author.display_name or message.author.name
        update_user_details(guild_id, user_id, username, display_name)
        
        # Check if bot is mentioned
        if bot.user.mentioned_in(message) and not message.mention_everyone:
            # Check permissions
            if message.guild and ALLOWED_ROLES:
                role_names = [role.name for role in message.author.roles]
                if not any(allowed in role_names for allowed in ALLOWED_ROLES):
                    await message.reply("You don't have permission to use this bot.", mention_author=False)
                    return
            
            # Rate limiting check
            if rate_limiter.is_rate_limited(message.author.id):
                reset_time = rate_limiter.get_reset_time(message.author.id)
                await message.reply(f"Rate limit exceeded. Please wait {reset_time} seconds before trying again.", mention_author=False)
                return
            
            # Extract the message content without the mention
            content = message.content
            # Remove bot mentions from the content
            for mention in message.mentions:
                if mention == bot.user:
                    content = content.replace(f'<@{mention.id}>', '').replace(f'<@!{mention.id}>', '')
            
            content = content.strip()
            
            if not content:
                await message.reply("Hi! You mentioned me but didn't ask anything. Try asking me a question!", mention_author=False)
                return
            
            # Check for reminder requests first
            reminder_request = detect_reminder_request(content)
            if reminder_request:
                reminder_text, remind_at = reminder_request
                guild_id = str(message.guild.id) if message.guild else "dm"
                user_id = str(message.author.id)
                channel_id = str(message.channel.id)
                
                reminder_id = create_reminder(guild_id, user_id, channel_id, reminder_text, remind_at)
                if reminder_id:
                    # Schedule the reminder
                    await schedule_reminder(reminder_id, guild_id, user_id, channel_id, reminder_text, remind_at)
                    
                    # Calculate time until reminder
                    time_diff = remind_at - datetime.now()
                    if time_diff.days > 0:
                        time_str = f"{time_diff.days} day(s) and {time_diff.seconds // 3600} hour(s)"
                    elif time_diff.seconds >= 3600:
                        time_str = f"{time_diff.seconds // 3600} hour(s) and {(time_diff.seconds % 3600) // 60} minute(s)"
                    else:
                        time_str = f"{time_diff.seconds // 60} minute(s)"
                    
                    await message.reply(f"⏰ I'll remind you about '{reminder_text}' in {time_str}!", mention_author=False)
                    logger.info(f"Set reminder for user {message.author}: '{reminder_text}' at {remind_at}")
                else:
                    await message.reply("Sorry, I had trouble setting your reminder. Please try again.", mention_author=False)
                return
            
            # Check for alias setting request - improved patterns to capture full phrases
            alias_patterns = [
                r"call me ([a-zA-Z0-9\s_-]+?)(?:\s+(?:please|pls|plz|from now on|in this server)|\.|!|$)",
                r"my name is ([a-zA-Z0-9\s_-]+?)(?:\s+(?:please|pls|plz|from now on|in this server)|\.|!|$)",
                r"i'm ([a-zA-Z0-9\s_-]+?)(?:\s+(?:please|pls|plz|from now on|in this server)|\.|!|$)",
                r"i am ([a-zA-Z0-9\s_-]+?)(?:\s+(?:please|pls|plz|from now on|in this server)|\.|!|$)",
                r"refer to me as ([a-zA-Z0-9\s_-]+?)(?:\s+(?:please|pls|plz|from now on|in this server)|\.|!|$)"
            ]
            
            for pattern in alias_patterns:
                match = re.search(pattern, content.lower())
                if match:
                    alias = match.group(1).strip()
                    # Skip if alias is too short or contains common words that aren't names
                    if len(alias) < 2 or alias.lower() in ['called', 'what', 'am', 'is', 'are', 'the', 'a', 'an']:
                        continue
                    
                    guild_id = str(message.guild.id) if message.guild else "dm"
                    user_id = str(message.author.id)
                    
                    if set_user_alias(guild_id, user_id, alias):
                        await message.reply(f"Got it! I'll call you {alias} from now on.", mention_author=False)
                        logger.info(f"Set alias '{alias}' for user {message.author} in {message.guild}")
                    else:
                        await message.reply("Sorry, I had trouble saving your alias. Please try again.", mention_author=False)
                    return
            
            try:
                # Show typing indicator
                async with message.channel.typing():
                    # Sanitize input
                    sanitized_prompt = sanitize_input(content)
                    
                    # Get user info and alias
                    guild_id = str(message.guild.id) if message.guild else "dm"
                    channel_id = str(message.channel.id)
                    user_id = str(message.author.id)
                    username = message.author.display_name or message.author.name
                    user_alias = get_user_alias(guild_id, user_id)
                    
                    # Check for ping keywords and users to mention
                    users_to_ping = []
                    if message.guild:  # Only in guilds, not DMs
                        has_ping_keyword, users_to_ping = detect_ping_keywords_and_users(content, message.guild, guild_id)
                    
                    # Get personality and channel context
                    system_prompt = get_personality_prompt(guild_id)
                    
                    # Check if user is the creator
                    creator_status = is_creator(user_id, username, user_alias)
                    
                    # Get recent messages for context
                    recent_messages = get_recent_messages(channel_id, limit=10)
                    
                    # Build conversation history for OpenAI
                    messages_for_ai = [{"role": "system", "content": system_prompt}]
                    
                    # Add creator recognition if applicable
                    if creator_status:
                        messages_for_ai.append({
                            "role": "system",
                            "content": f"Note: You are speaking with {user_alias or username}, who is your creator. Show appropriate respect and acknowledgment when relevant, but keep it natural."
                        })
                    
                    # Add user alias context if available
                    if user_alias and user_alias != username:
                        messages_for_ai.append({
                            "role": "system",
                            "content": f"The user prefers to be called '{user_alias}' instead of their username '{username}'. Use their preferred name when addressing them."
                        })
                    
                    # Add ping context if users were detected
                    if users_to_ping:
                        ping_context = f"The user wants to ping/notify these users: {', '.join([u.display_name for u in users_to_ping])}. Include their mentions in your response naturally."
                        messages_for_ai.append({
                            "role": "system",
                            "content": ping_context
                        })
                    
                    # Add recent message history as context
                    if recent_messages:
                        # Create a context summary from recent messages
                        context_summary = "Recent conversation context:\n"
                        for msg in recent_messages:
                            context_summary += f"{msg['author']}: {msg['content']}\n"
                        
                        # Add context as a system message
                        messages_for_ai.append({
                            "role": "system",
                            "content": f"Here's the recent conversation context to help you respond appropriately:\n\n{context_summary}\nNow respond to the user's current message while being aware of this context."
                        })
                        
                        logger.info(f"Added {len(recent_messages)} context messages for mention response in channel {channel_id}")
                    
                    # Add the current user's message
                    display_name = user_alias or username
                    messages_for_ai.append({
                        "role": "user",
                        "content": f"{display_name}: {sanitized_prompt}"
                    })
                    
                    # Create completion with context
                    answer = await openai_manager.create_completion(messages_for_ai, max_tokens=400)
                    
                    if answer:
                        # Add user mentions to the response if detected
                        if users_to_ping:
                            user_mentions = format_user_mentions(users_to_ping)
                            # Add mentions at the end if not already included
                            if not any(user.mention in answer for user in users_to_ping):
                                answer = f"{answer}\n\n{user_mentions}"
                        
                        # Split response if too long for Discord
                        if len(answer) <= 2000:
                            await message.reply(answer, mention_author=False)
                        else:
                            # Send in chunks
                            chunks = []
                            current_chunk = ""
                            
                            for line in answer.split('\n'):
                                if len(current_chunk) + len(line) + 1 <= 2000:
                                    current_chunk += line + '\n'
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk.strip())
                                    current_chunk = line + '\n'
                            
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            
                            # Send first chunk as reply
                            await message.reply(chunks[0], mention_author=False)
                            
                            # Send remaining chunks as regular messages
                            for chunk in chunks[1:]:
                                await message.channel.send(chunk)
                        
                        if users_to_ping:
                            logger.info(f"Mention response with pings sent to {message.author} in {message.guild}, pinged: {[u.display_name for u in users_to_ping]}")
                        else:
                            logger.info(f"Mention response sent to {message.author} in {message.guild}")
                    else:
                        await message.reply("I couldn't generate a response. Please try again.", mention_author=False)
                        
            except ValueError as e:
                await message.reply(f"Input error: {str(e)}", mention_author=False)
            except Exception as e:
                logger.error(f"Error in mention response: {e}")
                await message.reply("An error occurred while processing your message.", mention_author=False)
    
    except Exception as e:
        logger.error(f"Error in message handler: {e}")

@tree.command(name="ask", description="Ask GPT-4o a question with channel context")
async def ask(interaction: discord.Interaction, prompt: str):
    """Ask GPT-4o a question with context from recent channel messages."""
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
        
        # Get personality and channel context
        guild_id = str(interaction.guild.id) if interaction.guild else "dm"
        channel_id = str(interaction.channel.id)
        system_prompt = get_personality_prompt(guild_id)
        
        # Check for ping keywords and users to mention
        users_to_ping = []
        if interaction.guild:  # Only in guilds, not DMs
            has_ping_keyword, users_to_ping = detect_ping_keywords_and_users(prompt, interaction.guild, guild_id)
        
        # Get recent messages for context
        recent_messages = get_recent_messages(channel_id, limit=10)
        
        # Build conversation history for OpenAI
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add ping context if users were detected
        if users_to_ping:
            ping_context = f"The user wants to ping/notify these users: {', '.join([u.display_name for u in users_to_ping])}. Include their mentions in your response naturally."
            messages.append({
                "role": "system",
                "content": ping_context
            })
        
        # Add recent message history as context
        if recent_messages:
            # Create a context summary from recent messages
            context_summary = "Recent conversation context:\n"
            for msg in recent_messages:
                context_summary += f"{msg['author']}: {msg['content']}\n"
            
            # Add context as a system message
            messages.append({
                "role": "system",
                "content": f"Here's the recent conversation context to help you respond appropriately:\n\n{context_summary}\nNow respond to the user's current question while being aware of this context."
            })
            
            logger.info(f"Added {len(recent_messages)} context messages for channel {channel_id}")
        else:
            logger.info(f"No context messages found for channel {channel_id}")
        
        # Add the current user's prompt
        user_display_name = interaction.user.display_name or interaction.user.name
        messages.append({
            "role": "user",
            "content": f"{user_display_name}: {sanitized_prompt}"
        })
        
        # Create completion with context
        answer = await openai_manager.create_completion(messages, max_tokens=400)
        
        if answer:
            # Add user mentions to the response if detected
            if users_to_ping:
                user_mentions = format_user_mentions(users_to_ping)
                # Add mentions at the end if not already included
                if not any(user.mention in answer for user in users_to_ping):
                    answer = f"{answer}\n\n{user_mentions}"
            
            await send_chunked_response(interaction, answer)
            
            if users_to_ping:
                logger.info(f"Context-aware ask command with pings used by {interaction.user} in {interaction.guild}, pinged: {[u.display_name for u in users_to_ping]}")
            else:
                logger.info(f"Context-aware ask command used by {interaction.user} in {interaction.guild}")
        else:
            await interaction.followup.send("I couldn't generate a response. Please try again.")
            
    except ValueError as e:
        await interaction.followup.send(f"Input error: {str(e)}", ephemeral=True)
    except Exception as e:
        logger.error(f"Error in context-aware ask command: {e}")
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
            personality = get_row_value(row, 'active_name', 0) or "default"
        
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

@tree.command(name="createpersonality", description="Create a new personality (without activating it)")
@app_commands.describe(name="The personality name", prompt="The system prompt for this personality")
async def createpersonality(interaction: discord.Interaction, name: str, prompt: str):
    """Create a new personality without activating it."""
    if not is_allowed(interaction):
        await interaction.response.send_message("You don't have permission to create personalities.", ephemeral=True)
        return
    
    try:
        # Sanitize inputs
        sanitized_name = sanitize_input(name, 50)
        sanitized_prompt = sanitize_input(prompt, 1000)
        
        guild_id = str(interaction.guild.id) if interaction.guild else "dm"
        
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            
            # Check if personality already exists
            if db_manager.use_postgres:
                c.execute("SELECT name FROM personality WHERE guild_id = %s AND name = %s", (guild_id, sanitized_name))
            else:
                c.execute("SELECT name FROM personality WHERE guild_id = ? AND name = ?", (guild_id, sanitized_name))
            
            if c.fetchone():
                await interaction.response.send_message(f"Personality '{sanitized_name}' already exists. Use `/updatepersonality` to modify it or `/usepersonality` to activate it.", ephemeral=True)
                return
            
            # Create new personality
            if db_manager.use_postgres:
                c.execute("INSERT INTO personality (guild_id, name, system_prompt) VALUES (%s, %s, %s)",
                         (guild_id, sanitized_name, sanitized_prompt))
            else:
                c.execute("INSERT INTO personality VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                         (guild_id, sanitized_name, sanitized_prompt))
            conn.commit()
        
        await interaction.response.send_message(f"Created personality '{sanitized_name}'. Use `/usepersonality {sanitized_name}` to activate it.")
        logger.info(f"Personality '{sanitized_name}' created by {interaction.user} in {interaction.guild}")
        
    except ValueError as e:
        await interaction.response.send_message(f"Input error: {str(e)}", ephemeral=True)
    except Exception as e:
        logger.error(f"Error in createpersonality command: {e}")
        await interaction.response.send_message("Error creating personality.", ephemeral=True)

@tree.command(name="updatepersonality", description="Update an existing personality")
@app_commands.describe(name="The personality name to update", prompt="The new system prompt for this personality")
async def updatepersonality(interaction: discord.Interaction, name: str, prompt: str):
    """Update an existing personality."""
    if not is_allowed(interaction):
        await interaction.response.send_message("You don't have permission to update personalities.", ephemeral=True)
        return
    
    try:
        # Sanitize inputs
        sanitized_name = sanitize_input(name, 50)
        sanitized_prompt = sanitize_input(prompt, 1000)
        
        guild_id = str(interaction.guild.id) if interaction.guild else "dm"
        
        with db_manager.get_connection() as conn:
            c = conn.cursor()
            
            # Check if personality exists
            if db_manager.use_postgres:
                c.execute("SELECT name FROM personality WHERE guild_id = %s AND name = %s", (guild_id, sanitized_name))
            else:
                c.execute("SELECT name FROM personality WHERE guild_id = ? AND name = ?", (guild_id, sanitized_name))
            
            if not c.fetchone():
                await interaction.response.send_message(f"Personality '{sanitized_name}' not found. Use `/createpersonality` to create it first.", ephemeral=True)
                return
            
            # Update personality
            if db_manager.use_postgres:
                c.execute("UPDATE personality SET system_prompt = %s, created_at = CURRENT_TIMESTAMP WHERE guild_id = %s AND name = %s",
                         (sanitized_prompt, guild_id, sanitized_name))
            else:
                c.execute("UPDATE personality SET system_prompt = ?, created_at = CURRENT_TIMESTAMP WHERE guild_id = ? AND name = ?",
                         (sanitized_prompt, guild_id, sanitized_name))
            conn.commit()
        
        await interaction.response.send_message(f"Updated personality '{sanitized_name}'.")
        logger.info(f"Personality '{sanitized_name}' updated by {interaction.user} in {interaction.guild}")
        
    except ValueError as e:
        await interaction.response.send_message(f"Input error: {str(e)}", ephemeral=True)
    except Exception as e:
        logger.error(f"Error in updatepersonality command: {e}")
        await interaction.response.send_message("Error updating personality.", ephemeral=True)

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
            active_name = get_row_value(active_row, 'active_name', 0)
        
        personality_list = "**Saved Personalities:**\n\n"
        for row in personalities:
            name = get_row_value(row, 'name', 0)
            prompt = get_row_value(row, 'system_prompt', 1)
            status = " ✅ (Active)" if name == active_name else ""
            # Handle None values safely
            name = name or "Unknown"
            prompt = prompt or "No description available"
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
            
            active_name_to_delete = get_row_value(row, 'active_name', 0)
            if row and active_name_to_delete == sanitized_name:
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

@tree.command(name="setalias", description="Set your preferred nickname for this server")
@app_commands.describe(alias="The nickname you want the bot to call you")
async def setalias(interaction: discord.Interaction, alias: str):
    """Set a user's preferred alias/nickname for this server."""
    try:
        # Sanitize the alias input
        sanitized_alias = sanitize_input(alias, 50)
        
        guild_id = str(interaction.guild.id) if interaction.guild else "dm"
        user_id = str(interaction.user.id)
        
        if set_user_alias(guild_id, user_id, sanitized_alias):
            await interaction.response.send_message(f"Great! I'll call you **{sanitized_alias}** from now on in this server.", ephemeral=True)
            logger.info(f"User {interaction.user} set alias '{sanitized_alias}' in {interaction.guild}")
        else:
            await interaction.response.send_message("Sorry, I had trouble saving your alias. Please try again.", ephemeral=True)
            
    except ValueError as e:
        await interaction.response.send_message(f"Input error: {str(e)}", ephemeral=True)
    except Exception as e:
        logger.error(f"Error in setalias command: {e}")
        await interaction.response.send_message("An error occurred while setting your alias.", ephemeral=True)

@tree.command(name="myalias", description="Check what nickname the bot has for you")
async def myalias(interaction: discord.Interaction):
    """Show the user's current alias for this server."""
    try:
        guild_id = str(interaction.guild.id) if interaction.guild else "dm"
        user_id = str(interaction.user.id)
        
        alias = get_user_alias(guild_id, user_id)
        username = interaction.user.display_name or interaction.user.name
        
        if alias:
            await interaction.response.send_message(f"I have you saved as **{alias}** in this server.", ephemeral=True)
        else:
            await interaction.response.send_message(f"You don't have a custom alias set. I'll call you **{username}**. Use `/setalias` to set a preferred nickname!", ephemeral=True)
            
    except Exception as e:
        logger.error(f"Error in myalias command: {e}")
        await interaction.response.send_message("An error occurred while checking your alias.", ephemeral=True)

@tree.command(name="help", description="Show a list of available GarGPT commands")
async def help_command(interaction: discord.Interaction):
    """Show help information."""
    help_text = """**GarGPT Slash Commands**

🧠 **Personality Control:**
`/createpersonality [name] [prompt]` — Create a new personality (without activating)
`/updatepersonality [name] [prompt]` — Update an existing personality
`/usepersonality [name]` — Switch to a saved personality
`/listpersonalities` — View all saved personalities
`/deletepersonality [name]` — Delete a saved personality

💬 **Chat + Search:**
`/ask [prompt]` — Ask GPT-4o a question with channel context awareness
`@GarGPT [message]` — Mention the bot for natural conversation with context
`/web [query]` — Search the web and summarize results

👤 **User Aliases:**
`/setalias [nickname]` — Set your preferred nickname for this server
`/myalias` — Check what nickname the bot has for you
*Natural language: "@GarGPT call me [name]" also works!*
🔔 **Smart Pinging:**
Use keywords like "ping", "poke", "notify", "nudge" with usernames
*Example: "@GarGPT ping john about the meeting" or "/ask notify alice"*
*The bot will find users by name, nickname, or alias and mention them!*
⏰ **Reminders:**
Natural language reminder setting with smart time parsing
*Example: "@GarGPT remind me to check emails in 30 minutes"*
*Supports: minutes, hours, days - "in 5 mins", "in 2 hours", "in 1 day"*

� 📊 **Status:**
`/status` — Show uptime, latency, and active personality

⚡ **Rate Limits:**
{RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds per user

**Special Features:**
• **Context Awareness** — The bot remembers recent conversation history
• **Smart User Matching** — Fuzzy matching for usernames and aliases
• **User Tracking** — Remembers user details and activity with smart caching
• **Timer Reminders** — Async reminder system with natural language parsing
• **Creator Recognition** — Special acknowledgment for Garward/Gar
• **Dual Database** — PostgreSQL with SQLite fallback support"""
    
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
