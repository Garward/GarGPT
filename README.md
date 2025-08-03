# GarGPT Discord Bot

A production-ready Discord bot that integrates GPT-4o with web search capabilities and personality management.

## Features

- **GPT-4o Integration**: Ask questions and get intelligent responses
- **Web Search**: Search the web and get AI-summarized results using Brave Search API
- **Personality System**: Create, manage, and switch between different AI personalities
- **Rate Limiting**: User-level rate limiting to prevent API abuse
- **Error Handling**: Comprehensive error handling for all API calls
- **Response Chunking**: Automatically splits long responses into multiple messages
- **Input Validation**: Sanitizes and validates all user inputs for security
- **Logging**: Detailed logging for debugging and monitoring
- **Database Management**: Proper connection pooling and error handling

## Setup

### Prerequisites

1. Python 3.8 or higher
2. Discord Bot Token
3. OpenAI API Key
4. Brave Search API Key

### Installation

1. Clone or download the bot files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   export DISCORD_BOT_TOKEN="your_discord_bot_token"
   export OPENAI_API_KEY="your_openai_api_key"
   export BRAVE_API_KEY="your_brave_search_api_key"
   export ALLOWED_ROLES="Admin,Moderator"  # Optional: comma-separated role names
   ```

4. Run the bot:
   ```bash
   python main.py
   ```

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `DISCORD_BOT_TOKEN` | Yes | Discord bot token | - |
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o | - |
| `BRAVE_API_KEY` | Yes | Brave Search API key | - |
| `ALLOWED_ROLES` | No | Comma-separated list of allowed Discord roles | All users allowed |
| `MAX_PROMPT_LENGTH` | No | Maximum length for user prompts | 2000 |
| `RATE_LIMIT_REQUESTS` | No | Number of requests per time window | 10 |
| `RATE_LIMIT_WINDOW` | No | Rate limit time window in seconds | 60 |

## Commands

### Chat Commands
- `/ask [prompt]` - Ask GPT-4o a question
- `/web [query]` - Search the web and get AI-summarized results

### Personality Management
- `/setpersonality [name] [prompt]` - Create and activate a new personality
- `/usepersonality [name]` - Switch to an existing personality
- `/listpersonalities` - View all saved personalities
- `/deletepersonality [name]` - Delete a personality

### Utility Commands
- `/status` - Show bot status, uptime, and current personality
- `/help` - Display command help

## Security Features

- **Input Sanitization**: All user inputs are sanitized to remove potentially harmful characters
- **Rate Limiting**: Users are limited to 10 requests per 60 seconds by default
- **Role-based Access**: Optional role restrictions for bot usage
- **Error Handling**: Comprehensive error handling prevents crashes and information leakage

## Logging

The bot logs all activities to:
- `gargpt.log` file for persistent logging
- Console output for real-time monitoring

Log levels include:
- INFO: Normal operations and command usage
- WARNING: Rate limits and API issues
- ERROR: Recoverable errors
- CRITICAL: Fatal errors that prevent bot startup

## Database

The bot uses SQLite for local data storage:
- `message_cache.db` - Stores message cache and personality data
- Automatic table creation and migration
- Connection pooling for reliability

## Error Handling

The bot includes comprehensive error handling for:
- OpenAI API rate limits and errors
- Network connectivity issues
- Database connection problems
- Invalid user inputs
- Discord API errors

## Rate Limiting

Default rate limits:
- 10 requests per user per 60 seconds
- Configurable via environment variables
- Automatic reset notifications

## Contributing

1. Follow Python PEP 8 style guidelines
2. Add appropriate logging for new features
3. Include error handling for all external API calls
4. Update this README for new features or configuration options

## License

This project is provided as-is for educational and personal use.
