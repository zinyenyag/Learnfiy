# Streamlit Configuration Guide

## Environment Variables

The app uses environment variables loaded from `.env` file or system environment.

### Configuration

Set your API keys in the `backend/.env` file:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here
GEMINI_API_KEY=your-key-here
```

### How It Works

The app automatically loads environment variables using `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()

# Then use os.getenv() to access
api_key = os.getenv("ANTHROPIC_API_KEY")
```

### Security Notes

- ⚠️ Never commit `.env` files to git (they're in `.gitignore`)
- ✅ Use environment variables for all API keys
- ✅ The app works with both local `.env` files and system environment variables
