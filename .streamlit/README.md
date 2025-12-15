# Streamlit Configuration Guide

## Secrets Configuration

The `secrets.toml` file contains all API keys and sensitive configuration needed for the application.

### Current Configuration

- **ANTHROPIC_API_KEY**: Active and configured
- **OPENAI_API_KEY**: Optional (commented out)
- **GEMINI_API_KEY**: Optional (commented out)

### How to Use Secrets in Your Streamlit App

```python
import streamlit as st
import os

# Method 1: Direct access (recommended for Streamlit)
anthropic_key = st.secrets["api_keys"]["ANTHROPIC_API_KEY"]

# Method 2: Set as environment variable (for compatibility with existing code)
os.environ["ANTHROPIC_API_KEY"] = st.secrets["api_keys"]["ANTHROPIC_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["api_keys"].get("OPENAI_API_KEY", "")
os.environ["GEMINI_API_KEY"] = st.secrets["api_keys"].get("GEMINI_API_KEY", "")

# Now your existing code that uses os.getenv() will work!
from anthropic import Anthropic
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

### For Streamlit Cloud Deployment

1. Go to your app on Streamlit Cloud
2. Click "Settings" → "Secrets"
3. Paste the contents of `secrets.toml` into the secrets editor
4. Save and redeploy

### Security Notes

- ⚠️ Never commit `secrets.toml` to git (it's in `.gitignore`)
- ✅ Always use `st.secrets` to access sensitive data
- ✅ Use environment variables for compatibility with existing FastAPI backend
