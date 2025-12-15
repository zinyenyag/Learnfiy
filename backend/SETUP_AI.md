# AI Tutor Setup Guide

The AI Tutor now supports **automatic AI integration** with multiple providers and intelligent fallback.

## Quick Setup

### 1. Install Dependencies

```bash
cd backend
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Get an API Key (Choose One)

You only need **ONE** API key, but multiple provide better fallback:

#### Option A: OpenAI (ChatGPT) - Recommended
1. Go to https://platform.openai.com/api-keys
2. Create an account/login
3. Create a new API key
4. Copy the key (starts with `sk-`)

#### Option B: Google Gemini (Free tier available)
1. Go to https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Create API key
4. Copy the key

#### Option C: Anthropic (Claude)
1. Go to https://console.anthropic.com/
2. Create account/login
3. Create API key
4. Copy the key (starts with `sk-ant-`)

### 3. Configure API Key

Create a `.env` file in the `backend` directory:

```bash
cd backend
# Copy example file
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac
```

Edit `.env` and add your API key:

```env
# Add ONE of these (or multiple for fallback):
OPENAI_API_KEY=sk-your-actual-key-here
# OR
GEMINI_API_KEY=your-actual-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
```

### 4. Restart Backend

```bash
# Stop current server (Ctrl+C)
# Then restart:
python -m uvicorn main:app --reload --port 5050
```

## How It Works

### Auto-Mode Flow

1. **Student asks question** → Frontend sends to backend
2. **Syllabus Search** → System searches relevant syllabus PDFs for context
3. **AI Enhancement** → Uses AI to generate intelligent answer with syllabus context
4. **Auto-Fallback** → Tries providers in order:
   - OpenAI (ChatGPT) → 
   - Gemini → 
   - Anthropic (Claude) → 
   - Fallback response (if no keys)

### Features

✅ **Works for ALL subjects** automatically (Mathematics, Accounting, Economics, Computer Science, Business Studies)  
✅ **Syllabus-grounded** - Uses curriculum PDFs as context  
✅ **Intelligent responses** - AI generates clear, educational answers  
✅ **Auto-fallback** - If one provider fails, tries next automatically  
✅ **No configuration needed** - Works out of the box (with fallback if no keys)

## Testing

1. Start backend: `python -m uvicorn main:app --reload --port 5050`
2. Open frontend: `http://localhost:8080/index.html`
3. Select any subject
4. Ask a question in AI Tutor
5. You should get an intelligent AI-generated response!

## Troubleshooting

**"Backend not running" error:**
- Make sure backend is running on port 5050
- Check: `http://localhost:5050/health`

**"No AI response" or fallback message:**
- Check `.env` file exists in `backend/` directory
- Verify API key is correct (no extra spaces)
- Check API key has credits/quota available
- Restart backend after adding `.env` file

**Import errors:**
- Run: `pip install -r requirements.txt` again
- Make sure virtual environment is activated

