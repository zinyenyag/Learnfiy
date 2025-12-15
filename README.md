# AI Learning Platform

An intelligent learning platform for Zimbabwe Forms 5-6 commercial subjects (Mathematics, Accounting, Economics, Computer Science, Business Studies).

## Features

- 🤖 **AI Tutor** with intelligent responses using ChatGPT, Gemini, or Claude (auto-mode with fallback)
- 📚 **Syllabus Integration** - Answers grounded in curriculum PDFs
- 📝 Homework upload and analysis
- 📊 Interactive quizzes
- 📈 Progress tracking and analytics
- 🌐 **Multi-AI Support** - Automatically uses best available AI provider

## Setup

### 1. Set Up Backend

```bash
cd backend
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 1.5. Configure AI API Keys (Optional but Recommended)

The AI Tutor works best with an AI API key. You can use one or more of:

- **OpenAI (ChatGPT)**: Get key from https://platform.openai.com/api-keys
- **Google Gemini**: Get key from https://makersuite.google.com/app/apikey  
- **Anthropic (Claude)**: Get key from https://console.anthropic.com/

Create a `.env` file in the `backend` directory:

```bash
cd backend
# Copy the example file
copy .env.example .env  # Windows
# or
cp .env.example .env    # Linux/Mac
```

Then edit `.env` and add at least one API key:
```
OPENAI_API_KEY=sk-your-key-here
# OR
GEMINI_API_KEY=your-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Note:** The system will automatically try providers in order (OpenAI → Gemini → Anthropic) if one fails. You only need ONE key, but multiple provide better fallback.

### 2. Add Syllabus PDFs

Place your syllabus PDF files in the project root (not in backend folder):
- `Mathematics Syllabus.pdf`
- `Accounting Syllabus.pdf`
- `Economics Syllabus.pdf`
- `Computer Science Syllabus.pdf`
- `Business Studies Syllabus.pdf`

### 3. Start the Backend Server

```bash
cd backend

# Windows:
.venv\Scripts\activate
python -m uvicorn main:app --reload --port 5050

# Or use the script:
# Windows: start.bat
# Linux/Mac: bash start.sh
```

The API will be available at `http://localhost:5050`

### 4. Start the Frontend Server

In a new terminal:

```bash
# Option 1: Python HTTP Server
python -m http.server 8080

# Option 2: Node.js HTTP Server
npx http-server -p 8080
```

Then open `http://localhost:8080/index.html` in your browser.

## API Endpoints

- `POST /api/chat` - Chat with AI tutor
  ```json
  {
    "subject": "Mathematics",
    "message": "Explain simultaneous equations"
  }
  ```

- `GET /health` - Health check

## Development

The backend uses FastAPI with:
- **AI Integration**: OpenAI (ChatGPT), Google Gemini, Anthropic (Claude) with auto-fallback
- PDF text extraction (PyPDF2)
- Keyword-based syllabus search for context
- DuckDuckGo web search fallback
- CORS enabled for frontend access
- **Auto-mode**: Automatically selects best available AI provider

### How AI Auto-Mode Works

1. **Syllabus Context**: System searches syllabus PDFs for relevant content
2. **AI Enhancement**: Uses AI (ChatGPT/Gemini/Claude) to generate intelligent responses
3. **Auto-Fallback**: Tries providers in order: OpenAI → Gemini → Anthropic → Fallback
4. **All Subjects**: Works automatically for Mathematics, Accounting, Economics, Computer Science, Business Studies

## Project Structure

```
Learnfiy/
├── index.html              # Frontend application
├── backend/
│   ├── main.py            # FastAPI backend
│   ├── requirements.txt   # Python dependencies
│   ├── .venv/            # Virtual environment
│   └── start.bat/.sh     # Startup scripts
├── syllabus-reference.md  # Syllabus documentation
└── [Syllabus PDFs]        # Subject syllabus files (in root)
```

