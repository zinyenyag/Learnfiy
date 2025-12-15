# Streamlit Cloud Deployment Guide

## ✅ Files Created/Updated

1. **`app.py`** - Main Streamlit application
2. **`requirements.txt`** - All Python dependencies
3. **`.streamlit/secrets.toml`** - API keys configuration (local)
4. **`.streamlit/config.toml`** - Streamlit configuration

## 🚀 Deploy to Streamlit Cloud

### Step 1: Push to GitHub
Make sure all files are committed and pushed:
```bash
git add .
git commit -m "Add Streamlit app for cloud deployment"
git push
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `zinyenyag/Learnfiy`
5. Set main file path: `app.py`
6. Click "Deploy"

### Step 3: Add Secrets

1. In your Streamlit Cloud app dashboard, go to "Settings" → "Secrets"
2. Paste this configuration:

```toml
[api_keys]
ANTHROPIC_API_KEY = "sk-ant-your-api-key-here"
```

3. Click "Save"
4. The app will automatically redeploy

## 🔧 Local Testing

To test locally before deploying:

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 📋 Features

- ✅ AI Tutor Chat (using Anthropic Claude)
- ✅ Homework Submission
- ✅ Interactive Quiz (coming soon)
- ✅ API Key status monitoring
- ✅ Subject selection (Mathematics, Accounting, Economics, etc.)

## 🐛 Troubleshooting

### App shows blank page
- Check that `app.py` is in the root directory
- Verify `requirements.txt` includes all dependencies
- Check Streamlit Cloud logs for errors

### API not working
- Verify secrets are set correctly in Streamlit Cloud
- Check that the API key is valid
- Review the API status in the sidebar

### Import errors
- Ensure all packages in `requirements.txt` are installed
- Check that Python version is compatible (3.8+)

## 📝 Notes

- The `.streamlit/secrets.toml` file is for local development only
- For Streamlit Cloud, use the web dashboard to add secrets
- Never commit actual API keys to GitHub
