# How to Add API Key to Streamlit Cloud - Step by Step

## 🎯 Quick Steps (5 minutes)

### Step 1: Open Your Streamlit App
1. Go to: **https://share.streamlit.io**
2. Sign in with your GitHub account
3. Find your app: **zinyenyag-learnfiy-app-no1r4w** (or click on it from the list)

### Step 2: Open Settings
1. On your app page, look for **"⋮"** (three dots menu) - usually in the top right
2. Click the **"⋮"** menu
3. Click **"Settings"** from the dropdown

### Step 3: Add Environment Variable
1. In Settings, scroll down to find **"Secrets"** or **"Environment Variables"** section
2. Click **"Open secrets editor"** or **"Edit secrets"** button
3. You'll see a text editor

### Step 4: Paste This Code
**Copy this EXACT text and paste it into the editor:**

```toml
ANTHROPIC_API_KEY=YOUR_API_KEY_HERE
```

**OR if it asks for TOML format, use this:**

```toml
[api_keys]
ANTHROPIC_API_KEY = "YOUR_API_KEY_HERE"
```

**⚠️ IMPORTANT:** Replace `YOUR_API_KEY_HERE` with your actual Anthropic API key. Check your `backend/.env` file for the key value.

### Step 5: Save
1. Click **"Save"** button at the bottom
2. Wait for the app to redeploy (1-2 minutes)
3. You'll see a "Deploying..." or "Redeploying..." message

### Step 6: Verify
1. Go back to your app: **https://zinyenyag-learnfiy-app-no1r4w.streamlit.app**
2. Check the sidebar - it should show: **✅ Anthropic API Key: Active** (green)
3. Try asking a question in the AI Tutor chat

## 🔍 Alternative: If You See "Environment Variables" Instead

Some Streamlit Cloud versions use "Environment Variables" instead of "Secrets":

1. In Settings, look for **"Environment Variables"**
2. Click **"Add variable"** or **"New variable"**
3. Enter:
   - **Name:** `ANTHROPIC_API_KEY`
   - **Value:** (paste your Anthropic API key here)
4. Click **"Save"**

## 📸 Visual Guide

**Where to find Settings:**
```
Your App Page
    ↓
⋮ (three dots) → Settings → Secrets/Environment Variables
```

**What the editor looks like:**
- A text box where you paste the key
- A "Save" button at the bottom

## ❓ Troubleshooting

**Can't find Settings?**
- Make sure you're signed in
- Make sure you're the owner of the app
- Try refreshing the page

**App not redeploying?**
- Click "⋮" → "Reboot app" to force redeploy
- Wait 2-3 minutes

**Still showing "Not configured"?**
- Double-check you clicked "Save"
- Make sure there are no extra spaces in the key
- Hard refresh your browser (Ctrl+F5)

## ✅ Success Indicators

When it's working, you'll see:
- ✅ Green checkmark: "Anthropic API Key: Active"
- ✅ AI Tutor chat responds to questions
- ✅ No error messages

---

**Need help?** The key is already configured locally. For Streamlit Cloud, you just need to add it through the web interface as shown above.
