# Streamlit Cloud Secrets Setup Guide

## 🔑 How to Add Your Anthropic API Key

Your Streamlit app is deployed but needs the API key configured. Follow these steps:

### Step 1: Go to Streamlit Cloud Dashboard
1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Find your app: **zinyenyag-learnfiy-app-no1r4w**

### Step 2: Add Secrets
1. Click on your app
2. Click "⋮" (three dots menu) → **"Settings"**
3. Scroll down to **"Secrets"** section
4. Click **"Open secrets editor"**

### Step 3: Paste This Configuration

Copy and paste this exact configuration into the secrets editor:

```toml
[api_keys]
ANTHROPIC_API_KEY = "sk-ant-your-api-key-here"
```

**Important:** Replace `sk-ant-your-api-key-here` with your actual API key from your Anthropic account.

### Step 4: Save and Redeploy
1. Click **"Save"** at the bottom
2. The app will automatically redeploy
3. Wait 1-2 minutes for deployment to complete
4. Refresh your app page

### Step 5: Verify
After redeployment, check the sidebar in your app:
- ✅ Should show: "Anthropic API Key: Active" (green)
- ❌ Should NOT show: "Anthropic API Key: Not configured" (red)

## 🐛 Troubleshooting

### Still showing "Not configured"?
1. Make sure you saved the secrets (click "Save" button)
2. Wait for the app to finish redeploying (check the status)
3. Hard refresh your browser (Ctrl+F5 or Cmd+Shift+R)
4. Check that the key is exactly as shown above (no extra spaces)

### App not redeploying?
1. Go to your app dashboard
2. Click "⋮" → "Reboot app" to force a redeploy

## 📝 Notes

- The API key is stored securely in Streamlit Cloud
- It's NOT in your GitHub repository (protected by .gitignore)
- Only you can see and edit the secrets
- The key will be available to your app automatically
