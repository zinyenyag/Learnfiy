"""
Learnfiy - AI Learning Platform (Streamlit App)
Main entry point for Streamlit Cloud deployment
"""

import streamlit as st
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Learnfiy - AI Learning Platform",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load secrets and set environment variables
try:
    # Set environment variables from Streamlit secrets
    if "api_keys" in st.secrets:
        if "ANTHROPIC_API_KEY" in st.secrets["api_keys"]:
            os.environ["ANTHROPIC_API_KEY"] = st.secrets["api_keys"]["ANTHROPIC_API_KEY"]
        if "OPENAI_API_KEY" in st.secrets["api_keys"]:
            os.environ["OPENAI_API_KEY"] = st.secrets["api_keys"].get("OPENAI_API_KEY", "")
        if "GEMINI_API_KEY" in st.secrets["api_keys"]:
            os.environ["GEMINI_API_KEY"] = st.secrets["api_keys"].get("GEMINI_API_KEY", "")
    
    # Also load from dotenv as fallback
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass
except Exception as e:
    st.warning(f"⚠️ Could not load all secrets: {e}")

# Import AI functions
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Simple AI call function
def call_ai(prompt: str, subject: str = "Mathematics") -> str:
    """Call AI with fallback"""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Try Anthropic first
    if ANTHROPIC_AVAILABLE and api_key and api_key != "sk-ant-PASTE_YOUR_KEY_HERE":
        try:
            client = Anthropic(api_key=api_key)
            system_prompt = f"You are a helpful tutor for Forms 5-6 {subject} students in Zimbabwe. Answer questions directly, clearly, and completely."
            
            msg = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            parts = []
            for block in msg.content:
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
            return "\n".join(parts).strip()
        except Exception as e:
            return f"Error calling Anthropic: {str(e)}"
    
    # Fallback message
    return "AI service not available. Please check your API keys in the secrets configuration."

# Main App
def main():
    st.title("📚 Learnfiy - AI Learning Platform")
    st.markdown("An intelligent learning platform for Zimbabwe Forms 5-6 commercial subjects")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Subject selection
        subjects = ["Mathematics", "Accounting", "Economics", "Computer Science", "Business Studies"]
        selected_subject = st.selectbox("Select Subject", subjects, index=0)
        
        # Check API key status
        st.subheader("🔑 API Status")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if anthropic_key and anthropic_key != "sk-ant-PASTE_YOUR_KEY_HERE":
            st.success("✅ Anthropic API Key: Active")
        else:
            st.error("❌ Anthropic API Key: Not configured")
        
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key and openai_key != "your_openai_key_here":
            st.info("ℹ️ OpenAI API Key: Available")
        else:
            st.info("ℹ️ OpenAI API Key: Not set")
        
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_key and gemini_key != "your_gemini_key_here":
            st.info("ℹ️ Gemini API Key: Available")
        else:
            st.info("ℹ️ Gemini API Key: Not set")
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        Learnfiy helps students learn with AI-powered tutoring.
        
        **Features:**
        - 🤖 AI Tutor
        - 📝 Homework Submission
        - 📊 Interactive Quizzes
        - 📈 Progress Tracking
        """)
    
    # Main content area
    # Tabs for different features
    tab1, tab2, tab3 = st.tabs(["🤖 AI Tutor", "📝 Homework", "📊 Quiz"])
    
    # Tab 1: AI Tutor
    with tab1:
        st.header("AI Tutor Chat")
        st.markdown(f"Ask questions about **{selected_subject}**")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your subject..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Call AI
                        response = call_ai(prompt, selected_subject)
                        
                        if response:
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            error_msg = "Sorry, I couldn't generate a response. Please check your API keys."
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Tab 2: Homework
    with tab2:
        st.header("Submit Homework")
        st.markdown(f"Upload your **{selected_subject}** homework")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'doc', 'png', 'jpg', 'jpeg'],
            help="Upload your completed homework"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            if st.button("Submit Homework"):
                with st.spinner("Processing homework..."):
                    try:
                        # Here you would process the homework
                        # For now, just show a success message
                        st.success("✅ Homework submitted successfully!")
                        st.info("Your homework has been uploaded. No automatic marking is performed.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Tab 3: Quiz
    with tab3:
        st.header("Interactive Quiz")
        st.markdown(f"Test your knowledge in **{selected_subject}**")
        
        if st.button("Generate Quiz Questions"):
            with st.spinner("Generating quiz questions..."):
                try:
                    # Generate quiz questions
                    st.info("Quiz generation feature coming soon!")
                    st.markdown("""
                    **Quiz Features:**
                    - Multiple choice questions
                    - Instant feedback
                    - Progress tracking
                    """)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
