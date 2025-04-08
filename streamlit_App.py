import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="Document Genie ✨", layout="wide", initial_sidebar_state="expanded")

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import pdfplumber
import speech_recognition as sr
import pyttsx3
import langdetect
from langdetect import detect
from gtts import gTTS
import io
import base64

# Custom CSS for better UI
st.markdown("""
<style>
    /* Global Styles */
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Font settings for multilingual support */
    body, h1, h2, h3, p, div, span {
        font-family: 'Arial', 'Noto Sans', 'Noto Sans Devanagari', sans-serif !important;
    }
    
    /* Hindi text specific styles */
    .hindi-text {
        font-family: 'Noto Sans Devanagari', 'Arial', sans-serif !important;
        line-height: 1.6;
    }
    
    /* Card Styles */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    /* Button Styles */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        background: linear-gradient(135deg, #45a049 0%, #357a38 100%);
    }
    
    /* Response Box */
    .response-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .response-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar Styles */
    .sidebar .stButton>button {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
    }
    
    .sidebar .stButton>button:hover {
        background: linear-gradient(135deg, #1976D2 0%, #0d47a1 100%);
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.3);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a237e;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    
    h1 {
        background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Upload Box */
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background: rgba(76, 175, 80, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        border-color: #2196F3;
        background: rgba(33, 150, 243, 0.05);
    }
    
    /* Input Fields */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
    }
    
    /* Spinner */
    .stSpinner {
        text-align: center;
        color: #4CAF50;
    }
    
    /* Radio Buttons */
    .stRadio>div {
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Chat Container */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Keyword styling */
    .keyword-list {
        list-style: none;
        padding: 0;
    }
    .keyword-item {
        padding: 0.5rem 0;
        font-size: 1.1rem;
        color: #2c3e50;
        border-bottom: 1px solid #eee;
    }
    .keyword-item:last-child {
        border-bottom: none;
    }
    .keyword-bullet {
        color: #4CAF50;
        margin-right: 0.5rem;
    }
    
    /* Language selector styling */
    .language-selector {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Language badge */
    .language-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 20px;
        margin-left: 0.5rem;
    }
    .lang-en {
        background: #E3F2FD;
        color: #1976D2;
    }
    .lang-hi {
        background: #FFF8E1;
        color: #FFA000;
    }
</style>
""", unsafe_allow_html=True)

# ✅ Load API keys from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

# ✅ Extract Text from PDFs with language detection
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                extracted_text = page.extract_text() if page.extract_text() else ""
                text += extracted_text + "\n"
    
    # Detect language
    try:
        if text.strip():
            detected_lang = detect(text)
            if detected_lang == 'hi':
                st.session_state.document_language = 'Hindi'
            else:
                st.session_state.document_language = 'English'
        else:
            st.session_state.document_language = 'English'  # Default
    except langdetect.lang_detect_exception.LangDetectException:
        st.session_state.document_language = 'English'  # Default if detection fails
        
    return text

# ✅ Split Text into Chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

# ✅ Create FAISS Vector Store
def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# ✅ Retrieve Answers Using FAISS
def retrieve_answer(query, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(query)
    return docs

# ✅ Generate AI Answer (RAG) with language support
def generate_answer(query, api_key, response_language="auto"):
    docs = retrieve_answer(query, api_key)

    if not docs:
        return "❌ No relevant information found in the document."

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=api_key)

    # Set language instruction based on detected language or user preference
    lang_instruction = ""
    if response_language == "Hindi":
        lang_instruction = "Answer in Hindi language. "
    elif response_language == "English":
        lang_instruction = "Answer in English language. "
    elif response_language == "auto":
        # Auto mode - use the same language as detected in the document
        if "document_language" in st.session_state and st.session_state.document_language == "Hindi":
            lang_instruction = "Answer in Hindi language. "
        else:
            lang_instruction = "Answer in English language. "

    prompt_template = f"""
    {lang_instruction}Use the provided context to answer the user's question.

    If the answer is not in the context, say: "Answer is not available in the document."

    Context:\n {{context}}?\n
    Question: \n{{question}}\n
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)

    return response["output_text"]

# ✅ Speech-to-Text (Voice Search) with Hindi support
def recognize_speech(language="en-US"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=10)
            if language == "hi-IN":
                text = recognizer.recognize_google(audio, language="hi-IN")
            else:
                text = recognizer.recognize_google(audio, language="en-US")
            st.success(f"✅ You said: {text}")
            return text
        except sr.UnknownValueError:
            st.warning("❌ Could not understand audio. Please try again.")
        except sr.RequestError:
            st.error("❌ Could not request results. Check your internet.")
    return ""

# ✅ Text-to-Speech with Hindi support
def text_to_speech(text, language="en"):
    try:
        if language == "hi":
            # Use gTTS for Hindi
            tts = gTTS(text=text, lang=language, slow=False)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            
            # Create an HTML audio player with the base64-encoded audio
            audio_base64 = base64.b64encode(audio_bytes.read()).decode()
            audio_player = f"""
                <audio autoplay controls>
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
            """
            st.markdown(audio_player, unsafe_allow_html=True)
        else:
            # Use pyttsx3 for English (default)
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")

# ✅ Summarization Function with language support
def summarize_document(text, summary_type="medium", response_language="auto"):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")

    # Set language instruction based on detected language or user preference
    lang_instruction = ""
    if response_language == "Hindi":
        lang_instruction = "Respond in Hindi language. "
    elif response_language == "English":
        lang_instruction = "Respond in English language. "
    elif response_language == "auto":
        # Auto mode - use the same language as detected in the document
        if "document_language" in st.session_state and st.session_state.document_language == "Hindi":
            lang_instruction = "Respond in Hindi language. "
        else:
            lang_instruction = "Respond in English language. "

    summary_prompt = {
        "short": f"{lang_instruction}Summarize this document in 1 sentence. Return just the summary without any HTML tags.",
        "medium": f"{lang_instruction}Summarize this document in 3 bullet points. Format each point with a bullet (•) at the start. Return just the summary without any HTML tags.",
        "detailed": f"{lang_instruction}Summarize this document with a detailed paragraph. Return just the summary without any HTML tags."
    }

    try:
        response = model.generate_content(f"{summary_prompt[summary_type]}\n{text}")
        summary = response.text.strip()
        
        # Format bullet points if medium summary
        if summary_type == "medium":
            # Split by newlines and clean up
            points = [point.strip().replace('*', '•') for point in summary.split('\n') if point.strip()]
            # Ensure each point starts with a bullet
            points = ['• ' + point if not point.startswith('•') else point for point in points]
            summary = '\n'.join(points)
        
        return summary
    except Exception as e:
        return f"❌ Google Gemini API Error: {str(e)}"

# ✅ Keyword Extraction Function with language support
def extract_keywords(text, response_language="auto"):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")

    # Set language instruction based on detected language or user preference
    lang_instruction = ""
    if response_language == "Hindi":
        lang_instruction = "Respond in Hindi language. "
    elif response_language == "English":
        lang_instruction = "Respond in English language. "
    elif response_language == "auto":
        # Auto mode - use the same language as detected in the document
        if "document_language" in st.session_state and st.session_state.document_language == "Hindi":
            lang_instruction = "Respond in Hindi language. "
        else:
            lang_instruction = "Respond in English language. "

    try:
        response = model.generate_content(
            f"{lang_instruction}Extract exactly 5 important keywords or key phrases from the document. "
            "Return ONLY the keywords, one per line, with a bullet point (•) at the start. "
            "Do not include any other text or formatting.\n\n"
            f"Document text:\n{text}"
        )
        # Clean and format keywords
        keywords = response.text.strip().split('\n')
        # Format each keyword with bullet and styling
        formatted_keywords = []
        for kw in keywords:
            kw = kw.strip().replace('-', '').replace('*', '')  # Remove any existing bullets
            if kw:  # Only add non-empty keywords
                if not kw.startswith('•'):
                    kw = f'• {kw}'
                formatted_keywords.append(kw)
        
        # Join with HTML line breaks for proper display
        return '<br>'.join(formatted_keywords)
    except Exception as e:
        return f"❌ Google Gemini API Error: {str(e)}"

# ✅ Main Function
def main():
    global api_key
    
    # Initialize language settings in session state
    if 'document_language' not in st.session_state:
        st.session_state.document_language = 'English'  # Default
    if 'response_language' not in st.session_state:
        st.session_state.response_language = 'auto'  # Default to auto
    
    # Container for the entire app with reduced spacing
    st.markdown("""
        <style>
            .main .block-container {
                padding-top: 0.5rem;
                padding-bottom: 1.5rem;
                max-width: 85rem;
            }
            .main .element-container {
                margin-bottom: 0.5rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for answer
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = ""
    
    # Header with reduced spacing
    st.markdown("""
        <div style='text-align: center; padding: 1.5rem 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    border-radius: 12px; margin: 0.25rem 0 1rem 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
            <h1 style='margin-bottom: 0.5rem; font-size: 1.8rem;'>📚 Document Genie - Multilingual AI Chat Assistant ✨</h1>
            <p style='color: #666; font-size: 1rem; max-width: 600px; margin: 0 auto;'>
                Upload PDFs in English or Hindi, analyze documents, and chat with your content using AI
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with reduced spacing
    with st.sidebar:
        st.markdown("""
            <div style='padding: 0.75rem 0;'>
                <h2 style='text-align: center; margin-bottom: 1rem; padding: 0.75rem; 
                          background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                          border-radius: 8px; font-size: 1.3rem;'>
                    📄 Document Manager
                </h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Language information
        st.markdown("""
            <div class='card' style='margin: 1rem 0; padding: 0.75rem; text-align: center;'>
                <h3 style='margin-bottom: 0.5rem; font-size: 1.1rem;'>🌍 Multilingual Support</h3>
                <p style='font-size: 0.9rem; color: #555;'>
                    Document Genie now supports both English and Hindi documents!
                </p>
                <div style='display: flex; justify-content: center; margin-top: 0.5rem;'>
                    <span class='language-badge lang-en'>English</span>
                    <span class='language-badge lang-hi'>हिंदी</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # API Key input with reduced spacing
        if not api_key:
            st.markdown("<div style='margin: 1rem 0;'>", unsafe_allow_html=True)
            st.warning("⚠️ Please enter your Google API Key to continue")
            api_key = st.text_input("🔑 Enter Google API Key:", type="password", key="api_key_input")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Language settings
        st.markdown("""
            <div style='margin: 1rem 0;'>
                <div class='card' style='padding: 1rem;'>
                    <h3 style='margin-bottom: 0.5rem; color: #4CAF50; font-size: 1.1rem;'>🌐 Language Settings</h3>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        response_language = st.radio(
            "Response Language:",
            options=["Auto (Match Document)", "English", "Hindi"],
            index=0,
            key="response_language_select"
        )
        
        # Update session state based on selection
        if response_language == "Auto (Match Document)":
            st.session_state.response_language = "auto"
        else:
            st.session_state.response_language = response_language
            
        # Document language info if document is loaded
        if "raw_text" in st.session_state:
            st.info(f"📄 Detected document language: {st.session_state.document_language}")
        
        # Upload section with reduced spacing
        st.markdown("""
            <div style='margin: 1rem 0;'>
                <div class='upload-box' style='padding: 1rem;'>
                    <h3 style='margin-bottom: 0.5rem; color: #4CAF50; font-size: 1.1rem;'>📂 Upload Documents</h3>
                    <p style='color: #666; margin-bottom: 0.5rem; font-size: 0.9rem;'>Support for PDF files (English & Hindi)</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        pdf_docs = st.file_uploader("Upload PDFs", 
                                   accept_multiple_files=True, 
                                   key="pdf_uploader",
                                   help="Upload one or more PDF files to analyze",
                                   label_visibility="collapsed")

        if pdf_docs:
            st.markdown("<div style='margin: 1rem 0;'>", unsafe_allow_html=True)
            if "raw_text" not in st.session_state:
                with st.spinner("🔄 Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, api_key)
                    st.session_state.raw_text = raw_text
                    st.success("✅ Documents processed successfully!")

            if "raw_text" in st.session_state:
                st.markdown("""
                    <div style='background: white; padding: 1rem; border-radius: 12px; margin: 1rem 0;
                              box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
                        <h3 style='text-align: center; color: #1a237e; margin-bottom: 0.75rem; font-size: 1.1rem;'>
                            🛠️ Document Analysis
                        </h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Analysis options with reduced spacing
                st.markdown("<div style='margin: 0.75rem 0;'>", unsafe_allow_html=True)
                feature = st.radio("Select analysis type:", 
                                 ["📝 Summarize Document", "🔑 Extract Keywords"],
                                 key="selected_feature")
                st.markdown("</div>", unsafe_allow_html=True)

                if st.button("▶️ Analyze Document", key="run_feature"):
                    st.session_state.feature_result = None
                    
                    with st.spinner("⏳ Analyzing..."):
                        if "Summarize" in feature:
                            st.session_state.feature_result = summarize_document(
                                st.session_state.raw_text, 
                                "medium", 
                                st.session_state.response_language
                            )
                        elif "Keywords" in feature:
                            st.session_state.feature_result = extract_keywords(
                                st.session_state.raw_text,
                                st.session_state.response_language
                            )

                if "feature_result" in st.session_state and st.session_state.feature_result:
                    if "Summarize" in feature:
                        st.markdown("""
                            <div class='card' style='margin: 1rem 0; padding: 1rem;'>
                                <h3 style='margin-bottom: 0.75rem; font-size: 1.1rem;'>📊 Document Summary</h3>
                                <div class='response-box' style='padding: 1rem; white-space: pre-line;'>
                                    {}
                                </div>
                            </div>
                        """.format(st.session_state.feature_result), unsafe_allow_html=True)
                    else:
                        # Format keywords with custom styling
                        formatted_result = st.session_state.feature_result.replace('•', '<span class="keyword-bullet">•</span>')
                        st.markdown("""
                            <div class='card' style='margin: 1rem 0; padding: 1rem;'>
                                <h3 style='margin-bottom: 0.75rem; font-size: 1.1rem;'>🔑 Key Topics</h3>
                                <div class='response-box' style='padding: 1rem;'>
                                    <div class="keyword-list">
                                        {}
                                    </div>
                                </div>
                            </div>
                        """.format(formatted_result), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Main content area with reduced spacing
    main_container = st.container()
    
    with main_container:
        # Chat interface with reduced spacing
        st.markdown("""
            <div class='chat-container' style='padding: 1.5rem; margin: 1rem 0;'>
                <h2 style='text-align: center; margin-bottom: 1rem; font-size: 1.4rem;'>💬 Chat with your Documents</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Voice and text input with reduced spacing
        st.markdown("<div style='margin: 1rem 0;'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 5])

        user_question = ""

        with col1:
            # Voice input with language support
            voice_lang = "en-US"
            if st.session_state.response_language == "Hindi":
                voice_lang = "hi-IN"
            elif st.session_state.response_language == "auto" and st.session_state.document_language == "Hindi":
                voice_lang = "hi-IN"
                
            if st.button("🎤 Voice Search", key="voice_input_button", help="Click to use voice input"):
                user_question = recognize_speech(voice_lang)

        with col2:
            user_question = st.text_input("Ask a question", 
                                        key="user_question", 
                                        value=user_question,
                                        placeholder="💡 Type your question here...",
                                        label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        # Answer section with reduced spacing
        if user_question and api_key:
            if "raw_text" not in st.session_state:
                st.warning("⚠️ Please upload PDF documents first!")
            else:
                with st.spinner("🔍 Searching through documents..."):
                    answer = generate_answer(user_question, api_key, st.session_state.response_language)
                    st.markdown("""
                        <div class='card' style='margin: 1rem 0; padding: 1rem;'>
                            <h3 style='margin-bottom: 0.75rem; font-size: 1.1rem;'>📌 Answer</h3>
                            <div class='response-box' style='padding: 1rem;'>
                                {}
                            </div>
                        </div>
                    """.format(answer), unsafe_allow_html=True)
                    
                    # Text-to-speech button with reduced spacing
                    st.markdown("<div style='margin: 0.75rem 0; text-align: center;'>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        if st.button("🔊 Listen to Answer", key="tts_button"):
                            # Select TTS language based on detected document language or user preference
                            tts_lang = "en"
                            if st.session_state.response_language == "Hindi":
                                tts_lang = "hi"
                            elif st.session_state.response_language == "auto" and st.session_state.document_language == "Hindi":
                                tts_lang = "hi"
                            
                            text_to_speech(answer, tts_lang)
                    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
