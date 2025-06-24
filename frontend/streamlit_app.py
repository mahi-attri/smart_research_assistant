import streamlit as st
import sys
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent.parent / "app"
sys.path.append(str(app_dir))

try:
    from ollama_processor import OllamaProcessor
except ImportError:
    # Mock processor for demonstration
    class OllamaProcessor:
        def __init__(self):
            self.available_models = ["llama3.2:3b", "llama3.1:8b"]
            self.full_document_text = ""
        
        def extract_text_from_pdf(self, file):
            return "Sample extracted text from PDF..."
        
        def extract_text_from_docx(self, file):
            return "Sample extracted text from DOCX..."
        
        def extract_text_from_txt(self, file):
            return file.read().decode("utf-8")
        
        def generate_auto_summary(self, max_words=150):
            return {"summary": "This is a sample summary of the document content.", "word_count": 12}
        
        def chat(self, question, context=""):
            return f"Sample answer to: {question}"
        
        def smart_question_answering(self, question):
            return {
                "answer": f"Sample intelligent answer to: {question}",
                "justification": "Based on document analysis",
                "source_reference": "Document content",
                "relevant_quote": None
            }
        
        def generate_challenge_questions(self, text, num_questions=3):
            return [
                {"question": "What is the main topic of this document?", "hint": "Look at the title and introduction"},
                {"question": "What are the key findings presented?", "hint": "Check the conclusion section"},
                {"question": "What methodology was used?", "hint": "Look for the methods section"}
            ]
        
        def evaluate_challenge_answer(self, question, user_answer):
            return {
                "evaluation": "Correct",
                "explanation": "Good answer! You understood the key concepts.",
                "correct_answer": "The correct approach would be..."
            }

# Page configuration
st.set_page_config(
    page_title="Research Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Black theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header {visibility: hidden;}
    
    /* Black theme */
    .stApp {
        background: #000000;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header */
    .header {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .logo {
        font-family: 'Space Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        margin-bottom: 1rem;
    }
    
    .nav-menu {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #cccccc;
        margin-bottom: 2rem;
    }
    
    .nav-menu span {
        margin: 0 1rem;
        cursor: pointer;
        transition: color 0.3s ease;
    }
    
    .nav-menu span:hover {
        color: #ffffff;
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 4rem 2rem;
        margin-bottom: 4rem;
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 4.5rem;
        font-weight: 900;
        color: #ffffff;
        line-height: 1.1;
        margin-bottom: 2rem;
        letter-spacing: -0.02em;
    }
    
    .hero-description {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 400;
        color: #cccccc;
        line-height: 1.6;
        margin-bottom: 3rem;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Sections */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        line-height: 1.2;
    }
    
    /* Button Styling */
    .stButton > button {
        background: #ffffff !important;
        color: #000000 !important;
        border: none !important;
        padding: 1rem 2.5rem !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        width: auto !important;
        margin: 0 auto !important;
        display: block !important;
    }
    
    .stButton > button:hover {
        background: #f0f0f0 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Feature Cards */
    .feature-card {
        background: #111111;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        height: 100%;
    }
    
    .feature-card:hover {
        border-color: #ffffff;
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    .feature-number {
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 1.5rem;
    }
    
    .feature-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
        line-height: 1.3;
    }
    
    .feature-description {
        color: #cccccc;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* CUSTOM FILE UPLOADER STYLING */
    .stFileUploader {
        margin: 2rem 0 !important;
    }
    
    /* Hide the default file uploader label */
    .stFileUploader > label {
        display: none !important;
    }
    
    /* Style the main file uploader container */
    .stFileUploader > div {
        background: #111111 !important;
        border: 2px dashed #444444 !important;
        border-radius: 12px !important;
        padding: 3rem 2rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        position: relative !important;
        min-height: 200px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
    }
    
    .stFileUploader > div:hover {
        border-color: #ffffff !important;
        background: #1a1a1a !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Style the inner content container */
    .stFileUploader > div > div {
        background: transparent !important;
        border: none !important;
        color: #ffffff !important;
        width: 100% !important;
        text-align: center !important;
    }
    
    /* Add custom content before the default content */
    .stFileUploader > div::before {
        content: "📄" !important;
        font-size: 3rem !important;
        display: block !important;
        margin-bottom: 1rem !important;
    }
    
    /* Style the drag and drop text */
    .stFileUploader > div > div > div {
        color: #ffffff !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Style the browse files button */
    .stFileUploader button {
        background: #333333 !important;
        color: #ffffff !important;
        border: 1px solid #555555 !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        transition: all 0.3s ease !important;
        margin-top: 1rem !important;
    }
    
    .stFileUploader button:hover {
        background: #555555 !important;
        border-color: #777777 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Add custom styling for file type info */
    .upload-info {
        color: #888888 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.8rem !important;
        margin-top: 1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }
    
    /* Override Streamlit defaults for dark theme */
    .stSuccess > div,
    .stError > div,
    .stInfo > div,
    .stWarning > div {
        background: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
    }
    
    .stExpander > div {
        background: #111111 !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
    }
    
    /* Answer boxes */
    .answer-box {
        background: #111111;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 2rem;
        margin: 2rem 0;
        color: #ffffff;
        font-size: 1rem;
        line-height: 1.6;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .answer-box.correct {
        border-left-color: #22c55e;
        background: #0a2f0a;
    }
    
    .answer-box.incorrect {
        border-left-color: #ef4444;
        background: #2f0a0a;
    }
    
    .answer-box.partial {
        border-left-color: #f59e0b;
        background: #2f1f0a;
    }
    
    /* Question cards */
    .question-card {
        background: #111111;
        border: 1px solid #333333;
        border-left: 4px solid #ffffff;
        border-radius: 12px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease;
    }
    
    .question-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    }
    
    .question-text {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        line-height: 1.4;
    }
    
    /* Progress bar */
    .progress-container {
        background: #333333;
        height: 6px;
        border-radius: 3px;
        margin: 2rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #ffffff 0%, #cccccc 100%);
        transition: width 0.8s ease;
        border-radius: 3px;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 3rem;
        }
        
        .section-title {
            font-size: 2rem;
        }
        
        .hero-section {
            padding: 2rem 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize processor
@st.cache_resource
def initialize_processor():
    return OllamaProcessor()

# Initialize session state - SIMPLE VERSION
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"
if 'challenge_questions' not in st.session_state:
    st.session_state.challenge_questions = []
if 'current_question_idx' not in st.session_state:
    st.session_state.current_question_idx = 0
if 'challenge_feedback' not in st.session_state:
    st.session_state.challenge_feedback = None
if 'auto_summary' not in st.session_state:
    st.session_state.auto_summary = None

# Initialize processor
processor = initialize_processor()

# SIMPLE PAGE ROUTING
if st.session_state.current_page == "home":
    # Header
    st.markdown("""
    <div class="header">
        <div class="logo">Research Intelligence</div>
        <div class="nav-menu">
            <span>Home</span>
            <span>Upload</span>
            <span>Features</span>
            <span>About</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-subtitle">Issue No. 01 — AI Research Tools</div>
        <h1 class="hero-title">INTELLIGENT<br>DOCUMENT<br>ANALYSIS</h1>
        <div class="hero-description">
            Transform static documents into interactive knowledge systems. 
            Extract insights, generate questions, and test understanding 
            with cutting-edge AI technology.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple button
    if st.button("🚀 Begin Analysis", type="primary"):
        st.session_state.current_page = "upload"
        st.rerun()
    
    st.markdown("---")
    st.markdown('<h2 class="section-title">Features</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-number">01 — Extract</div>
            <div class="feature-title">Text Extraction</div>
            <div class="feature-description">
                Advanced text extraction from PDF, DOCX, and TXT files with 
                intelligent content parsing and structure recognition.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-number">02 — Analyze</div>
            <div class="feature-title">Smart Q&A</div>
            <div class="feature-description">
                Context-aware question answering system powered by local 
                language models with intelligent search capabilities.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-number">03 — Test</div>
            <div class="feature-title">Knowledge Testing</div>
            <div class="feature-description">
                Automated question generation and comprehension testing 
                with detailed feedback and performance evaluation.
            </div>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.current_page == "upload":
    # Prominent back button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Home", type="secondary", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()
    
    st.markdown('<h1 class="section-title">📄 Document Upload</h1>', unsafe_allow_html=True)
    
    # Custom-styled file uploader that's actually functional
    uploaded_file = st.file_uploader(
        "Select Document",
        type=['pdf', 'txt', 'docx'],
        help="Upload PDF, DOCX, or TXT files for analysis",
        label_visibility="hidden"
    )
    
    # Add custom info below the uploader
    st.markdown("""
    <div class="upload-info" style="text-align: center; margin-top: -1rem; margin-bottom: 2rem;">
        PDF • DOCX • TXT • Maximum 200MB
    </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        st.success(f"✅ **File uploaded:** {uploaded_file.name}")
        
        try:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📊 **File size:** {file_size:.1f} MB")
        except:
            st.info("📄 File ready for processing")
        
        if st.button("🔍 Process Document", type="primary"):
            with st.spinner("🤖 Processing document with AI..."):
                try:
                    file_extension = uploaded_file.name.lower().split('.')[-1]
                    
                    if file_extension == 'pdf':
                        extracted_text = processor.extract_text_from_pdf(uploaded_file)
                    elif file_extension == 'docx':
                        extracted_text = processor.extract_text_from_docx(uploaded_file)
                    elif file_extension == 'txt':
                        extracted_text = processor.extract_text_from_txt(uploaded_file)
                    else:
                        st.error(f"❌ Unsupported file type: {file_extension}")
                        extracted_text = ""
                    
                    if extracted_text and extracted_text.strip():
                        st.session_state.extracted_text = extracted_text
                        st.session_state.document_name = uploaded_file.name
                        processor.full_document_text = extracted_text
                        st.session_state.current_page = "analysis"
                        st.rerun()
                    else:
                        st.error("❌ Could not extract text from document")
                except Exception as e:
                    st.error(f"❌ Processing error: {str(e)}")


elif st.session_state.current_page == "analysis":
    st.markdown('<h1 class="section-title">📊 Document Analysis</h1>', unsafe_allow_html=True)
    
    if 'extracted_text' in st.session_state:
        word_count = len(st.session_state.extracted_text.split())
        char_count = len(st.session_state.extracted_text)
        
        st.success("✅ **Text extraction complete**")
        st.info(f"📄 **{st.session_state.document_name}** • {word_count:,} words • {char_count:,} characters")
        
        # Document preview
        with st.expander("📖 **Document Preview** (first 500 characters)"):
            preview = st.session_state.extracted_text[:500] + "..." if len(st.session_state.extracted_text) > 500 else st.session_state.extracted_text
            st.text(preview)
        

        # SUMMARY SECTION WITH ONLY DETAILED OPTION
        st.markdown("### 📝 Document Summary")

        # Show summary if already generated
        if st.session_state.get('auto_summary'):
            summary_data = st.session_state.auto_summary
            summary_text = summary_data.get("summary", "No summary available")
            word_count_summary = summary_data.get("word_count", 0)
            
            st.markdown(f"""
            <div class="answer-box">
                {summary_text}<br>
                <small><strong>Word count:</strong> {word_count_summary}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Regenerate option
            if st.button("🔄 Generate New Summary"):
                st.session_state.auto_summary = None
                st.rerun()

        else:
            # Single detailed summary button
            if st.button("📊 Generate Detailed Summary", type="primary", use_container_width=True):
                with st.spinner("🤖 Generating comprehensive summary..."):
                    try:
                        if not processor.full_document_text:
                            processor.full_document_text = st.session_state.extracted_text
                        auto_summary = processor.generate_auto_summary(max_words=200)  # Increased word limit for detailed
                        st.session_state.auto_summary = auto_summary
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
            # # Speed choice buttons
            # st.write("Choose summary speed:")
            
            # col1, col2, col3 = st.columns(3)
            
            # with col1:
            #     if st.button("⚡⚡⚡ Instant", use_container_width=True, help="1-2 seconds - Basic info"):
            #         with st.spinner("⚡ Generating instant summary..."):
            #             try:
            #                 if not processor.full_document_text:
            #                     processor.full_document_text = st.session_state.extracted_text
            #                 auto_summary = processor.generate_instant_summary()
            #                 st.session_state.auto_summary = auto_summary
            #                 st.rerun()
            #             except Exception as e:
            #                 st.error(f"Error: {e}")
            
            # with col2:
            #     if st.button("⚡⚡ Fast", use_container_width=True, help="2-5 seconds - AI summary"):
            #         with st.spinner("⚡ Generating fast summary..."):
            #             try:
            #                 if not processor.full_document_text:
            #                     processor.full_document_text = st.session_state.extracted_text
            #                 auto_summary = processor.generate_ultra_fast_summary(max_words=100)
            #                 st.session_state.auto_summary = auto_summary
            #                 st.rerun()
            #             except Exception as e:
            #                 st.error(f"Error: {e}")
            
            # with col3:
            #     if st.button("⚡ Detailed", use_container_width=True, help="5-15 seconds - Full analysis"):
            #         with st.spinner("⚡ Generating detailed summary..."):
            #             try:
            #                 if not processor.full_document_text:
            #                     processor.full_document_text = st.session_state.extracted_text
            #                 auto_summary = processor.generate_auto_summary(max_words=150)
            #                 st.session_state.auto_summary = auto_summary
            #                 st.rerun()
            #             except Exception as e:
            #                 st.error(f"Error: {e}")
        
        # Analysis Options (ONLY ONCE!)
        st.markdown("### 🎯 Analysis Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("❓ Q&A Session", use_container_width=True):
                st.session_state.current_page = "questions"
                st.rerun()
        
        with col2:
            if st.button("🧠 Knowledge Test", use_container_width=True):
                st.session_state.current_page = "challenge"
                st.rerun()
        
        with col3:
            if st.button("🔄 New Document", use_container_width=True):
                st.session_state.current_page = "home"
                for key in ['extracted_text', 'document_name', 'auto_summary', 'challenge_questions', 'current_question_idx', 'challenge_feedback']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

elif st.session_state.current_page == "questions" and 'extracted_text' in st.session_state:
    st.markdown('<h1 class="section-title">❓ Question & Answer</h1>', unsafe_allow_html=True)
    
    st.info(f"📄 **Document:** {st.session_state.document_name}")
    
    user_question = st.text_input(
        "**Enter your question:**",
        placeholder="What are the main findings of this document?",
        key="question_input"
    )
    
    use_smart_search = st.checkbox("🔍 Enable smart search", value=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🤖 Get Answer", type="primary"):
            if not user_question.strip():
                st.warning("⚠️ Please enter a question")
            else:
                with st.spinner("🧠 Analyzing..."):
                    try:
                        if not processor.full_document_text:
                            processor.full_document_text = st.session_state.extracted_text
                        
                        if use_smart_search:
                            answer_data = processor.smart_question_answering(user_question)
                        else:
                            context = st.session_state.extracted_text[:15000]
                            answer = processor.chat(user_question, context=context)
                            answer_data = {
                                "answer": answer,
                                "justification": "Based on document analysis",
                                "source_reference": "Document content",
                                "relevant_quote": None
                            }
                        
                        if answer_data and "answer" in answer_data:
                            st.success("✅ **Answer found**")
                            
                            st.markdown("#### 🤖 Response")
                            st.markdown(f"""
                            <div class="answer-box">
                                {answer_data["answer"]}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if answer_data.get("justification"):
                                with st.expander("🧠 **Justification**"):
                                    st.write(answer_data["justification"])
                            
                            if answer_data.get("source_reference"):
                                with st.expander("📚 **Source Reference**"):
                                    st.write(answer_data["source_reference"])
                        else:
                            st.error("❌ Could not generate answer")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if st.button("🏠 Back to Analysis"):
            st.session_state.current_page = "analysis"
            st.rerun()

# In your Streamlit UI code (paste-2.txt), REPLACE the challenge section with this:
elif st.session_state.current_page == "challenge" and 'extracted_text' in st.session_state:
    st.markdown('<h1 class="section-title">🧠 Knowledge Challenge</h1>', unsafe_allow_html=True)
    
    st.info(f"📄 **Document:** {st.session_state.document_name}")
    
    if not st.session_state.challenge_questions:
        st.markdown("### 🎯 Test Your Understanding")
        st.write("Generate AI-powered questions based on your document content")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🎯 Generate Questions", type="primary"):
                with st.spinner("⚡ Creating questions quickly..."):
                    try:
                        if not processor.full_document_text:
                            processor.full_document_text = st.session_state.extracted_text
                        
                        questions = processor.generate_challenge_questions(st.session_state.extracted_text, 3)
                        
                        if questions and len(questions) > 0:
                            valid_questions = [q for q in questions if isinstance(q, dict) and "question" in q]
                            if valid_questions:
                                st.session_state.challenge_questions = valid_questions
                                st.session_state.current_question_idx = 0
                                st.session_state.challenge_feedback = None
                                st.success(f"✅ Generated {len(valid_questions)} questions quickly!")
                                st.rerun()
                            else:
                                st.error("❌ Invalid question format")
                        else:
                            st.error("❌ Could not generate questions")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("🏠 Back to Analysis", key="challenge_back_main"):
                st.session_state.current_page = "analysis"
                st.rerun()
    
    # Display questions
    if st.session_state.challenge_questions:
        questions = st.session_state.challenge_questions
        current_idx = st.session_state.current_question_idx
        current_question = questions[current_idx]
        
        # Progress indicator
        progress = (current_idx + 1) / len(questions)
        st.markdown(f"""
        <div class="progress-container">
            <div class="progress-bar" style="width: {progress * 100}%"></div>
        </div>
        <p style="text-align: center; color: #cccccc; margin: 1rem 0;">
            Question {current_idx + 1} of {len(questions)}
        </p>
        """, unsafe_allow_html=True)
        
        # Question display
        st.markdown(f"""
        <div class="question-card">
            <div class="question-text">{current_question.get('question', 'Question unavailable')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if current_idx > 0:
                if st.button("⬅️ Previous", use_container_width=True):
                    st.session_state.current_question_idx -= 1
                    st.session_state.challenge_feedback = None
                    st.rerun()
        
        with col3:
            if current_idx < len(questions) - 1:
                if st.button("Next ➡️", use_container_width=True):
                    st.session_state.current_question_idx += 1
                    st.session_state.challenge_feedback = None
                    st.rerun()
        
        # Hint
        if current_question.get('hint'):
            st.info(f"💡 **Hint:** {current_question['hint']}")
        
        # Answer input
        user_answer = st.text_area(
            "**Your answer:**",
            key=f"challenge_answer_{current_idx}",
            placeholder="Enter your response...",
            height=100
        )
        
        # Submit and back buttons - SINGLE ROW, NO DUPLICATES
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("✅ Submit Answer", type="primary", use_container_width=True):
                if not user_answer.strip():
                    st.warning("⚠️ Please enter an answer")
                else:
                    with st.spinner("⚡ Evaluating quickly..."):
                        try:
                            evaluation = processor.evaluate_challenge_answer(current_question, user_answer)
                            st.session_state.challenge_feedback = evaluation
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("🔄 New Questions", use_container_width=True):
                st.session_state.challenge_questions = []
                st.session_state.current_question_idx = 0
                st.session_state.challenge_feedback = None
                st.rerun()
        
        with col3:
            if st.button("🏠 Back to Analysis", key="challenge_back_questions", use_container_width=True):
                st.session_state.current_page = "analysis"
                st.rerun()
        
        # Feedback display
        if st.session_state.challenge_feedback:
            feedback = st.session_state.challenge_feedback
            eval_text = feedback.get("evaluation", "").lower()
            
            if "correct" in eval_text and "incorrect" not in eval_text:
                card_class = "correct"
                icon = "🎉"
            elif "incorrect" in eval_text:
                card_class = "incorrect"  
                icon = "❌"
            else:
                card_class = "partial"
                icon = "⚠️"
            
            st.markdown(f"""
            <div class="answer-box {card_class}">
                <strong>{icon} {feedback.get('evaluation', 'No evaluation')}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            if feedback.get("explanation"):
                with st.expander("💭 **Detailed Explanation**"):
                    st.markdown(feedback["explanation"])
            
            if feedback.get("correct_answer"):
                with st.expander("✅ **Correct Answer**"):
                    st.markdown(feedback["correct_answer"])


# Fallback for connection issues
if not processor or not processor.available_models:
    if st.session_state.current_page != "home":
        st.markdown('<h1 class="section-title">❌ Connection Error</h1>', unsafe_allow_html=True)
        st.error("**Ollama Service Unavailable** - Please check your installation and setup")
        
        st.markdown("""
        ### 🔧 Setup Instructions
        
        1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
        2. **Start Service**: Run `ollama serve` in terminal  
        3. **Download Model**: Run `ollama pull llama3.2:3b`
        4. **Refresh Page**: Reload this application
        """)