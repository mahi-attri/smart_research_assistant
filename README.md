# Research Intelligence 

An AI-powered document analysis platform that transforms static documents into interactive knowledge systems. Extract insights, generate questions and test understanding using cutting-edge local AI technology.


## Features

- **📄 Multi-Format Document Processing**: Extract and analyze text from PDF, DOCX, and TXT files
- **🤖 Local AI Integration**: Powered by Ollama for privacy-focused, offline AI processing  
- **📊 Intelligent Summarization**: Generate comprehensive document summaries with customizable detail levels
- **❓ Context-Aware Q&A**: Ask questions about your documents with smart search and contextual understanding
- **🧠 Knowledge Testing**: Auto-generate comprehension questions with intelligent evaluation
- **🎯 Interactive Learning**: Get detailed feedback on answers with justifications and suggestions

## Architecture Overview

### Component Breakdown

#### 1. **Frontend Layer (Streamlit UI)**
- **Multi-page navigation**: Home, Upload, Analysis, Q&A, Challenge
- **Responsive design**: Dark theme with modern styling
- **Real-time feedback**: Progress indicators and status updates
- **Interactive elements**: File upload, question input, answer evaluation

#### 2. **Backend Layer (OllamaProcessor)**
- **Document Processing**: Text extraction and preprocessing
- **AI Orchestration**: Manages interactions with Ollama models
- **Context Management**: Maintains conversation history and document state
- **Question Generation**: Creates contextual questions based on document content

#### 3. **AI Engine (Ollama Service)**
- **Local Processing**: Privacy-focused, no data sent to external services
- **Model Management**: Supports multiple LLM models (qwen2:1.5b, llama3.2:3b, llama3.1:8b)
- **Inference Optimization**: Efficient prompt engineering and response handling
