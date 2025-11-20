## Requirements
- Python 3.10+

# X-ray AI Agent Demo  
FastAPI + Pinecone RAG + Spectral Analysis

This demo implements an AI assistant for X-ray spectroscopy using:
- FastAPI backend
- OpenAI LLMs (summary + CoT)
- Pinecone vector database for RAG
- CSV spectral upload + normalization + peak detection
- Front-end interface with real-time plotting (energy vs intensity)

---

## 🚀 Features

### 📡 X-ray Spectrum Analysis
- CSV upload
- Normalization
- Peak detection
- Scientific summary generation
- Short chain-of-thought reasoning
- Auto JSON parsing
- Interactive plotting

### 🤖 Chat Assistant (RAG + CoT)
- Pinecone vector retrieval
- Source similarity scoring
- LLM answer + reasoning steps
- Clean, structured JSON output

---

## 🛠 Tech Stack
- Python 3.10+
- FastAPI
- OpenAI SDK
- Pinecone
- NumPy / SciPy
- Frontend: HTML + JS + Plotly

---

## 🏁 Run Locally

### 1. Clone
```bash
git clone https://github.com/YOURNAME/xray-ai-agent-demo.git
cd xray-ai-agent-demo
```
### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Set environment variables
Create .env:
```bash
OPENAI_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
```
### 5. Start backend
```bash
uvicorn backend.main:app --reload
```
### 6. Open frontend
```bash
http://127.0.0.1:8000
```
### 7. Run analysis
The example xRay data is provided, feel free to test the function with it.