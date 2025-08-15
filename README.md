# Retrieval-Augmented Generation (RAG) System

This project implements an advanced **Retrieval-Augmented Generation (RAG)** framework with **multimodal processing**, leveraging **Hugging Face** models and modern AI techniques.

The system supports:
- **Basic RAG**: Simple PDF ingestion, Hugging Face model integration, and query handling.
- **Advanced RAG**: Dense Passage Retrieval (DPR), intelligent ranking, and optimized performance.
- **Multimodal RAG**: Combined text & image understanding using **OpenCLIP** for vision-language tasks.

---

## Project Structure

├── 1.RAG and Hugging Face/  
│ ├── 2024_Annual_Report.pdf(of LT Finance) # Sample document for RAG processing  
│ ├── app.ipynb # Main application notebook  
│ └── huggingface.ipynb # Hugging Face integration examples  
├── 2.AdvancedRAG/  
│ ├── dpr.py # Dense Passage Retrieval implementation  
│ ├── investor-presentation.pdf(of Microsoft) # Business document sample   
│ └── rag_ranking.py # RAG ranking & scoring algorithms  
├── 3.MULTI MODAL RAG/    
│ ├── data/ # Processed data storage  
│ ├── Images/ # Image assets for multimodal processing   
│ ├── multi_ad.py # Multimodal ad content processing  
│ └── multimodal.py # Core multimodal RAG implementation 
└── requirements.txt # Project dependencies  


---

## Installation

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```


---

## Usage Workflow

### **Basic RAG & Hugging Face**
```bash
cd "1.RAG and Hugging Face"
```
**What you will see:**
- Document ingestion and PDF processing
- Hugging Face model setup
- Simple query & retrieval system
- Fundamental RAG concepts in action

**Key Files:**
- `app.ipynb` — Main application with interactive examples
- `huggingface.ipynb` — Hugging Face integration walkthroughs
- `2024_Annual_Report.pdf` — Sample input document

---

### **Advanced RAG**
```bash
cd "../2.AdvancedRAG"
python dpr.py
python rag_ranking.py
```
**What you will see:**
- **Dense Passage Retrieval (DPR)** for better accuracy
- Advanced ranking algorithms for relevant results
- Handling of complex business documents
- Performance tuning strategies

**Key Files:**
- `dpr.py` — DPR implementation
- `rag_ranking.py` — Ranking & scoring logic
- `investor-presentation.pdf` — Complex document example

---

### **Multimodal RAG**
```bash
cd "../3.MULTI MODAL RAG"
python multimodal.py
python multi_ad.py
```
**What you will see:**
- Combined **text & image understanding** with OpenCLIP
- Multimodal embeddings for search & retrieval
- Cross-modal query capabilities
- Advertisement content analysis & AI insights

**Key Files:**
- `multimodal.py` — Core multimodal RAG logic
- `multi_ad.py` — Ad-specific multimodal analysis
- `Images/` — Sample test images
- `data/` — Processed multimodal data

---

## Features by Module

### **Basic RAG (1.RAG and Hugging Face)**
- PDF/Text document ingestion
- Hugging Face transformer integration
- Interactive RAG demonstrations
- Simple Q&A retrieval

### **Advanced RAG (2.AdvancedRAG)**
- Dense Passage Retrieval (DPR)
- Intelligent scoring & ranking
- Complex document processing
- Optimized performance

### **Multimodal RAG (3.MULTI MODAL RAG)**
- Image + text embeddings with OpenCLIP
- Cross-modal retrieval (ask in text, get related images and vice versa)
- Multimodal advertisement analysis
- Powerful AI-assisted content understanding

---
