# 📄 MCQ Generation System with AI

This project implements an AI-driven Multiple Choice Question (MCQ) generation system, utilizing **Streamlit**, **FAISS**, **LangChain**, and **Hugging Face**. The system extracts content from uploaded PDF documents to generate relevant MCQs, providing an excellent tool for learning and assessment.

## 🚀 Live Demo

You can try out the live demo here:  
[![Open in Streamlit]([https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://document-question-answering-template.streamlit.app/](https://ai-mcqgenerator.streamlit.app/))

## ✨ Features

- **Upload PDFs**: Upload any PDF document for question generation.
- **Customizable Question Count**: Select the number of questions (5 to 20).
- **Difficulty Levels**: Choose from Easy, Medium, or Hard MCQs.
- **Multiple Topics**: Supports MCQ generation from various topics within the document.
- **Interactive UI**: Navigate through generated MCQs and review them easily.
- **Export Options**: Download the MCQs as a PDF for offline use.
- **Hugging Face Integration**: Utilizes state-of-the-art NLP models from Hugging Face for question generation.

## 🛠️ How to Run Locally

Follow these steps to run the app on your local machine.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd document-question-answering

pip install -r requirements.txt

streamlit run bot.py
