# 🏦 XAI Loan Approval System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?style=for-the-badge&logo=streamlit)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent-green?style=for-the-badge)
![Telegram](https://img.shields.io/badge/Telegram-Bot-blue?style=for-the-badge&logo=telegram)
![Render](https://img.shields.io/badge/Render-Deployed-purple?style=for-the-badge)

**An Explainable AI system for Loan Approval Prediction with LangGraph Agent, SHAP, LIME, DiCE, and Telegram Bot Integration**

[🌐 Live Demo](https://xai-loan-approval.onrender.com) • [🤖 Telegram Bot](https://t.me/xai_loan_approval_bot) • [📊 GitHub](https://github.com/Yashshakya1)

</div>

---

## 📌 Project Overview

This project builds a complete **Explainable AI (XAI) pipeline** for loan approval prediction. Unlike black-box ML models, this system explains *why* a loan is approved or denied using three XAI techniques — SHAP, LIME, and DiCE — and exposes the full pipeline via a Streamlit dashboard deployed on Render, and a **LangGraph AI Agent** integrated into a Telegram chatbot.

---

## 🚀 Live Deployments

| Platform | Link | Description |
|----------|------|-------------|
| 🌐 **Streamlit App** | [xai-loan-approval.onrender.com](https://xai-loan-approval.onrender.com) | Interactive dashboard |
| 🤖 **Telegram Bot** | [@xai_loan_approval_bot](https://t.me/xai_loan_approval_bot) | AI Agent chatbot |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   XAI Loan System                   │
├─────────────────┬───────────────────────────────────┤
│  Streamlit App  │         Telegram Bot               │
│  (Render)       │         (Render)                   │
├─────────────────┴───────────────────────────────────┤
│              LangGraph AI Agent                      │
│  Predict → SHAP → LLM → DiCE → Reply               │
├─────────────────────────────────────────────────────┤
│           Random Forest Model (from scratch)         │
│              + Sklearn Pipeline                      │
├──────────────┬──────────────┬───────────────────────┤
│     SHAP     │     LIME     │        DiCE            │
│  (Global +   │   (Local     │  (Counterfactual       │
│   Local)     │  Explanation)│   Explanations)        │
└──────────────┴──────────────┴───────────────────────┘
```

---

## ✨ Key Features

### 🔮 ML Model
- **Random Forest from Scratch** — Built using pure Python (Node, DecisionTree, RandomForest classes)
- **Sklearn RandomForestClassifier** — For production pipeline
- **Dataset** — 45,000 loan applications
- **Accuracy: 92.8%** | Precision: 90.3% | Recall: 75.8% | F1: 82.4% | ROC-AUC: 97.5%

### 🔍 XAI Techniques
| Technique | Type | Purpose |
|-----------|------|---------|
| **SHAP** | Global + Local | Feature importance & contribution |
| **LIME** | Local | Why this specific prediction? |
| **DiCE** | Counterfactual | What to change to get approved? |

### 🤖 LangGraph AI Agent
```
User Input
    ↓
Predict Node  →  Loan decision (Approved/Denied)
    ↓
SHAP Node     →  Why this decision? (Feature analysis)
    ↓
LLM Node      →  Practical financial advice (Ollama/LLaMA 3.1)
    ↓
DiCE Node     →  What to improve? (Actionable suggestions)
    ↓
Reply Node    →  Formatted final response
```

### 🌐 Streamlit Dashboard
- **5 Pages** — Dashboard, Predict & Explain, SHAP Analysis, LIME, DiCE Counterfactuals
- **Premium UI** — Glass-morphism dark theme, Plotly charts
- **Top Navigation** — Session-state based, sidebar independent

### 🤖 Telegram Bot Features
- **/apply** — Step-by-step loan application (9 steps with buttons)
- **/improve** — Credit score improvement simulator
- **/history** — Past 5 applications tracking
- **/tips** — Credit score improvement tips
- **/language** — Hindi / English support
- **Approval Meter** — Visual progress bar `[||||......] 41.8%`
- **Fun Messages** — Random funny responses
- **AI Insight** — LLM-powered personalized advice

---

## 🛠️ Tech Stack

```python
# Core ML
scikit-learn==1.6.1      # RandomForestClassifier, Pipeline
numpy==2.3.5             # Numerical operations
pandas==2.3.3            # Data manipulation

# XAI
shap                     # SHAP explanations
lime                     # LIME explanations
dice-ml                  # DiCE counterfactuals

# AI Agent
langgraph                # Graph-based AI agent
langchain                # LLM integration

# LLM
ollama                   # Local LLM (LLaMA 3.1 8B)

# Web App
streamlit                # Dashboard UI
plotly                   # Interactive charts

# Telegram Bot
python-telegram-bot      # Telegram integration

# Deployment
render                   # Cloud deployment
```

---

## 📁 Project Structure

```
XAI-Loan-Approval-System/
├── app_5_3_2.py              # Streamlit dashboard
├── telegram_bot_4.py         # Telegram bot with LangGraph
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── model/
│   └── xai_model.pkl         # Trained model + preprocessor
└── notebook/
    └── Counterfactual_Explanations___XAI_Prediction_System.ipynb
```

---

## ⚙️ Installation & Setup

### 1. Clone the repo
```bash
git clone https://github.com/Yashshakya1/XAI-Loan-Approval-System.git
cd XAI-Loan-Approval-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App
```bash
streamlit run app_5_3_2.py
```

### 4. Run Telegram Bot
```bash
export TELEGRAM_TOKEN="your_token_here"
python telegram_bot_4.py
```

### 5. (Optional) Setup Ollama for LLM
```bash
brew install ollama
ollama pull llama3.1:8b
ollama serve
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **92.8%** |
| Precision | **90.3%** |
| Recall | **75.8%** |
| F1 Score | **82.4%** |
| ROC-AUC | **97.5%** |

---

## 🎯 Features Used

| Feature | Type | Description |
|---------|------|-------------|
| person_age | Numerical | Applicant age |
| person_income | Numerical | Annual income |
| person_emp_exp | Numerical | Employment experience |
| loan_amnt | Numerical | Loan amount requested |
| loan_int_rate | Numerical | Interest rate |
| loan_percent_income | Numerical | Loan as % of income |
| cb_person_cred_hist_length | Numerical | Credit history length |
| credit_score | Numerical | Credit score (300-850) |
| person_home_ownership | Categorical | RENT/OWN/MORTGAGE |
| loan_intent | Categorical | Purpose of loan |
| previous_loan_defaults_on_file | Binary | Past defaults |

*Note: person_gender and person_education removed (fairness + feature importance analysis)*

---

## 🤖 LangGraph Agent Flow

```python
class LoanAgentState(TypedDict):
    applicant   : dict    # User input
    prediction  : int     # 0=Denied, 1=Approved
    probability : float   # Approval probability
    shap_result : list    # SHAP feature analysis
    suggestions : list    # DiCE improvements
    llm_insight : str     # LLM financial advice
    final_reply : str     # Formatted response

# Agent Graph
START → predict_node → shap_node → llm_node → suggest_node → reply_node → END
```

---

## 🌐 Deployment

### Streamlit App on Render
```
Build Command : pip install -r requirements.txt
Start Command : streamlit run app_5_3_2.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

### Telegram Bot on Render
```
Build Command : pip install -r requirements.txt
Start Command : python telegram_bot_4.py
Environment   : TELEGRAM_TOKEN = <your_token>
```

---

## 📱 Telegram Bot Demo

```
User: /apply

Bot:  Step 1/9 — Aapki umar kya hai?
User: 25

Bot:  Step 6/9 — Credit score?
User: 650

Bot:  🤖 AI Agent analyze kar raha hai...
      → Predict Node   ⏳
      → SHAP Node      ⏳
      → LLM Node       ⏳
      → DiCE Node      ⏳
      → Reply Node     ⏳

Bot:  XAI Loan Approval Result
      ========================
      LOAN DENIED
      
      Approval Meter:
      [||||......] 41.8%
      
      AI Analysis (SHAP):
      OK previous_default: Good (+0.170)
      XX loan_percent_income: Bad (-0.041)
      
      AI Suggestions (DiCE):
      - Credit score 50 points badhao
      - Loan Rs15,000 tak kam karo
      
      AI Insight:
      Focus on paying existing EMIs on time
      to improve your credit score gradually.
      
      Fun Fact:
      Bank ne kehna hai - Abhi nahi yaar!
```

---

## 👨‍💻 About the Developer

**Yash Shakya**
- 🎓 B.Tech CSE — Surabhi College of Engineering & Technology, RGPV (2027)
- 🏅 IIT Guwahati AI/ML Certification (2024)
- 🔗 [LinkedIn](https://linkedin.com/in/yash-shakya-71bab72b5)
- 💻 [GitHub](https://github.com/Yashshakya1)

---

## 📄 License

MIT License — feel free to use and modify!

---

<div align="center">

⭐ **Star this repo if you found it useful!** ⭐

Built with ❤️ by Yash Shakya | Powered by LangGraph + SHAP + DiCE + LLaMA

</div>
