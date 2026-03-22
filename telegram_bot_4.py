import os
import pickle
import random
import requests
import pandas as pd
import numpy as np
import shap
from typing import TypedDict
from langgraph.graph import StateGraph, END
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    filters, ContextTypes, ConversationHandler, CallbackQueryHandler
)

# ── binary_encode ─────────────────────────────────────────────
def binary_encode(x):
    return (pd.DataFrame(x) == 'Yes').astype(int).values

# ── Load Model ────────────────────────────────────────────────
with open('model/xai_model.pkl', 'rb') as f:
    d = pickle.load(f)

rf            = d['rf_sklearn']
pipe          = d['pipe']
preprocessing = d['preprocessing']
feat_names    = list(d['feat_names'])

try:
    raw_names  = preprocessing.get_feature_names_out()
    feat_names = [n.split('__')[-1] if '__' in n else n for n in raw_names]
except:
    pass

print("Model loaded!")

# ── History Store ─────────────────────────────────────────────
user_history  = {}   # {uid: [results]}
user_language = {}   # {uid: 'hindi' or 'english'}

# ── Language Strings ──────────────────────────────────────────
LANG = {
    'hindi': {
        'start_title'   : 'XAI Loan Approval Bot',
        'start_desc'    : 'Namaste! Main ek AI Agent hun jo aapka loan approve/deny karega aur explain bhi karega kyun!',
        'commands'      : 'Commands',
        'apply_cmd'     : '/apply — Loan apply karo',
        'improve_cmd'   : '/improve — Score simulate karo',
        'history_cmd'   : '/history — Past results',
        'tips_cmd'      : '/tips — Credit score tips',
        'lang_cmd'      : '/language — Bhasha badlo',
        'step'          : 'Step',
        'age_q'         : 'Aapki umar (age) kya hai? (18-80)',
        'income_q'      : 'Salana income kitni hai? (Rs)\n(Example: 50000)',
        'exp_q'         : 'Kaam ka anubhav kitne saal?\n(Example: 3)',
        'loan_q'        : 'Kitna loan chahiye? (Rs)\n(Example: 10000)',
        'rate_q'        : 'Interest rate kitni? (%)\n(Example: 12.5)',
        'credit_q'      : 'Credit score kya hai? (300-850)',
        'home_q'        : 'Ghar ki sthiti?',
        'intent_q'      : 'Loan kis kaam ke liye?',
        'default_q'     : 'Pehle loan default hua hai?',
        'analyzing'     : 'AI Agent analyze kar raha hai...',
        'approved'      : 'LOAN APPROVED!',
        'denied'        : 'LOAN DENIED',
        'meter'         : 'Approval Meter',
        'profile'       : 'Aapka Profile',
        'ai_analysis'   : 'AI Analysis (SHAP)',
        'suggestions'   : 'AI Suggestions (DiCE)',
        'fun_fact'      : 'Fun Fact',
        'dobara'        : '/apply - Dobara try\n/improve - Score badhao\n/history - Past results',
        'powered'       : 'Powered by LangGraph + SHAP + DiCE',
        'no_history'    : 'Koi history nahi! /apply se shuru karo.',
        'past_apps'     : 'Aapki Past Applications',
        'imp_title'     : 'Improve Simulator',
        'imp_last'      : 'Aapka last result',
        'imp_credit_q'  : 'Credit score kitna badhana chahte ho?\n(Example: 50)',
        'imp_loan_q'    : 'Loan kitna kam karna chahte ho? (Rs)\n(0 = same rahega)',
        'imp_before'    : 'Pehle',
        'imp_after'     : 'Baad mein',
        'imp_change'    : 'Change',
        'already_app'   : 'Loan pehle se APPROVED hai! Aur kya chahiye? 😄',
        'cancel_msg'    : 'Cancel ho gaya! /apply se dobara shuru karo.',
        'default_reply' : '/apply se loan check karo!\n/help se commands dekho.',
        'error_num'     : 'Sirf number likho!',
        'error_btn'     : 'Button se select karo!',
        'error_yesno'   : 'Yes ya No select karo!',
        'error_age'     : 'Age 18-80 honi chahiye!',
        'error_credit'  : 'Credit score 300-850 hona chahiye!',
    },
    'english': {
        'start_title'   : 'XAI Loan Approval Bot',
        'start_desc'    : 'Hello! I am an AI Agent that will approve/deny your loan and explain why!',
        'commands'      : 'Commands',
        'apply_cmd'     : '/apply — Apply for loan',
        'improve_cmd'   : '/improve — Simulate score improvement',
        'history_cmd'   : '/history — Past results',
        'tips_cmd'      : '/tips — Credit score tips',
        'lang_cmd'      : '/language — Change language',
        'step'          : 'Step',
        'age_q'         : 'What is your age? (18-80)',
        'income_q'      : 'What is your annual income? (Rs)\n(Example: 50000)',
        'exp_q'         : 'Years of work experience?\n(Example: 3)',
        'loan_q'        : 'How much loan do you need? (Rs)\n(Example: 10000)',
        'rate_q'        : 'What is the interest rate? (%)\n(Example: 12.5)',
        'credit_q'      : 'What is your credit score? (300-850)',
        'home_q'        : 'Home ownership status?',
        'intent_q'      : 'Purpose of loan?',
        'default_q'     : 'Any previous loan defaults?',
        'analyzing'     : 'AI Agent is analyzing...',
        'approved'      : 'LOAN APPROVED!',
        'denied'        : 'LOAN DENIED',
        'meter'         : 'Approval Meter',
        'profile'       : 'Your Profile',
        'ai_analysis'   : 'AI Analysis (SHAP)',
        'suggestions'   : 'AI Suggestions (DiCE)',
        'fun_fact'      : 'Fun Fact',
        'dobara'        : '/apply - Try again\n/improve - Improve score\n/history - Past results',
        'powered'       : 'Powered by LangGraph + SHAP + DiCE',
        'no_history'    : 'No history found! Start with /apply.',
        'past_apps'     : 'Your Past Applications',
        'imp_title'     : 'Improve Simulator',
        'imp_last'      : 'Your last result',
        'imp_credit_q'  : 'How many points to increase credit score?\n(Example: 50)',
        'imp_loan_q'    : 'How much to reduce loan amount? (Rs)\n(0 = keep same)',
        'imp_before'    : 'Before',
        'imp_after'     : 'After',
        'imp_change'    : 'Change',
        'already_app'   : 'Your loan is already APPROVED! What else do you need? 😄',
        'cancel_msg'    : 'Cancelled! Start again with /apply.',
        'default_reply' : 'Use /apply to check loan!\nUse /help for commands.',
        'error_num'     : 'Enter numbers only!',
        'error_btn'     : 'Please select from buttons!',
        'error_yesno'   : 'Select Yes or No!',
        'error_age'     : 'Age must be between 18-80!',
        'error_credit'  : 'Credit score must be 300-850!',
    }
}

def T(uid, key):
    lang = user_language.get(uid, 'hindi')
    return LANG[lang].get(key, key)

# ── Funny Messages ────────────────────────────────────────────
FUNNY = {
    'approved_hindi' : [
        "Bank wale khush hain! Paisa milega!",
        "Cha gaye bhai! Bank ne haan bol diya!",
        "Credit score ne kaam kar diya! Welcome to loan club!",
        "AI ne stamp maar diya - APPROVED! Party karo!",
        "Profile itni strong hai ki bank seedha haan bol diya!",
    ],
    'denied_hindi'   : [
        "Bhai credit score ne bewafa ki tarah dhoka diya!",
        "Bank ne kehna hai - Abhi nahi yaar, thoda aur mehnat karo!",
        "AI bola - Credit score dekha? Main kya karun?",
        "Credit score itna low hai ki bank ne aankhein band kar li!",
        "Chinta mat karo! /improve karo aur wapas aao!",
    ],
    'approved_english': [
        "Congratulations! The bank said YES!",
        "Your credit score did the magic! Welcome to the loan club!",
        "AI has stamped APPROVED! Time to celebrate!",
        "Profile so strong, the bank couldn't say no!",
    ],
    'denied_english'  : [
        "Your credit score played villain today!",
        "Bank says - Not yet buddy, work a little harder!",
        "AI says - Have you seen your credit score? What can I do?",
        "Don't worry! Use /improve and come back stronger!",
    ],
}

def get_funny(uid, pred):
    lang = user_language.get(uid, 'hindi')
    key  = f"{'approved' if pred==1 else 'denied'}_{lang}"
    return random.choice(FUNNY.get(key, FUNNY['denied_hindi']))

# ── Credit Tips ───────────────────────────────────────────────
TIPS_HINDI = """💡 Credit Score Improve Karne Ke Tips

📊 Score Range:
300-579  = Bahut Kharab
580-669  = Theek Hai
670-739  = Accha
740-799  = Bahut Accha
800-850  = Excellent

✅ Score Badhane Ke Tarike:

1. EMI Time Pe Bharo
   Har month time pe payment karo
   Late payment se score girta hai

2. Credit Utilization 30% Se Kam Rakho
   Agar limit Rs10,000 hai
   Rs3,000 se zyada use mat karo

3. Purane Account Band Mat Karo
   Credit history lamba hona chahiye
   Purane cards active rakho

4. Naya Credit Jaldi Mat Lo
   Ek saath kai jagah apply mat karo
   Har inquiry score thoda giraa deti hai

5. Loan Default Mat Karo
   Ek bhi default score ko bahut giraa deta hai
   /improve se simulate karo

⚡ Quick Tips:
- Credit card bill FULL bharo, minimum nahi
- SIP ya savings start karo
- Income badhane ki koshish karo

/apply se loan check karo!"""

TIPS_ENGLISH = """💡 Credit Score Improvement Tips

📊 Score Range:
300-579  = Very Poor
580-669  = Fair
670-739  = Good
740-799  = Very Good
800-850  = Excellent

✅ Ways to Improve Score:

1. Pay EMI On Time
   Pay every month on time
   Late payments reduce score

2. Keep Credit Utilization Below 30%
   If limit is Rs10,000
   Use less than Rs3,000

3. Don't Close Old Accounts
   Longer credit history is better
   Keep old cards active

4. Don't Apply for Too Much Credit
   Don't apply at multiple places at once
   Each inquiry drops score slightly

5. Avoid Loan Defaults
   Even one default drops score badly
   Use /improve to simulate

Quick Tips:
- Pay credit card bill in FULL not minimum
- Start SIP or savings
- Try to increase income

Check loan with /apply!"""

# ── LLM Node (Ollama) ─────────────────────────────────────────
def call_llm(prompt: str) -> str:
    try:
        resp = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model' : 'llama3.1:8b',
                'prompt': prompt,
                'stream': False
            },
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json().get('response', '').strip()
    except:
        pass
    return ""


# ══════════════════════════════════════════════════════════════
# LANGGRAPH AGENT
# ══════════════════════════════════════════════════════════════

class LoanAgentState(TypedDict):
    applicant   : dict
    prediction  : int
    probability : float
    shap_result : list
    suggestions : list
    llm_insight : str
    final_reply : str
    uid         : int

def predict_node(state: LoanAgentState) -> LoanAgentState:
    print("Agent: Predicting...")
    df    = pd.DataFrame([state['applicant']])
    pred  = pipe.predict(df)[0]
    proba = pipe.predict_proba(df)[0][1] * 100
    return {**state, 'prediction': int(pred), 'probability': round(float(proba), 1)}

def shap_node(state: LoanAgentState) -> LoanAgentState:
    print("Agent: SHAP...")
    df     = pd.DataFrame([state['applicant']])
    inp_np = preprocessing.transform(df)
    exp    = shap.TreeExplainer(rf)
    sv     = exp(inp_np)
    vals   = sv.values[0,:,1] if sv.values.ndim == 3 else sv.values[0]
    fn     = feat_names[:len(vals)]
    pairs  = sorted(zip(fn, vals), key=lambda x: abs(x[1]), reverse=True)[:4]
    return {**state, 'shap_result': [
        {'feature': f, 'value': round(float(v), 3),
         'direction': 'Good' if v > 0 else 'Bad',
         'icon': 'OK' if v > 0 else 'XX'}
        for f, v in pairs
    ]}

def suggest_node(state: LoanAgentState) -> LoanAgentState:
    print("Agent: Suggestions...")
    if state['prediction'] == 1:
        return {**state, 'suggestions': []}
    app  = state['applicant']
    sugg = []
    if app['credit_score'] < 650:
        sugg.append(f"Credit score {700 - app['credit_score']} points badhao (700+ karo)")
    if app['loan_percent_income'] > 0.35:
        sugg.append(f"Loan Rs{int(app['person_income']*0.30):,} tak kam karo")
    if app['person_income'] < 40000:
        sugg.append(f"Income badhao (abhi Rs{app['person_income']:,})")
    if app['previous_loan_defaults_on_file'] == 'Yes':
        sugg.append("Pehle ke defaults clear karo")
    if not sugg:
        sugg = ["Credit score 700+ karo", "Loan amount thoda kam karo"]
    return {**state, 'suggestions': sugg[:3]}

def llm_node(state: LoanAgentState) -> LoanAgentState:
    print("Agent: LLM insight...")
    app  = state['applicant']
    pred = state['prediction']
    prob = state['probability']
    uid  = state.get('uid', 0)
    lang = user_language.get(uid, 'hindi')

    if lang == 'hindi':
        prompt = (
            f"Ek loan applicant hai:\n"
            f"Age: {app['person_age']}, Income: Rs{app['person_income']:,}, "
            f"Credit Score: {app['credit_score']}, Loan: Rs{app['loan_amnt']:,}\n"
            f"Result: {'APPROVED' if pred==1 else 'DENIED'} ({prob}%)\n\n"
            f"Sirf 1-2 lines mein simple Hindi mein ek practical financial advice do. "
            f"Koi header ya list mat banao. Seedha advice do."
        )
    else:
        prompt = (
            f"Loan applicant profile:\n"
            f"Age: {app['person_age']}, Income: Rs{app['person_income']:,}, "
            f"Credit Score: {app['credit_score']}, Loan: Rs{app['loan_amnt']:,}\n"
            f"Result: {'APPROVED' if pred==1 else 'DENIED'} ({prob}%)\n\n"
            f"Give 1-2 lines of practical financial advice in simple English. "
            f"No headers or lists. Direct advice only."
        )

    insight = call_llm(prompt)
    return {**state, 'llm_insight': insight}

def reply_node(state: LoanAgentState) -> LoanAgentState:
    print("Agent: Building reply...")
    pred   = state['prediction']
    prob   = state['probability']
    app    = state['applicant']
    uid    = state.get('uid', 0)
    t      = lambda k: T(uid, k)

    filled = int(prob / 10)
    bar    = '|' * filled + '.' * (10 - filled)
    funny  = get_funny(uid, pred)

    shap_lines = "\n".join([
        f"{'OK' if s['icon']=='OK' else 'XX'} {s['feature']}: {s['direction']} ({s['value']:+.3f})"
        for s in state['shap_result']
    ])

    llm_part = ""
    if state.get('llm_insight'):
        llm_part = f"\nAI Insight:\n{state['llm_insight']}\n"

    if pred == 1:
        reply = (
            f"XAI Loan Approval Result\n"
            f"========================\n\n"
            f"{t('approved')}\n\n"
            f"{t('meter')}:\n"
            f"[{bar}] {prob}%\n\n"
            f"{t('profile')}:\n"
            f"Age: {app['person_age']}\n"
            f"Income: Rs{app['person_income']:,}\n"
            f"Credit Score: {app['credit_score']}\n"
            f"Home: {app['person_home_ownership']}\n"
            f"Intent: {app['loan_intent']}\n\n"
            f"{t('ai_analysis')}:\n"
            f"{shap_lines}\n"
            f"{llm_part}\n"
            f"{t('fun_fact')}:\n"
            f"{funny}\n\n"
            f"========================\n"
            f"{t('dobara')}\n"
            f"{t('powered')}"
        )
    else:
        sugg_text = "\n".join([f"- {s}" for s in state['suggestions']])
        reply = (
            f"XAI Loan Approval Result\n"
            f"========================\n\n"
            f"{t('denied')}\n\n"
            f"{t('meter')}:\n"
            f"[{bar}] {prob}%\n\n"
            f"{t('profile')}:\n"
            f"Age: {app['person_age']}\n"
            f"Income: Rs{app['person_income']:,}\n"
            f"Credit Score: {app['credit_score']}\n"
            f"Home: {app['person_home_ownership']}\n"
            f"Intent: {app['loan_intent']}\n\n"
            f"{t('ai_analysis')}:\n"
            f"{shap_lines}\n\n"
            f"{t('suggestions')}:\n"
            f"{sugg_text}\n"
            f"{llm_part}\n"
            f"{t('fun_fact')}:\n"
            f"{funny}\n\n"
            f"========================\n"
            f"{t('dobara')}\n"
            f"{t('powered')}"
        )
    return {**state, 'final_reply': reply}

def build_loan_agent():
    graph = StateGraph(LoanAgentState)
    graph.add_node("predict", predict_node)
    graph.add_node("shap",    shap_node)
    graph.add_node("suggest", suggest_node)
    graph.add_node("llm",     llm_node)
    graph.add_node("reply",   reply_node)
    graph.set_entry_point("predict")
    graph.add_edge("predict", "shap")
    graph.add_edge("shap",    "suggest")
    graph.add_edge("suggest", "llm")
    graph.add_edge("llm",     "reply")
    graph.add_edge("reply",   END)
    return graph.compile()

loan_agent = build_loan_agent()
print("LangGraph Agent ready!")


# ══════════════════════════════════════════════════════════════
# CONVERSATION STATES
# ══════════════════════════════════════════════════════════════
AGE, INCOME, EMP_EXP, LOAN_AMNT, INT_RATE, CREDIT_SCORE, HOME, INTENT, DEFAULT = range(9)
IMP_CREDIT, IMP_LOAN = range(9, 11)


# ── /start ────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    await update.message.reply_text(
        f"🏦 {T(uid, 'start_title')}\n\n"
        f"{T(uid, 'start_desc')}\n\n"
        f"🤖 LangGraph Agent Nodes:\n"
        f"Predict → SHAP → LLM → DiCE → Reply\n\n"
        f"📋 {T(uid, 'commands')}:\n"
        f"{T(uid, 'apply_cmd')}\n"
        f"{T(uid, 'improve_cmd')}\n"
        f"{T(uid, 'history_cmd')}\n"
        f"{T(uid, 'tips_cmd')}\n"
        f"{T(uid, 'lang_cmd')}"
    )

# ── /help ─────────────────────────────────────────────────────
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    await update.message.reply_text(
        f"📖 Help\n\n"
        f"/apply   — Loan check karo\n"
        f"/improve — Score simulate karo\n"
        f"/history — Past 5 results\n"
        f"/tips    — Credit score tips\n"
        f"/language — Hindi/English\n"
        f"/cancel  — Band karo\n\n"
        f"AI Flow:\n"
        f"Predict → SHAP → LLM → DiCE → Reply"
    )

# ── /language ─────────────────────────────────────────────────
async def language_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🇮🇳 Hindi",   callback_data='lang_hindi'),
            InlineKeyboardButton("🇬🇧 English", callback_data='lang_english'),
        ]
    ])
    await update.message.reply_text(
        "🌐 Bhasha chuniye / Choose language:",
        reply_markup=keyboard
    )

async def language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid  = query.from_user.id
    lang = query.data.split('_')[1]
    user_language[uid] = lang
    msg  = "Hindi chunli! /apply se shuru karo." if lang == 'hindi' else "English selected! Start with /apply."
    await query.edit_message_text(f"✅ {msg}")

# ── /tips ─────────────────────────────────────────────────────
async def tips_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid  = update.effective_user.id
    lang = user_language.get(uid, 'hindi')
    await update.message.reply_text(
        TIPS_HINDI if lang == 'hindi' else TIPS_ENGLISH
    )

# ── /apply ────────────────────────────────────────────────────
async def apply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    uid = update.effective_user.id
    await update.message.reply_text(
        f"📝 Loan Application\n\n"
        f"Step 1/9 — {T(uid, 'age_q')}",
        reply_markup=ReplyKeyboardRemove()
    )
    return AGE

async def get_age(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    try:
        age = int(update.message.text)
        if not (18 <= age <= 80):
            await update.message.reply_text(f"❌ {T(uid, 'error_age')}")
            return AGE
        context.user_data['age'] = age
        await update.message.reply_text(
            f"✅ Age: {age}\n\nStep 2/9 — {T(uid, 'income_q')}"
        )
        return INCOME
    except:
        await update.message.reply_text(f"❌ {T(uid, 'error_num')}")
        return AGE

async def get_income(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    try:
        income = int(update.message.text.replace(',', ''))
        context.user_data['income'] = income
        await update.message.reply_text(
            f"✅ Income: Rs{income:,}\n\nStep 3/9 — {T(uid, 'exp_q')}"
        )
        return EMP_EXP
    except:
        await update.message.reply_text(f"❌ {T(uid, 'error_num')}")
        return INCOME

async def get_emp_exp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    try:
        exp = int(update.message.text)
        context.user_data['exp'] = exp
        await update.message.reply_text(
            f"✅ Experience: {exp} years\n\nStep 4/9 — {T(uid, 'loan_q')}"
        )
        return LOAN_AMNT
    except:
        await update.message.reply_text(f"❌ {T(uid, 'error_num')}")
        return EMP_EXP

async def get_loan_amnt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    try:
        loan = int(update.message.text.replace(',', ''))
        context.user_data['loan'] = loan
        await update.message.reply_text(
            f"✅ Loan: Rs{loan:,}\n\nStep 5/9 — {T(uid, 'rate_q')}"
        )
        return INT_RATE
    except:
        await update.message.reply_text(f"❌ {T(uid, 'error_num')}")
        return LOAN_AMNT

async def get_int_rate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    try:
        rate = float(update.message.text)
        context.user_data['rate'] = rate
        await update.message.reply_text(
            f"✅ Rate: {rate}%\n\nStep 6/9 — {T(uid, 'credit_q')}"
        )
        return CREDIT_SCORE
    except:
        await update.message.reply_text(f"❌ {T(uid, 'error_num')}")
        return INT_RATE

async def get_credit_score(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    try:
        credit = int(update.message.text)
        if not (300 <= credit <= 850):
            await update.message.reply_text(f"❌ {T(uid, 'error_credit')}")
            return CREDIT_SCORE
        context.user_data['credit'] = credit
        keyboard = [['RENT', 'OWN'], ['MORTGAGE', 'OTHER']]
        await update.message.reply_text(
            f"✅ Credit: {credit}\n\nStep 7/9 — {T(uid, 'home_q')}",
            reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        )
        return HOME
    except:
        await update.message.reply_text(f"❌ {T(uid, 'error_num')}")
        return CREDIT_SCORE

async def get_home(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid  = update.effective_user.id
    home = update.message.text.upper()
    if home not in ['RENT', 'OWN', 'MORTGAGE', 'OTHER']:
        await update.message.reply_text(f"❌ {T(uid, 'error_btn')}")
        return HOME
    context.user_data['home'] = home
    keyboard = [
        ['PERSONAL', 'EDUCATION'],
        ['MEDICAL', 'VENTURE'],
        ['HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
    ]
    await update.message.reply_text(
        f"✅ Home: {home}\n\nStep 8/9 — {T(uid, 'intent_q')}",
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
    )
    return INTENT

async def get_intent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid    = update.effective_user.id
    intent = update.message.text.upper()
    valid  = ['PERSONAL','EDUCATION','MEDICAL','VENTURE','HOMEIMPROVEMENT','DEBTCONSOLIDATION']
    if intent not in valid:
        await update.message.reply_text(f"❌ {T(uid, 'error_btn')}")
        return INTENT
    context.user_data['intent'] = intent
    keyboard = [['No', 'Yes']]
    await update.message.reply_text(
        f"✅ Intent: {intent}\n\nStep 9/9 — {T(uid, 'default_q')}",
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
    )
    return DEFAULT

async def get_default(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid     = update.effective_user.id
    default = update.message.text
    if default not in ['Yes', 'No']:
        await update.message.reply_text(f"❌ {T(uid, 'error_yesno')}")
        return DEFAULT
    context.user_data['default'] = default

    await update.message.reply_text(
        f"🤖 {T(uid, 'analyzing')}\n\n"
        f"→ Predict Node   ⏳\n"
        f"→ SHAP Node      ⏳\n"
        f"→ LLM Node       ⏳\n"
        f"→ DiCE Node      ⏳\n"
        f"→ Reply Node     ⏳",
        reply_markup=ReplyKeyboardRemove()
    )

    try:
        u         = context.user_data
        income    = u['income']
        loan_amnt = u['loan']

        applicant = {
            'person_age'                     : u['age'],
            'person_income'                  : income,
            'person_emp_exp'                 : u['exp'],
            'loan_amnt'                      : loan_amnt,
            'loan_int_rate'                  : u['rate'],
            'loan_percent_income'            : round(loan_amnt / income, 2),
            'cb_person_cred_hist_length'     : max(1, u['age'] - 20),
            'credit_score'                   : u['credit'],
            'person_home_ownership'          : u['home'],
            'loan_intent'                    : u['intent'],
            'previous_loan_defaults_on_file' : u['default']
        }

        result = loan_agent.invoke({
            'applicant': applicant, 'prediction': 0,
            'probability': 0.0, 'shap_result': [],
            'suggestions': [], 'llm_insight': '',
            'final_reply': '', 'uid': uid
        })

        # Save history
        if uid not in user_history:
            user_history[uid] = []
        user_history[uid].append({
            'age': u['age'], 'credit': u['credit'],
            'income': income, 'loan': loan_amnt,
            'pred': result['prediction'], 'prob': result['probability']
        })
        if len(user_history[uid]) > 5:
            user_history[uid] = user_history[uid][-5:]

        await update.message.reply_text(result['final_reply'])

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}\nDobara /apply karo")

    return ConversationHandler.END


# ── /history ──────────────────────────────────────────────────
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in user_history or not user_history[uid]:
        await update.message.reply_text(T(uid, 'no_history'))
        return

    msg = f"📋 {T(uid, 'past_apps')}:\n\n"
    for i, h in enumerate(reversed(user_history[uid]), 1):
        filled = int(h['prob'] / 10)
        bar    = '|' * filled + '.' * (10 - filled)
        icon   = "APPROVED" if h['pred'] == 1 else "DENIED"
        msg   += (
            f"{i}. {icon}\n"
            f"Age: {h['age']} | Credit: {h['credit']}\n"
            f"Income: Rs{h['income']:,} | Loan: Rs{h['loan']:,}\n"
            f"[{bar}] {h['prob']}%\n\n"
        )
    await update.message.reply_text(msg)


# ── /improve ──────────────────────────────────────────────────
async def improve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in user_history or not user_history[uid]:
        await update.message.reply_text(T(uid, 'no_history'))
        return ConversationHandler.END

    last = user_history[uid][-1]
    if last['pred'] == 1:
        await update.message.reply_text(T(uid, 'already_app'))
        return ConversationHandler.END

    await update.message.reply_text(
        f"🔧 {T(uid, 'imp_title')}\n\n"
        f"{T(uid, 'imp_last')}: DENIED ({last['prob']}%)\n\n"
        f"{T(uid, 'imp_credit_q')}"
    )
    context.user_data['last_app'] = last
    return IMP_CREDIT

async def imp_credit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    try:
        delta = int(update.message.text)
        context.user_data['imp_credit_delta'] = delta
        await update.message.reply_text(
            f"✅ Credit +{delta}\n\n{T(uid, 'imp_loan_q')}"
        )
        return IMP_LOAN
    except:
        await update.message.reply_text(f"❌ {T(uid, 'error_num')}")
        return IMP_CREDIT

async def imp_loan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid        = update.effective_user.id
    try:
        loan_delta = int(update.message.text)
        last       = context.user_data['last_app']
        credit_d   = context.user_data['imp_credit_delta']
        old_credit = last['credit']
        new_credit = min(850, old_credit + credit_d)
        old_loan   = last['loan']
        new_loan   = max(500, old_loan - loan_delta)
        income     = last['income']

        new_applicant = {
            'person_age'                     : last['age'],
            'person_income'                  : income,
            'person_emp_exp'                 : 3,
            'loan_amnt'                      : new_loan,
            'loan_int_rate'                  : 12.0,
            'loan_percent_income'            : round(new_loan / income, 2),
            'cb_person_cred_hist_length'     : max(1, last['age'] - 20),
            'credit_score'                   : new_credit,
            'person_home_ownership'          : 'RENT',
            'loan_intent'                    : 'PERSONAL',
            'previous_loan_defaults_on_file' : 'No'
        }

        df       = pd.DataFrame([new_applicant])
        new_pred = pipe.predict(df)[0]
        new_prob = round(pipe.predict_proba(df)[0][1] * 100, 1)
        old_prob = last['prob']
        diff     = round(new_prob - old_prob, 1)
        result   = "APPROVED" if new_pred == 1 else "DENIED"
        funny    = get_funny(uid, new_pred)
        arrow    = "UP" if diff > 0 else "DOWN"

        old_bar = '|' * int(old_prob/10) + '.' * (10-int(old_prob/10))
        new_bar = '|' * int(new_prob/10) + '.' * (10-int(new_prob/10))

        msg = (
            f"Improve Simulator Result\n\n"
            f"{T(uid, 'imp_before')}:\n"
            f"[{old_bar}] {old_prob}% DENIED\n"
            f"Credit: {old_credit} | Loan: Rs{old_loan:,}\n\n"
            f"{T(uid, 'imp_after')}:\n"
            f"[{new_bar}] {new_prob}% {result}\n"
            f"Credit: {new_credit} | Loan: Rs{new_loan:,}\n\n"
            f"{T(uid, 'imp_change')}: {diff:+.1f}% ({arrow})\n\n"
            f"{funny}\n\n"
            f"/apply - Real application\n"
            f"/history - Past results"
        )
        await update.message.reply_text(msg)

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    await update.message.reply_text(
        T(uid, 'cancel_msg'),
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END

async def default_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    await update.message.reply_text(T(uid, 'default_reply'))


# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    TOKEN = os.environ.get('TELEGRAM_TOKEN', '8663010744:AAEXs29ahTYmKHC43xa3VdkykbtfwE5MLVo')

    bot = ApplicationBuilder().token(TOKEN).build()

    apply_handler = ConversationHandler(
        entry_points=[CommandHandler('apply', apply)],
        states={
            AGE         : [MessageHandler(filters.TEXT & ~filters.COMMAND, get_age)],
            INCOME      : [MessageHandler(filters.TEXT & ~filters.COMMAND, get_income)],
            EMP_EXP     : [MessageHandler(filters.TEXT & ~filters.COMMAND, get_emp_exp)],
            LOAN_AMNT   : [MessageHandler(filters.TEXT & ~filters.COMMAND, get_loan_amnt)],
            INT_RATE    : [MessageHandler(filters.TEXT & ~filters.COMMAND, get_int_rate)],
            CREDIT_SCORE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_credit_score)],
            HOME        : [MessageHandler(filters.TEXT & ~filters.COMMAND, get_home)],
            INTENT      : [MessageHandler(filters.TEXT & ~filters.COMMAND, get_intent)],
            DEFAULT     : [MessageHandler(filters.TEXT & ~filters.COMMAND, get_default)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    improve_handler = ConversationHandler(
        entry_points=[CommandHandler('improve', improve)],
        states={
            IMP_CREDIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, imp_credit)],
            IMP_LOAN  : [MessageHandler(filters.TEXT & ~filters.COMMAND, imp_loan)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    bot.add_handler(CommandHandler("start",    start))
    bot.add_handler(CommandHandler("help",     help_cmd))
    bot.add_handler(CommandHandler("history",  history))
    bot.add_handler(CommandHandler("tips",     tips_cmd))
    bot.add_handler(CommandHandler("language", language_cmd))
    bot.add_handler(CallbackQueryHandler(language_callback, pattern='^lang_'))
    bot.add_handler(apply_handler)
    bot.add_handler(improve_handler)
    bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, default_msg))

    print("Bot + LangGraph Agent starting!")
    bot.run_polling()
