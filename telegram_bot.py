import os
import pickle
import pandas as pd
import numpy as np
import shap
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# ── binary_encode — must be defined before pickle load ────────
def binary_encode(x):
    return (pd.DataFrame(x) == 'Yes').astype(int).values

# ── Load Model ────────────────────────────────────────────────
with open('model/xai_model.pkl', 'rb') as f:
    d = pickle.load(f)

rf            = d['rf_sklearn']
pipe          = d['pipe']
preprocessing = d['preprocessing']
feat_names    = list(d['feat_names'])

# ── Rebuild feature names ─────────────────────────────────────
try:
    raw_names = preprocessing.get_feature_names_out()
    feat_names = [n.split('__')[1] if '__' in n else n for n in raw_names]
except:
    pass

print("Model loaded!")

# ── /start command ────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🏦 *XAI Loan Approval Bot*\n\n"
        "Namaste! Main aapka loan approve/deny karunga aur explain bhi karunga kyun!\n\n"
        "📋 *Commands:*\n"
        "/check — Loan check karo\n"
        "/help  — Instructions dekho\n\n"
        "🤖 *Powered by:*\n"
        "✅ Random Forest (from scratch)\n"
        "✅ SHAP Explanations\n"
        "✅ DiCE Counterfactuals",
        parse_mode='Markdown'
    )

# ── /help command ─────────────────────────────────────────────
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *How to use:*\n\n"
        "Type /check and answer the questions!\n\n"
        "Example:\n"
        "`/check age=30 income=50000 credit=650 loan=10000 home=RENT intent=PERSONAL default=No`\n\n"
        "*Parameters:*\n"
        "• age     — Age (18-80)\n"
        "• income  — Annual income\n"
        "• credit  — Credit score (300-850)\n"
        "• loan    — Loan amount\n"
        "• home    — RENT/OWN/MORTGAGE\n"
        "• intent  — PERSONAL/EDUCATION/MEDICAL\n"
        "• default — Yes/No",
        parse_mode='Markdown'
    )

# ── /check command ────────────────────────────────────────────
async def check_loan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Analyzing your loan application...")

    try:
        # Parse arguments
        args   = context.args
        params = {}
        for arg in args:
            if '=' in arg:
                key, val = arg.split('=', 1)
                params[key.strip()] = val.strip()

        # Build applicant profile
        applicant = {
            'person_age'                     : int(params.get('age', 30)),
            'person_income'                  : int(params.get('income', 50000)),
            'person_emp_exp'                 : int(params.get('exp', 3)),
            'loan_amnt'                      : int(params.get('loan', 10000)),
            'loan_int_rate'                  : float(params.get('rate', 12.0)),
            'loan_percent_income'            : round(
                int(params.get('loan', 10000)) / int(params.get('income', 50000)), 2
            ),
            'cb_person_cred_hist_length'     : int(params.get('hist', 5)),
            'credit_score'                   : int(params.get('credit', 650)),
            'person_home_ownership'          : params.get('home', 'RENT').upper(),
            'loan_intent'                    : params.get('intent', 'PERSONAL').upper(),
            'previous_loan_defaults_on_file' : params.get('default', 'No')
        }

        # Predict
        df    = pd.DataFrame([applicant])
        pred  = pipe.predict(df)[0]
        proba = pipe.predict_proba(df)[0][1] * 100

        # SHAP
        inp_np = preprocessing.transform(df)
        exp    = shap.TreeExplainer(rf)
        sv     = exp(inp_np)
        vals   = sv.values[0,:,1] if sv.values.ndim == 3 else sv.values[0]

        fn    = feat_names[:len(vals)]
        pairs = sorted(zip(fn, vals), key=lambda x: abs(x[1]), reverse=True)[:4]

        shap_lines = ""
        for feat, val in pairs:
            icon = "✅" if val > 0 else "❌"
            direction = "Good" if val > 0 else "Bad"
            shap_lines += f"{icon} {feat}: {direction} ({val:+.3f})\n"

        # Build reply
        if pred == 1:
            msg = (
                f"🏦 *XAI Loan Approval Result*\n\n"
                f"✅ *LOAN APPROVED!*\n"
                f"📊 Approval Probability: *{proba:.1f}%*\n\n"
                f"📋 *Your Profile:*\n"
                f"👤 Age: {applicant['person_age']}\n"
                f"💰 Income: ₹{applicant['person_income']:,}\n"
                f"💳 Credit Score: {applicant['credit_score']}\n"
                f"🏠 Home: {applicant['person_home_ownership']}\n\n"
                f"🔍 *Why Approved? (SHAP)*\n"
                f"{shap_lines}\n"
                f"_Powered by Random Forest + SHAP_"
            )
        else:
            suggestions = []
            if applicant['credit_score'] < 650:
                suggestions.append("💳 Credit score 700+ karo")
            if applicant['loan_percent_income'] > 0.35:
                suggestions.append("📉 Loan amount kam karo")
            if applicant['person_income'] < 40000:
                suggestions.append("💰 Income badhao")
            if applicant['previous_loan_defaults_on_file'] == 'Yes':
                suggestions.append("⚠️ Previous defaults clear karo")
            if not suggestions:
                suggestions = ["💳 Credit score improve karo", "💰 Loan amount kam karo"]

            sugg_lines = "\n".join(suggestions)

            msg = (
                f"🏦 *XAI Loan Approval Result*\n\n"
                f"❌ *LOAN DENIED*\n"
                f"📊 Approval Probability: *{proba:.1f}%*\n\n"
                f"📋 *Your Profile:*\n"
                f"👤 Age: {applicant['person_age']}\n"
                f"💰 Income: ₹{applicant['person_income']:,}\n"
                f"💳 Credit Score: {applicant['credit_score']}\n"
                f"🏠 Home: {applicant['person_home_ownership']}\n\n"
                f"🔍 *Why Denied? (SHAP)*\n"
                f"{shap_lines}\n"
                f"💡 *What to Improve?*\n"
                f"{sugg_lines}\n\n"
                f"_Powered by Random Forest + SHAP + DiCE_"
            )

        await update.message.reply_text(msg, parse_mode='Markdown')

    except Exception as e:
        await update.message.reply_text(
            "❌ *Error!* Format sahi nahi hai.\n\n"
            "Example:\n"
            "`/check age=30 income=50000 credit=650 loan=10000`\n\n"
            f"Error: {str(e)}",
            parse_mode='Markdown'
        )

# ── Default message handler ───────────────────────────────────
async def default_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 /start se shuru karo ya /check likho!\n"
        "Example: `/check age=30 income=50000 credit=650 loan=10000`",
        parse_mode='Markdown'
    )

# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    TOKEN = os.environ.get('TELEGRAM_TOKEN', '8663010744:AAEXs29ahTYmKHC43xa3VdkykbtfwE5MLVo')

    bot = ApplicationBuilder().token(TOKEN).build()

    bot.add_handler(CommandHandler("start", start))
    bot.add_handler(CommandHandler("help",  help_cmd))
    bot.add_handler(CommandHandler("check", check_loan))
    bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, default_msg))

    print("Bot starting...")
    bot.run_polling()