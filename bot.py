import os
import requests
import time
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langdetect import detect
from engine import get_answer

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
# The token is now pulled securely from the environment
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("❌ Error: TELEGRAM_BOT_TOKEN not found in .env file!")

URL = f"https://api.telegram.org/bot{TOKEN}"

def main():
    offset = None
    print("🤖 AgriScribe V3 (Universal Edition) is Live...")
    print("🔐 Security: Bot token loaded from environment.")
    
    while True:
        try:
            # Short polling for updates
            res = requests.get(f"{URL}/getUpdates", params={"offset": offset, "timeout": 20}).json()
            if not res.get("ok"): 
                continue

            for update in res.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                text = msg.get("text")
                chat_id = msg.get("chat", {}).get("id")

                if text:
                    # Detect input language
                    try:
                        user_lang = detect(text)
                    except:
                        user_lang = 'en'
                    
                    print(f"📩 Input: {text} | Language: {user_lang}")

                    # 1. Input Translation (Farmer -> English Search)
                    # Translation ensures cross-lingual retrieval works in engine.py
                    try:
                        query_en = GoogleTranslator(source='auto', target='en').translate(text) if user_lang != 'en' else text
                    except Exception as te:
                        print(f"⚠️ Translation Error (Input): {te}")
                        query_en = text

                    # 2. Engine Retrieval & Reasoning
                    ans_en = get_answer(query_en)
                    
                    # 3. Output Handling (Split Answer from Sources to avoid translating technical paths)
                    if "📚 **Sources:**" in ans_en:
                        parts = ans_en.split("📚 **Sources:**")
                        advice_en = parts[0].strip()
                        sources = "\n\n📚 **Sources:**" + parts[1]
                    else:
                        advice_en = ans_en
                        sources = ""

                    # 4. Output Translation (English -> Farmer's detected language)
                    if user_lang != 'en':
                        try:
                            final_advice = GoogleTranslator(source='en', target=user_lang).translate(advice_en)
                        except Exception as te:
                            print(f"⚠️ Translation Error (Output): {te}")
                            final_advice = advice_en
                    else:
                        final_advice = advice_en

                    # 5. Final Assembly
                    final_msg = f"{final_advice}{sources}"

                    # 6. Send message to user
                    requests.post(f"{URL}/sendMessage", data={"chat_id": chat_id, "text": final_msg})
                    print(f"📤 Response Sent in {user_lang}")

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Connection Error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"⚠️ Unexpected Bot Error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()