import os
import json
import random
import time
from tqdm import tqdm
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# 🔴 PASTE YOUR DEEPSEEK API KEY HERE
API_KEY = "..."

# 🔴 INPUT/OUTPUT
INPUT_FOLDER = "./raw_data"        # Folder with your .txt/.html files
OUTPUT_FILE = "barq_universal_finetune.jsonl"

# 🔴 SETTINGS
NEGATIVE_RATIO = 0.2               # 20% "I don't know" examples
MODEL_NAME = "deepseek-chat"       # DeepSeek V3 ($0.27/1M tokens)

# ==========================================
# 🧠 THE SYSTEM PROMPT (UNIVERSAL PERSONA)
# ==========================================
SYSTEM_PROMPT = """
You are 'Barq' (برق), a smart AI assistant.
You are reading a specific document to answer the user's question.

Your Goal:
Generate a training example with an ENGLISH 'Chain of Thought' (<think>) and an ARABIC 'Final Answer'.

Output JSON Format:
{
  "thought_process": "<think> [Analyze User Intent] -> [Check Releted Context] -> [Think About Answer] </think>",
  "final_response": "[The response in clear, professional Arabic]"
}

Rules:
1. THINKING MUST BE IN ENGLISH. (This helps the small model reason better).
2. FINAL ANSWER MUST BE IN ARABIC.
   - Style: Simple, Informative, Direct and comprehensive.
   - Avoid flowery language like "Dear User" or "Honored Guest".
3. Source: Use ONLY the provided Context.
4. Refusal: If the Context does not contain the answer, your Final Answer MUST be exactly:
   "عذراً، لا أملك معلومات حول هذا السؤال في الوقت الحالي."
   (Sorry, I do not have information about this question at the moment.)
"""

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

def get_deepseek_completion(context, question, is_negative=False):
    """
    Robust API Call with Exponential Backoff
    """
    if is_negative:
        # TRICK QUESTION LOGIC
        user_content = f"""
        CONTEXT:
        {context[:2000]}...

        USER QUESTION:
        "{question}"

        INSTRUCTION:
        The user's question is unrelated to the context.
        1. Generate a <think> block explaining there is no match.
        2. Set 'final_response' to the refusal message defined in System Prompt.
        """
    else:
        # STANDARD LOGIC
        user_content = f"""
        CONTEXT:
        {context[:3000]}

        USER QUESTION:
        "{question}"

        INSTRUCTION:
        Answer the question using ONLY the context above.
        1. Generate a <think> block citing the specific text used.
        2. Provide a helpful, direct answer in clear Arabic.
        """

    # Robust Retry Loop
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"}, 
                temperature=0.7,
                timeout=45
            )
            content = response.choices[0].message.content
            if not content: return None
            return json.loads(content)

        except (RateLimitError, APIConnectionError, APIStatusError) as e:
            wait = 2 ** attempt
            print(f"⚠️ Network Error: Retrying in {wait}s...")
            time.sleep(wait)
        except json.JSONDecodeError:
            print("⚠️ JSON Error. Skipping chunk.")
            return None
        except Exception as e:
            print(f"❌ Critical Error: {e}")
            return None
    
    return None

def main():
    # 1. Setup
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: Create a folder named '{INPUT_FOLDER}' and put your text files inside.")
        return

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.txt', '.html', '.md', '.json'))]
    random.shuffle(files)
    print(f"🚀 Found {len(files)} docs. Generating Universal Barq Dataset...")

    # 2. Resume Capability
    lines_done = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            lines_done = sum(1 for _ in f)
        print(f"🔄 Resuming from line {lines_done}...")

    # 3. Generation Loop
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        for i, filename in enumerate(tqdm(files)):
            if i < lines_done: continue 

            try:
                with open(os.path.join(INPUT_FOLDER, filename), "r", encoding="utf-8") as f_in:
                    text_chunk = f_in.read()
            except:
                continue 

            if len(text_chunk) < 100: continue 

            # Decision: Positive or Negative?
            is_neg = random.random() < NEGATIVE_RATIO
            
            if is_neg:
                # Negative: Universal General Knowledge questions
                # These ensure the model knows boundaries without being country-specific
                universal_irrelevant_topics = [
                    "كيف أعدل النوم؟", "ما هي فوائد الشاي الأخضر؟", 
                    "متى اخترعت الطائرة؟", "كيف أتعلم البرمجة؟",
                    "ما هي عاصمة اليابان؟", "طريقة عمل البيتزا"
                ]
                q = random.choice(universal_irrelevant_topics)
                result = get_deepseek_completion(text_chunk, q, is_negative=True)
            
            else:
                # Positive: Ask DeepSeek to invent a relevant question
                try:
                    q_prompt = f"Read this text: '{text_chunk[:800]}...'\nGenerate a short, realistic Arabic question about this text."
                    q_resp = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": q_prompt}]
                    )
                    q = q_resp.choices[0].message.content.strip().replace('"', '')
                    result = get_deepseek_completion(text_chunk, q, is_negative=False)
                except:
                    continue

            # 4. Save
            if result:
                entry = {
                    "conversations": [
                        {"role": "system", "content": "أنت 'برق'، مساعد عربي ذكي."},
                        {"role": "user", "content": f"السياق:\n{text_chunk[:2500]}\n\nالسؤال:\n{q}"},
                        {"role": "assistant", "content": f"{result.get('thought_process', '')}\n{result.get('final_response', '')}"}
                    ]
                }
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f_out.flush()

    print(f"\n✅ Done! Dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()