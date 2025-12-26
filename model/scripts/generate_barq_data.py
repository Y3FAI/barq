import os
import json
import random
import time
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
import threading

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
API_KEY = "sk-ecab4d871ce849a883866687b31409d3"
INPUT_FOLDER = "./raw_data"        # Folder with your 2000 files
OUTPUT_FILE = "barq_2k_finetune.jsonl"
MODEL_NAME = "deepseek-chat"
MAX_WORKERS = 10                   # 🚀 Process 10 files at the same time
NEGATIVE_RATIO = 0.15              # 15% Refusal/General questions

# Lock for writing to file safely across threads
file_lock = threading.Lock()
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# ==========================================
# 🧠 PROMPTS
# ==========================================
SYSTEM_PROMPT = """
You are a synthetic data generator for 'Barq', a Saudi Traffic Law AI.
Generate a training example based on the provided text.

OUTPUT JSON FORMAT:
{
  "question": "[Question in informal Saudi Dialect (e.g. using words like 'موتر', 'علي مخالفة', 'صادني')]",
  "thought_process": "<think> [Identify User Intent] -> [Find Fact in Text] -> [Draft Answer] </think>",
  "final_response": "[Direct, professional Arabic answer based ONLY on the text]"
}

RULES:
1. Question must be realistic and short.
2. Response must be grammatically correct Standard Arabic.
3. If the text does not contain enough info, return null.
"""

# Universal "Negative" questions to teach the model what it doesn't know
IRRELEVANT_QUESTIONS = [
    "طريقة عمل الكبسة؟", "من هو بطل كأس العالم؟", "كيف أصلح مكيف السبلت؟",
    "متى تأسست شركة أبل؟", "وش أفضل مطعم في الرياض؟", "كيف أسوي شاي كرك؟",
    "كم سعر الايفون الجديد؟", "هل القهوة مضرة؟", "قصة مسلسل طاش ما طاش"
]

def generate_entry(text_chunk, filename):
    """
    Process a single document text and return a JSONL entry.
    """
    if len(text_chunk) < 50: return None

    # 🎲 Decide: Positive (Fact) or Negative (Refusal)?
    is_negative = random.random() < NEGATIVE_RATIO

    if is_negative:
        # Generate a refusal example
        irrelevant_q = random.choice(IRRELEVANT_QUESTIONS)
        return {
            "conversations": [
                {"role": "system", "content": "أنت 'برق'، مساعد قانوني سعودي."},
                {"role": "user", "content": f"السياق:\n{text_chunk[:1000]}\n\nالسؤال:\n{irrelevant_q}"},
                {"role": "assistant", "content": "<think>The user asked an irrelevant question unrelated to the context.</think>\nعذراً، أنا متخصص فقط في أنظمة المرور السعودية ولا أملك معلومات حول هذا الموضوع."}
            ]
        }

    # ✅ Positive Example: Ask LLM to generate Q&A Pair
    user_prompt = f"""
    CONTEXT FROM FILE ({filename}):
    {text_chunk[:3500]}

    INSTRUCTION:
    Generate a JSON object with a Saudi Dialect question about this text and the correct answer.
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.8, # Higher temp = better dialect variety
                timeout=40
            )
            
            content = response.choices[0].message.content
            if not content: return None
            
            data = json.loads(content)
            
            # Validate keys exist
            if "question" not in data or "final_response" not in data:
                return None

            return {
                "conversations": [
                    {"role": "system", "content": "أنت 'برق'، مساعد قانوني سعودي."},
                    {"role": "user", "content": f"السياق:\n{text_chunk[:3000]}\n\nالسؤال:\n{data['question']}"},
                    {"role": "assistant", "content": f"{data.get('thought_process', '<think>Answer generated based on context.</think>')}\n{data['final_response']}"}
                ]
            }

        except Exception as e:
            time.sleep(1 * (attempt + 1))
            continue
    
    return None

def process_file_wrapper(filepath):
    """
    Helper to read file and call generator.
    """
    try:
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return generate_entry(text, filename)
    except Exception as e:
        return None

def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: Folder '{INPUT_FOLDER}' not found.")
        return

    # 1. Gather Files
    all_files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.endswith(('.txt', '.html', '.md', '.json', '.pdf'))]
    random.shuffle(all_files)
    
    # Filter out already processed if needed (simple count check)
    completed_lines = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            completed_lines = sum(1 for _ in f)
    
    files_to_process = all_files[completed_lines:]
    print(f"🚀 Found {len(all_files)} docs. Processing {len(files_to_process)} new files...")
    print(f"⚡ Using {MAX_WORKERS} parallel threads.")

    # 2. Parallel Processing Loop
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            futures = {executor.submit(process_file_wrapper, f): f for f in files_to_process}
            
            # Watch progress
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(files_to_process)):
                result = future.result()
                if result:
                    # Thread-safe write
                    with file_lock:
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush()

    print(f"\n✅ Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()