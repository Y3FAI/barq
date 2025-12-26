import json
import random
import time
import concurrent.futures
import threading
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
API_KEY = "..."  # 🔴 PASTE KEY HERE
OUTPUT_FILE = "general_qfs_train.jsonl"
TARGET_COUNT = 10                # How many examples you want
MAX_WORKERS = 10                   # Parallel threads

# We use Arabic Wikipedia as the source of "All Fields of Life"
DATASET_ID = "MagedSaeed/tnqeet-training-datasets"
DATASET_CONFIG = "arabic_wikipedia"

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
file_lock = threading.Lock()

# ==========================================
# 🧠 THE "GENERALIST" PROMPT
# ==========================================
# We explicitly tell DeepSeek to be a data generator, NOT a chat bot.
SYSTEM_PROMPT = """
You are a synthetic data generator for a Query-Focused Summarization model.
Your goal is to create training data that teaches a small model to EXTRACT answers directly.

INPUT: A text snippet from Wikipedia (History, Science, Culture, etc.).
TASK:
1. Generate a **specific user query** (in Arabic) that can be answered ONLY using the text.
2. Generate the **direct answer** (in Arabic) extracted from the text.
3. The answer must be concise and comprehensive.
4. NO reasoning, NO introductory phrases like "Based on the text", answer directly.

OUTPUT JSON FORMAT:
{
  "instruction": "استخرج الإجابة من النص التالي.",
  "input": "DOCUMENT: [The provided text snippet] ... \\n\\n QUERY: [Your generated question]",
  "output": "[The direct extracted answer]"
}
"""

def generate_example(text_snippet):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"TEXT SNIPPET:\n{text_snippet[:1500]}"} # Limit context to save tokens
            ],
            response_format={ "type": "json_object" },
            temperature=0.8 # High creativity for diverse questions
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        # print(f"⚠️ Gen Error: {e}")
        return None

def main():
    print("🚀 Loading Arabic Wikipedia Stream...")
    # Stream mode: starts instantly, doesn't download the whole internet
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", streaming=True)
    
    # Shuffle buffer to get random topics (not just letters starting with 'A')
    shuffled_ds = ds.shuffle(seed=42, buffer_size=10000)
    
    data_iterator = iter(shuffled_ds)
    generated_count = 0
    
    # Resume check
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            generated_count = sum(1 for _ in f)
    
    print(f"🔄 Resuming from {generated_count}. Target: {TARGET_COUNT}")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            
            # We keep a buffer of futures to keep the GPU/API busy
            futures = []
            
            pbar = tqdm(total=TARGET_COUNT, initial=generated_count)
            
            while generated_count < TARGET_COUNT:
                # 1. Fill the buffer with Wiki articles
                while len(futures) < MAX_WORKERS * 2:
                    try:
                        article = next(data_iterator)
                        text = article.get('text', article.get('content', ''))
                        # Skip short stubs
                        if len(text) < 300: continue 
                        
                        futures.append(executor.submit(generate_example, text))
                    except StopIteration:
                        break

                # 2. Process completed futures
                done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                
                for future in done:
                    result = future.result()
                    if result:
                        # Write immediately to file
                        with file_lock:
                            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        generated_count += 1
                        pbar.update(1)
                    
                    futures.remove(future)
                    
                    if generated_count >= TARGET_COUNT:
                        break
                        
                futures = list(not_done)

    print("\n✅ Dataset Generation Complete!")

if __name__ == "__main__":
    import os
    main()