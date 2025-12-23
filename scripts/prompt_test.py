import os
import json
from openai import OpenAI

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
API_KEY = "..." # 🔴 PASTE YOUR KEY HERE
MODEL_NAME = "deepseek-chat"

# ==========================================
# 🧪 MOCK DATA (The "Test Case")
# ==========================================
# We use a fake document about "Electronic Visas" to test the logic.
mock_context = """
نظام التأشيرات الإلكترونية الجديد (2025):
يسمح للسياح من جميع الجنسيات بالتقدم بطلب تأشيرة سياحية إلكترونية صالحة لمدة عام واحد.
رسوم التأشيرة هي 400 ريال سعودي، تشمل التأمين الطبي الأساسي.
يسمح لحامل التأشيرة بالإقامة لمدة 90 يوماً متواصلة كحد أقصى.
لا يسمح لحامل هذه التأشيرة بالعمل أو أداء فريضة الحج، ولكن يسمح بأداء العمرة في غير موسم الحج.
"""

mock_question = "أنا سائح، هل أقدر أشتغل في مطعم وأنا معي فيزا سياحية؟"

# ==========================================
# 🧠 THE SYSTEM PROMPT (The Logic We Are Testing)
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


# ==========================================
# 🚀 EXECUTION
# ==========================================
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

print("⏳ Sending Test Request to DeepSeek...")

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{mock_context}\n\nUSER QUESTION:\n{mock_question}"}
        ],
        response_format={"type": "json_object"}, 
        temperature=0.7
    )
    
    # Parse Result
    content = response.choices[0].message.content
    result = json.loads(content)
    
    print("\n✅ SUCCESS! Here is what the model generated:\n")
    print("="*60)
    print(f"🧠 THOUGHT PROCESS:\n{result['thought_process']}")
    print("-" * 60)
    print(f"🗣️ FINAL ANSWER:\n{result['final_response']}")
    print("="*60)

except Exception as e:
    print(f"\n❌ FAILED: {e}")