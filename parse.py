import os
import base64
import json
import re
import google.generativeai as genai
from PIL import Image
import time
import io

# ----------------------------
# 1. Configure API Key
# ----------------------------
GEN_API_KEY = xyz  # Replace with your actual API key
genai.configure(api_key=GEN_API_KEY)
print("[DEBUG] API key configured.")

# ----------------------------
# 2. Load and optionally resize image
# ----------------------------
image_path = "answer.png"
print(f"[DEBUG] Loading image: {image_path}")
start_time = time.time()
img = Image.open(image_path)
img = img.convert("RGB")

# Resize if image is too large (to avoid response truncation)
max_dimension = 2048
if img.width > max_dimension or img.height > max_dimension:
    ratio = min(max_dimension / img.width, max_dimension / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    print(f"[DEBUG] Image resized to {new_size}")

print(f"[DEBUG] Image loaded in {time.time() - start_time:.2f}s")

# ----------------------------
# 3. Encode image to Base64
# ----------------------------
print("[DEBUG] Encoding image to Base64")
start_time = time.time()
buffer = io.BytesIO()
img.save(buffer, format="JPEG", quality=85)
img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
print(f"[DEBUG] Image encoded in {time.time() - start_time:.2f}s, size: {len(img_b64)/1024:.2f} KB")

# ----------------------------
# 4. Prepare improved prompt
# ----------------------------
prompt = """
You are an OCR model that can detect layout in a handwritten page.
The page contains multiple answers, some with associated diagrams.

IMPORTANT INSTRUCTIONS:
1. Extract text from each answer
2. For diagrams: DO NOT include base64 images in the JSON. Instead, indicate that a diagram exists.
3. Return ONLY valid JSON (no markdown, no code blocks)

Return in this format:
{
  "q1": {"text": "extracted text here", "has_diagram": false},
  "q2": {"text": "extracted text here", "has_diagram": false},
  "q3": {"text": "extracted text here", "has_diagram": true}
}

Rules:
- Set "has_diagram" to true if the answer contains a diagram, false otherwise
- DO NOT include actual image data
- Return ONLY the JSON object, nothing else
"""

# ----------------------------
# 5. Send request to Gemini
# ----------------------------
print("[DEBUG] Sending request to Gemini...")
start_time = time.time()
model = genai.GenerativeModel("gemini-2.0-flash")

try:
    response = model.generate_content([
        {
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
            ]
        }
    ])
    print(f"[DEBUG] Gemini response received in {time.time() - start_time:.2f}s")
except Exception as e:
    print(f"❌ Error calling Gemini API: {e}")
    exit(1)

# ----------------------------
# 6. Extract JSON from response
# ----------------------------
def extract_json_from_response(text):
    """Extract JSON from markdown code blocks or raw text"""
    # Remove markdown code blocks
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    
    # Try to find JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return text.strip()

print("[DEBUG] Extracting and parsing JSON output")
try:
    raw_response = response.text
    print(f"[DEBUG] Raw response length: {len(raw_response)} chars")
    
    json_text = extract_json_from_response(raw_response)
    
    # Additional cleanup for common JSON issues
    json_text = json_text.strip()
    
    # Try to detect truncated JSON and warn user
    if json_text.count('{') != json_text.count('}'):
        print("⚠️  WARNING: JSON appears truncated (mismatched braces)")
    
    parsed_json = json.loads(json_text)
    print("[DEBUG] ✓ JSON parsed successfully")
    print(f"[DEBUG] Found {len(parsed_json)} questions")
    
except json.JSONDecodeError as e:
    print(f"❌ Failed to parse JSON: {e}")
    print(f"\nExtracted JSON text (first 500 chars):\n{json_text[:500]}")
    print(f"\n...last 500 chars:\n{json_text[-500:]}")
    
    # Try to salvage partial data
    print("\n[DEBUG] Attempting to salvage partial data...")
    try:
        # Find the last complete question entry
        truncated_match = re.search(r'\{.*"q\d+".*?\}(?=\s*,?\s*"q\d+"|\s*\})', json_text, re.DOTALL)
        if truncated_match:
            print("[DEBUG] Found partial valid JSON, saving what we can...")
    except:
        pass
    
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nRaw response (first 1000 chars):\n", raw_response[:1000])
    exit(1)

# ----------------------------
# 7. Save JSON output
# ----------------------------
output_file = "ocr_output.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(parsed_json, f, ensure_ascii=False, indent=2)
print(f"[DEBUG] ✓ JSON output saved to {output_file}")

# ----------------------------
# 8. Display extracted text
# ----------------------------
print("\n" + "="*60)
print("EXTRACTED TEXT:")
print("="*60)
for qno, data in parsed_json.items():
    print(f"\n{qno.upper()}:")
    print(f"Text: {data.get('text', 'N/A')}")
    print(f"Has diagram: {data.get('has_diagram', False)}")

# ----------------------------
# 9. Count statistics
# ----------------------------
total_questions = len(parsed_json)
questions_with_diagrams = sum(1 for q in parsed_json.values() if q.get('has_diagram', False))
print("\n" + "="*60)
print(f"Total questions extracted: {total_questions}")
print(f"Questions with diagrams: {questions_with_diagrams}")
print("="*60)

print("\n[DEBUG] All done!")
