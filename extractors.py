# extractors.py

import cv2
import pytesseract
import json
import os
import re
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import uuid

# ---------- CONFIG ----------
IMAGE_OUTPUT_DIR = "extracted_images"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# Hardcoded pattern for question detection
QUESTION_SPLIT_PATTERN = r"(Q\d+\.?|\d+\)|Question\s+\d+)"

# Path to tesseract binary if needed (Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def pdf_to_images(pdf_path):
    """Convert PDF to list of PIL images."""
    return convert_from_path(pdf_path)


def extract_images_from_page(pil_image, min_area=5000):
    """
    Detect image-like regions from a scanned page using OpenCV.
    Returns list of saved image paths with their positions.
    """
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    saved_paths = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > min_area:
            roi = pil_image.crop((x, y, x + w, y + h))
            img_path = os.path.join(IMAGE_OUTPUT_DIR, f"extracted_img_{uuid.uuid4().hex[:8]}.png")
            roi.save(img_path)
            saved_paths.append((y, img_path))  # Use y-coordinate for vertical ordering
    
    # Sort by y-coordinate (top to bottom)
    saved_paths.sort(key=lambda x: x[0])
    return [path for _, path in saved_paths]


def ocr_page(pil_image):
    """Extract raw text from page via Tesseract."""
    return pytesseract.image_to_string(pil_image)


def split_questions(raw_text):
    """
    Split OCR'd text into question chunks using regex.
    Returns dict: { "Q1": "answer text...", "Q2": "..." }
    """
    # Clean up the text first
    cleaned_text = re.sub(r'\n+', '\n', raw_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    parts = re.split(QUESTION_SPLIT_PATTERN, cleaned_text, flags=re.IGNORECASE)
    result = {}
    current_q = None
    
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
            
        # Check if this part is a question identifier
        if re.match(QUESTION_SPLIT_PATTERN, part, flags=re.IGNORECASE):
            # Save previous question if exists
            if current_q and i + 1 < len(parts):
                # The answer should be in the next part
                answer_text = parts[i + 1].strip()
                if answer_text:
                    result[current_q] = answer_text
            
            # Normalize question identifier
            q_match = re.search(r'(\d+)', part)
            if q_match:
                current_q = f"Q{q_match.group(1)}"
        elif current_q and not re.match(QUESTION_SPLIT_PATTERN, part, flags=re.IGNORECASE):
            # This is answer text for the current question
            result[current_q] = part
            current_q = None
    
    return result


def match_images_to_questions(questions, images):
    """
    Simple matching: assign images to questions in order.
    You can implement more sophisticated matching based on position if needed.
    """
    matched = {}
    
    # Sort questions by number
    sorted_questions = sorted(questions.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
    
    for i, q_num in enumerate(sorted_questions):
        matched[q_num] = {
            "Text": questions[q_num],
            "Image": images[i] if i < len(images) else None
        }
    
    return matched


def process_answer_sheet(input_path, output_json="student_answers.json"):
    """Main function to extract per-question text and image paths."""
    print(f"Processing: {input_path}")
    
    # Convert input to images
    if input_path.lower().endswith(".pdf"):
        pages = pdf_to_images(input_path)
        print(f"Converted PDF to {len(pages)} pages")
    else:
        pages = [Image.open(input_path).convert("RGB")]
        print("Processing single image")

    all_questions = {}
    all_images = []

    for page_num, page in enumerate(pages, 1):
        print(f"Processing page {page_num}...")
        
        # Extract text and split into questions
        raw_text = ocr_page(page)
        print(f"Extracted text length: {len(raw_text)} characters")
        
        page_questions = split_questions(raw_text)
        print(f"Found questions: {list(page_questions.keys())}")
        
        # Extract images from page
        page_images = extract_images_from_page(page)
        print(f"Found {len(page_images)} images")
        
        # Merge questions and images
        all_questions.update(page_questions)
        all_images.extend(page_images)

    # Match images to questions
    final_data = match_images_to_questions(all_questions, all_images)
    
    # Save to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    print(f"Extraction complete. Found {len(final_data)} questions.")
    print(f"Results saved to {output_json}")
    
    # Print summary
    for q_num in sorted(final_data.keys(), key=lambda x: int(re.search(r'\d+', x).group())):
        data = final_data[q_num]
        text_len = len(data["Text"]) if data["Text"] else 0
        has_image = "Yes" if data["Image"] else "No"
        print(f"{q_num}: Text={text_len} chars, Image={has_image}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract text and images from scanned answer sheet.")
    parser.add_argument("--input", required=True, help="Path to scanned answer sheet (image or PDF).")
    parser.add_argument("--output", default="student_answers.json", help="Path to output JSON file.")
    args = parser.parse_args()

    process_answer_sheet(args.input, args.output)