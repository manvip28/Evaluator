import json
import re
from PIL import Image
from pdf2image import convert_from_path

from surya.layout import LayoutPredictor
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor

def extract_text_from_crop(cropped_image, recognition_predictor, detection_predictor):
    """Extract text from a cropped image using Surya OCR"""
    try:
        predictions = recognition_predictor([cropped_image], det_predictor=detection_predictor)
        
        if predictions and len(predictions) > 0:
            page_pred = predictions[0]
            
            # Handle different prediction structures
            if hasattr(page_pred, 'text_lines'):
                text_lines = [line.text for line in page_pred.text_lines if hasattr(line, 'text')]
            elif isinstance(page_pred, dict) and 'text_lines' in page_pred:
                text_lines = [line.get('text', '') for line in page_pred['text_lines']]
            else:
                print(f"Unexpected prediction structure: {type(page_pred)}")
                return []
            
            return [line.strip() for line in text_lines if line.strip()]
    except Exception as e:
        print(f"Error extracting text: {e}")
        return []

def identify_questions(text_lines):
    """Identify question numbers from text lines"""
    questions = {}
    current_question = None
    current_text = []
    
    for line in text_lines:
        # Look for question patterns like "Q1.", "Q2.", "1.", "2.", etc.
        question_match = re.match(r'^[Qq]?(\d+)\.?\s*(.*)$', line.strip())
        
        if question_match:
            # Save previous question if it exists
            if current_question is not None and current_text:
                questions[current_question] = '\n'.join(current_text).strip()
            
            # Start new question
            current_question = f"Q{question_match.group(1)}"
            remaining_text = question_match.group(2).strip()
            current_text = [remaining_text] if remaining_text else []
        else:
            # Add to current question
            if current_question is not None:
                current_text.append(line.strip())
    
    # Save the last question
    if current_question is not None and current_text:
        questions[current_question] = '\n'.join(current_text).strip()
    
    return questions

# Path to your PDF file
pdf_path = 'student_answer.pdf'
output_json_path = 'extracted_answers.json'

# Convert PDF pages to images
pages = convert_from_path(pdf_path)

# Initialize Surya predictors
layout_predictor = LayoutPredictor()
detection_predictor = DetectionPredictor()
foundation_predictor = FoundationPredictor()
recognition_predictor = RecognitionPredictor(foundation_predictor)

# Store all extracted data
all_results = {
    "pages": [],
    "questions": {}
}

for page_num, page_image in enumerate(pages, start=1):
    print(f"\n--- Processing Page {page_num} ---")
    
    page_data = {
        "page_number": page_num,
        "elements": [],
        "all_text": [],
        "images": []
    }
    
    # Step 1: Layout detection
    layout_predictions = layout_predictor([page_image])
    page_layout = layout_predictions[0]
    bboxes = page_layout.bboxes
    
    for element_idx, element in enumerate(bboxes):
        bbox = element.bbox
        label = element.label
        confidence = element.confidence
        
        x_min, y_min, x_max, y_max = map(int, bbox)
        
        print(f"Found element - Label: {label}, Confidence: {confidence:.3f}, BBox: ({x_min}, {y_min}, {x_max}, {y_max})")
        
        element_data = {
            "type": label,
            "confidence": confidence,
            "bbox": [x_min, y_min, x_max, y_max],
            "text": "",
            "image_saved": False
        }
        
        # Crop the element
        cropped = page_image.crop((x_min, y_min, x_max, y_max))
        
        if label in ['Text', 'Section-header', 'List-item', 'Handwriting']:
            # Try OCR on all text-like elements, including handwriting
            print(f"Attempting OCR on {label} block...")
            
            text_lines = extract_text_from_crop(cropped, recognition_predictor, detection_predictor)
            
            if text_lines:
                extracted_text = '\n'.join(text_lines)
                element_data["text"] = extracted_text
                page_data["all_text"].extend(text_lines)
                
                print(f"\nExtracted Text from {label} block:")
                print(extracted_text)
                print("----------")
            else:
                print(f"No text extracted from {label} block")
        
        elif label in ['Picture', 'Figure']:
            # Save image blocks
            image_filename = f'page_{page_num}_image_{element_idx}_{x_min}_{y_min}.png'
            cropped.save(image_filename)
            element_data["image_saved"] = True
            element_data["image_filename"] = image_filename
            page_data["images"].append({
                "filename": image_filename,
                "bbox": [x_min, y_min, x_max, y_max]
            })
            print(f"[{label} saved as {image_filename}]")
        
        page_data["elements"].append(element_data)
    
    # Try to identify questions from all text on this page
    if page_data["all_text"]:
        page_questions = identify_questions(page_data["all_text"])
        print(f"\nIdentified questions on page {page_num}: {list(page_questions.keys())}")
        
        # Merge with overall questions
        all_results["questions"].update(page_questions)
    
    all_results["pages"].append(page_data)
    print(f"--- End of Page {page_num} ---")

# Create structured question format as requested
structured_questions = {}
for q_num, q_text in all_results["questions"].items():
    # Find images that might belong to this question
    # This is a simple approach - you might want to improve this logic
    question_images = []
    
    # For now, just assign all images to questions (you can improve this logic)
    for page_data in all_results["pages"]:
        question_images.extend([img["filename"] for img in page_data["images"]])
    
    structured_questions[q_num] = {
        "text": q_text if q_text else "No text extracted",
        "images": question_images
    }

# If no questions were identified, create a generic structure
if not structured_questions:
    # Combine all text and images
    all_text = []
    all_images = []
    
    for page_data in all_results["pages"]:
        all_text.extend(page_data["all_text"])
        all_images.extend([img["filename"] for img in page_data["images"]])
    
    structured_questions["Q1"] = {
        "text": '\n'.join(all_text) if all_text else "No text extracted",
        "images": all_images
    }

# Final output structure
final_output = {
    "document": pdf_path,
    "total_pages": len(pages),
    "extraction_summary": {
        "total_questions_found": len(structured_questions),
        "total_images_saved": sum(len(page["images"]) for page in all_results["pages"])
    },
    "questions": structured_questions,
    "raw_data": all_results  # Include raw data for debugging
}

# Save to JSON
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

print(f"\n=== EXTRACTION COMPLETE ===")
print(f"Results saved to: {output_json_path}")
print(f"Questions found: {list(structured_questions.keys())}")

# Print summary in your requested format
print(f"\n=== STRUCTURED OUTPUT ===")
for q_num, q_data in structured_questions.items():
    print(f"\n{q_num}.")
    print("Text:")
    print(q_data["text"])
    print("--------")
    print("Images:")
    if q_data["images"]:
        for img in q_data["images"]:
            print(f"- {img}")
    else:
        print("No images found")
    print("-------")
