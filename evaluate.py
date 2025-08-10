# evaluate.py
"""
Evaluate student answers (text + image) against an answer key.

Requires:
    - student_answers.json from extractors.py
    - answer_key.json from generation module
    - answer_evaluator.py (must have evaluate_text_answer function)
    - clip_image_compare.py (must have compare_images function)

Input JSON formats:
student_answers.json:
{
    "Q1": {
        "Text": "student answer text...",
        "Image": "path/to/student/image.png" or null
    },
    ...
}

answer_key.json:
{
    "Q1": {
        "Text": "correct answer text...",
        "Image": "path/to/reference/image.png" or null,
        "BloomLevel": "Understand" (optional),
        "Keywords": ["keyword1", "keyword2"] (optional),
        "Question": "What is...?" (optional)
    },
    ...
}
"""

import json
import os
import sys
from pathlib import Path

# Import evaluation functions
try:
    from answer_evaluator import evaluate_text_answer
except ImportError:
    print("Error: Could not import answer_evaluator. Make sure answer_evaluator.py is in the same directory.")
    sys.exit(1)

try:
    from clip_image_compare import compare_images
except ImportError:
    print("Error: Could not import clip_image_compare. Make sure clip_image_compare.py is in the same directory.")
    sys.exit(1)

# Weights for combining text and image scores
TEXT_WEIGHT = 0.6   # weight for text score
IMAGE_WEIGHT = 0.4  # weight for image score

def safe_float(value, default=0.0):
    """Safely convert value to float with default fallback."""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def generate_detailed_explanation(student_data, key_data, text_score, image_score, final_score, detailed_text_eval=None):
    """
    Generate detailed explanation for why student got specific marks.
    
    Args:
        student_data: Student's answer data
        key_data: Reference answer data
        text_score: Text evaluation score
        image_score: Image evaluation score
        final_score: Final combined score
        detailed_text_eval: Detailed text evaluation results
    
    Returns:
        Dict with detailed explanations
    """
    explanations = []
    strengths = []
    weaknesses = []
    suggestions = []
    
    # Text evaluation explanation
    if text_score is not None:
        student_text = student_data.get("Text", "").strip()
        reference_text = key_data.get("Text", "").strip()
        
        if text_score >= 0.8:
            strengths.append("Excellent text answer with high semantic similarity to reference")
        elif text_score >= 0.6:
            strengths.append("Good text answer covering most key concepts")
            if detailed_text_eval:
                if detailed_text_eval.get("keyword_coverage", 0) < 0.7:
                    weaknesses.append("Missing some important keywords")
                    suggestions.append("Include more technical terms and key concepts")
        elif text_score >= 0.4:
            weaknesses.append("Average text answer with moderate similarity to reference")
            suggestions.append("Expand your explanation with more details and key concepts")
            if detailed_text_eval:
                if detailed_text_eval.get("semantic_score", 0) < 0.5:
                    weaknesses.append("Low conceptual understanding")
                    suggestions.append("Review the fundamental concepts and their relationships")
        else:
            weaknesses.append("Poor text answer with low similarity to reference")
            suggestions.append("Revisit the topic and provide a more comprehensive answer")
        
        # Keyword analysis
        if detailed_text_eval and key_data.get("Keywords"):
            kw_coverage = detailed_text_eval.get("keyword_coverage", 0)
            total_keywords = len(key_data.get("Keywords", []))
            found_keywords = int(kw_coverage * total_keywords)
            
            explanations.append(f"Keyword coverage: {found_keywords}/{total_keywords} key terms found")
            if kw_coverage < 0.5:
                weaknesses.append("Missing critical technical keywords")
                missing_kw = [kw for kw in key_data.get("Keywords", []) 
                             if kw.lower() not in student_text.lower()]
                if missing_kw:
                    suggestions.append(f"Include these important terms: {', '.join(missing_kw[:3])}")
        
        # Bloom's taxonomy analysis
        if detailed_text_eval:
            expected_bloom = detailed_text_eval.get("bloom_expected", "")
            classified_bloom = detailed_text_eval.get("bloom_classified", "")
            
            if expected_bloom and classified_bloom:
                if expected_bloom == classified_bloom:
                    strengths.append(f"Answer demonstrates appropriate {expected_bloom} level thinking")
                else:
                    bloom_hierarchy = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
                    expected_idx = bloom_hierarchy.index(expected_bloom) if expected_bloom in bloom_hierarchy else 1
                    actual_idx = bloom_hierarchy.index(classified_bloom) if classified_bloom in bloom_hierarchy else 1
                    
                    if actual_idx < expected_idx:
                        weaknesses.append(f"Answer shows {classified_bloom} level but {expected_bloom} level required")
                        suggestions.append(f"Provide more {expected_bloom.lower()}-level analysis and reasoning")
    else:
        if not student_data.get("Text", "").strip():
            weaknesses.append("No text answer provided")
            suggestions.append("Provide a written explanation for the question")
    
    # Image evaluation explanation
    if image_score is not None:
        if image_score >= 0.8:
            strengths.append("Excellent visual answer with high similarity to reference")
        elif image_score >= 0.6:
            strengths.append("Good visual answer with acceptable similarity")
            suggestions.append("Consider adding more details or improving clarity in diagrams")
        elif image_score >= 0.4:
            weaknesses.append("Average visual answer with moderate similarity")
            suggestions.append("Review the expected diagram format and include missing elements")
        else:
            weaknesses.append("Poor visual answer with low similarity to reference")
            suggestions.append("Redraw the diagram following the reference format more closely")
    else:
        if key_data.get("Image") and not student_data.get("Image"):
            weaknesses.append("Missing required diagram or visual component")
            suggestions.append("Include the requested diagram or visual element")
    
    # Overall score explanation
    percentage = final_score * 100
    if percentage >= 90:
        overall_comment = "Excellent performance"
    elif percentage >= 80:
        overall_comment = "Very good performance"
    elif percentage >= 70:
        overall_comment = "Good performance"
    elif percentage >= 60:
        overall_comment = "Satisfactory performance"
    elif percentage >= 50:
        overall_comment = "Below average performance"
    else:
        overall_comment = "Poor performance - requires significant improvement"
    
    # Grade recommendation
    if percentage >= 90:
        grade = "A+"
    elif percentage >= 85:
        grade = "A"
    elif percentage >= 80:
        grade = "A-"
    elif percentage >= 75:
        grade = "B+"
    elif percentage >= 70:
        grade = "B"
    elif percentage >= 65:
        grade = "B-"
    elif percentage >= 60:
        grade = "C+"
    elif percentage >= 55:
        grade = "C"
    elif percentage >= 50:
        grade = "C-"
    elif percentage >= 45:
        grade = "D+"
    elif percentage >= 40:
        grade = "D"
    else:
        grade = "F"
    
    return {
        "overall_comment": overall_comment,
        "grade_recommendation": grade,
        "percentage": round(percentage, 1),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "detailed_explanations": explanations
    }

def evaluate_single_question(student_data, key_data, q_num):
    """
    Evaluate a single question's answer (both text and image) with detailed explanations.
    
    Args:
        student_data: Dict with "Text" and "Image" keys
        key_data: Dict with "Text", "Image", and optional metadata
        q_num: Question number for logging
    
    Returns:
        Dict with evaluation results and detailed explanations
    """
    text_score = None
    image_score = None
    detailed_text_eval = None
    
    # Evaluate text component
    student_text = student_data.get("Text", "").strip()
    reference_text = key_data.get("Text", "").strip()
    
    if student_text and reference_text:
        try:
            # Import the detailed evaluation function
            from answer_evaluator import evaluate_answer
            
            detailed_text_eval = evaluate_answer(
                gt_question=key_data.get("Question", ""),
                gt_answer=reference_text,
                stu_answer=student_text,
                bloom_gt=key_data.get("BloomLevel"),
                keywords=key_data.get("Keywords")
            )
            text_score = detailed_text_eval["final_score"]
            print(f"{q_num} - Text evaluation: {text_score:.3f}")
        except Exception as e:
            print(f"{q_num} - Text evaluation failed: {e}")
            text_score = 0.0
    elif not student_text and reference_text:
        text_score = 0.0
        print(f"{q_num} - No student text found")
    elif not reference_text:
        print(f"{q_num} - No reference text found")
    
    # Evaluate image component
    student_image = student_data.get("Image")
    reference_image = key_data.get("Image")
    
    if student_image and reference_image:
        if os.path.exists(student_image) and os.path.exists(reference_image):
            try:
                image_score = compare_images(student_image, reference_image)
                print(f"{q_num} - Image evaluation: {image_score:.3f}")
            except Exception as e:
                print(f"{q_num} - Image evaluation failed: {e}")
                image_score = 0.0
        else:
            missing_files = []
            if not os.path.exists(student_image):
                missing_files.append(f"student image: {student_image}")
            if not os.path.exists(reference_image):
                missing_files.append(f"reference image: {reference_image}")
            print(f"{q_num} - Missing files: {', '.join(missing_files)}")
            image_score = 0.0
    elif not student_image and reference_image:
        image_score = 0.0
        print(f"{q_num} - No student image found")
    elif not reference_image:
        print(f"{q_num} - No reference image found")
    
    # Combine scores
    final_score = None
    score_components = []
    
    if text_score is not None and image_score is not None:
        final_score = TEXT_WEIGHT * text_score + IMAGE_WEIGHT * image_score
        score_components = [f"text({TEXT_WEIGHT}*{text_score:.3f})", f"image({IMAGE_WEIGHT}*{image_score:.3f})"]
    elif text_score is not None:
        final_score = text_score
        score_components = ["text only"]
    elif image_score is not None:
        final_score = image_score
        score_components = ["image only"]
    else:
        final_score = 0.0
        score_components = ["no valid components"]
    
    print(f"{q_num} - Final score: {final_score:.3f} ({', '.join(score_components)})")
    
    # Generate detailed explanation
    explanation = generate_detailed_explanation(
        student_data, key_data, text_score, image_score, final_score, detailed_text_eval
    )
    
    return {
        "TextScore": safe_float(text_score),
        "ImageScore": safe_float(image_score),
        "FinalScore": safe_float(final_score),
        "ScoreComponents": score_components,
        "DetailedTextEvaluation": detailed_text_eval,
        "Explanation": explanation
    }

def evaluate(student_json_path, key_json_path, output_json_path="results.json"):
    """
    Main evaluation function.
    
    Args:
        student_json_path: Path to student answers JSON
        key_json_path: Path to answer key JSON
        output_json_path: Path to save results
    """
    # Validate input files
    if not os.path.exists(student_json_path):
        raise FileNotFoundError(f"Student answers file not found: {student_json_path}")
    if not os.path.exists(key_json_path):
        raise FileNotFoundError(f"Answer key file not found: {key_json_path}")
    
    # Load data
    print(f"Loading student answers from: {student_json_path}")
    with open(student_json_path, "r", encoding="utf-8") as f:
        student_answers = json.load(f)
    
    print(f"Loading answer key from: {key_json_path}")
    with open(key_json_path, "r", encoding="utf-8") as f:
        answer_key = json.load(f)
    
    print(f"Found {len(student_answers)} student answers")
    print(f"Found {len(answer_key)} answer key entries")
    
    # Evaluate each question
    results = {}
    all_questions = set(student_answers.keys()) | set(answer_key.keys())
    
    for q_num in sorted(all_questions, key=lambda x: int(''.join(filter(str.isdigit, x)) or '0')):
        print(f"\n--- Evaluating {q_num} ---")
        
        student_data = student_answers.get(q_num, {})
        key_data = answer_key.get(q_num, {})
        
        if not student_data:
            print(f"{q_num} - No student answer found")
            results[q_num] = {
                "TextScore": 0.0,
                "ImageScore": 0.0,
                "FinalScore": 0.0,
                "ScoreComponents": ["no student answer"]
            }
            continue
        
        if not key_data:
            print(f"{q_num} - No answer key found")
            results[q_num] = {
                "TextScore": None,
                "ImageScore": None,
                "FinalScore": None,
                "ScoreComponents": ["no answer key"]
            }
            continue
        
        results[q_num] = evaluate_single_question(student_data, key_data, q_num)
    
    # Calculate overall statistics
    valid_scores = [r["FinalScore"] for r in results.values() if r["FinalScore"] is not None]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        print(f"\n--- Overall Statistics ---")
        print(f"Questions evaluated: {len(valid_scores)}")
        print(f"Average score: {avg_score:.3f}")
        print(f"Score range: {min(valid_scores):.3f} - {max(valid_scores):.3f}")
    else:
        avg_score = 0.0
        print(f"\nNo valid scores calculated")
    
    # Add summary to results
    results["_summary"] = {
        "total_questions": len(all_questions),
        "evaluated_questions": len(valid_scores),
        "average_score": avg_score,
        "text_weight": TEXT_WEIGHT,
        "image_weight": IMAGE_WEIGHT
    }
    
    # Save results
    print(f"\nSaving results to: {output_json_path}")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("Evaluation complete!")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate student answers against answer key.")
    parser.add_argument("--student", required=True, help="Path to student_answers.json")
    parser.add_argument("--key", required=True, help="Path to answer_key.json")
    parser.add_argument("--output", default="results.json", help="Path to output results JSON")
    parser.add_argument("--text-weight", type=float, default=0.6, help="Weight for text score (default: 0.6)")
    parser.add_argument("--image-weight", type=float, default=0.4, help="Weight for image score (default: 0.4)")
    
    args = parser.parse_args()
    
    # Update weights if provided
    TEXT_WEIGHT = args.text_weight
    IMAGE_WEIGHT = args.image_weight
    
    # Validate weights
    if abs(TEXT_WEIGHT + IMAGE_WEIGHT - 1.0) > 0.01:
        print(f"Warning: Text weight ({TEXT_WEIGHT}) + Image weight ({IMAGE_WEIGHT}) != 1.0")
    
    try:
        evaluate(args.student, args.key, args.output)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)