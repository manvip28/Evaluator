# enhanced_answer_evaluator.py
import json
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from clip_image_compare import compare_images
from bloom_utils import classify_bloom

# -------------------- NLTK Downloads --------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# -------------------- Model & Tools --------------------
sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
rouge = Rouge()
smoothie = SmoothingFunction().method4
stop_words = set(stopwords.words('english'))

# -------------------- Helper Functions --------------------
# Removed extract_keywords function as it's unused - keyword coverage is handled directly in keyword_coverage_score

def keyword_coverage_score(student_text, reference_keywords):
    if not reference_keywords:
        return 1.0
    matches = sum(1 for keyword in reference_keywords if keyword.lower() in student_text.lower())
    return matches / len(reference_keywords)

def evaluate_answer(gt_question, gt_answer, stu_answer, bloom_gt=None, keywords=None, image_score=None):
    """
    Evaluate a single student answer against the ground truth.

    Args:
        gt_question: Ground truth question text
        gt_answer: Ground truth answer text
        stu_answer: Student's answer text
        bloom_gt: Expected Bloom's taxonomy level (optional)
        keywords: List of expected keywords (optional)
        image_score: Optional image similarity score (float between 0-1)

    Returns:
        Dictionary with evaluation metrics
    """
    # Handle empty or None answers
    if not stu_answer or not stu_answer.strip():
        # If student provided only an image, score based on image similarity
        if image_score is not None:
            epsilon = 0.1  # weight for image contribution
            final_score = epsilon * image_score
            curved_score = pow(final_score, 0.8)
        else:
            final_score = 0.0
            curved_score = 0.0

        return {
            "semantic_score": 0.0,
            "bleu": 0.0,
            "rouge_l": 0.0,
            "keyword_coverage": 0.0,
            "bloom_classified": "Remember",
            "bloom_expected": bloom_gt or "Understand",
            "bloom_penalty": 0.0,
            "image_similarity": round(image_score, 4) if image_score is not None else None,
            "raw_score": round(final_score, 4),
            "final_score": round(curved_score, 4)
        }

    # --- Text-based evaluation ---
    emb_gt = sbert_model.encode(gt_answer, convert_to_tensor=True)
    emb_stu = sbert_model.encode(stu_answer, convert_to_tensor=True)
    sem_score = util.cos_sim(emb_gt, emb_stu).item()

    reference = [gt_answer.split()]
    candidate = stu_answer.split()
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)

    try:
        rouge_score = rouge.get_scores(stu_answer, gt_answer)[0]['rouge-l']['f']
    except:
        rouge_score = 0.0

    kw_coverage = keyword_coverage_score(stu_answer, keywords) if keywords else 0.5
    classified = classify_bloom(gt_question or "", stu_answer)

    # Bloom's taxonomy penalty
    if bloom_gt:
        bloom_hierarchy = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        expected_idx = bloom_hierarchy.index(bloom_gt)
        actual_idx = bloom_hierarchy.index(classified)
        bloom_diff = expected_idx - actual_idx

        if bloom_diff == 0:
            penalty = 0.0
        elif bloom_diff > 0:
            penalty = min(0.05 * bloom_diff, 0.15)
        else:
            penalty = -0.02
    else:
        penalty = 0.0

    # --- Weighted scoring with epsilon for image ---
    alpha, beta, gamma, delta, epsilon = 0.4, 0.2, 0.3, 0.1, 0.1
    traditional_score = (bleu_score + rouge_score) / 2
    final_score = (alpha * sem_score) + (beta * traditional_score) + (gamma * kw_coverage) - (delta * penalty)

    if image_score is not None:
        final_score += epsilon * image_score

    final_score = min(max(final_score, 0.0), 1.0)
    curved_score = pow(final_score, 0.8)

    return {
        "semantic_score": round(sem_score, 4),
        "bleu": round(bleu_score, 4),
        "rouge_l": round(rouge_score, 4),
        "keyword_coverage": round(kw_coverage, 4),
        "bloom_classified": classified,
        "bloom_expected": bloom_gt or "Understand",
        "bloom_penalty": round(penalty, 4),
        "image_similarity": round(image_score, 4) if image_score is not None else None,
        "raw_score": round(final_score, 4),
        "final_score": round(curved_score, 4)
    }



def load_json_file(filename):
    """Load JSON file with error handling"""
    try:
        print(f"Loading file: {filename}")
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Successfully loaded {filename} with {len(data)} items")
            return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        print("Make sure the file exists in the current directory.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{filename}': {e}")
        return None

def evaluate_from_json_files(student_file, answer_key_file):
    # Load JSON files
    student_data = load_json_file(student_file)
    answer_key_data = load_json_file(answer_key_file)
    
    if not student_data or not answer_key_data:
        return None
    
    results = {}
    total_score = 0
    evaluated_questions = 0
    
    # Process each question in the answer key
    for question_id, answer_key_info in answer_key_data.items():
        if question_id in student_data:
            student_info = student_data[question_id]
            
            # Extract data from answer key
            gt_question = answer_key_info.get("Question", "")
            gt_answer = answer_key_info.get("Text", "")
            bloom_level = answer_key_info.get("BloomLevel", "Understand")
            keywords = answer_key_info.get("Keywords", [])
            
            # Image comparison if both student and reference provide images
            student_img = student_info.get("Image")
            reference_img = answer_key_info.get("Image")

            image_score = None
            if student_img and reference_img:
                try:
                    image_score = compare_images(student_img, reference_img)
                except Exception as e:
                    print(f"Image comparison failed for {question_id}: {e}")
                    image_score = None

            # Extract student answer
            student_answer = student_info.get("Text", "")
            
            # Only evaluate if there's a ground truth answer
            if gt_answer.strip():
                evaluation = evaluate_answer(
                    gt_question, 
                    gt_answer, 
                    student_answer, 
                    bloom_level, 
                    keywords
                )
                
                percentage_score = round(evaluation["final_score"] * 100, 1)
                
                results[question_id] = {
                    "question": gt_question,
                    "expected_answer": gt_answer,
                    "student_answer": student_answer,
                    "evaluation_details": evaluation,
                    "percentage_score": percentage_score,
                    "has_student_image": bool(student_info.get("Image")),
                    "has_reference_image": bool(answer_key_info.get("Image"))
                }

                if image_score is not None:
                    results[question_id]["evaluation_details"]["image_similarity"] = round(image_score, 4)
                
                total_score += percentage_score
                evaluated_questions += 1
            else:
                results[question_id] = {
                    "question": gt_question,
                    "expected_answer": "No reference answer provided",
                    "student_answer": student_answer,
                    "percentage_score": "N/A",
                    "note": "Cannot evaluate - no reference answer"
                }
        else:
            # Question exists in answer key but student didn't answer
            results[question_id] = {
                "question": answer_key_data[question_id].get("Question", ""),
                "expected_answer": answer_key_data[question_id].get("Text", ""),
                "student_answer": "No answer provided",
                "percentage_score": 0.0,
                "evaluation_details": {
                    "semantic_score": 0.0,
                    "final_score": 0.0
                }
            }
            evaluated_questions += 1
    
    # Calculate overall statistics
    overall_average = total_score / evaluated_questions if evaluated_questions > 0 else 0
    
    return {
        "individual_results": results,
        "summary": {
            "total_questions": len(answer_key_data),
            "answered_questions": len([q for q in student_data.keys() if q in answer_key_data and student_data[q].get("Text", "").strip()]),
            "evaluated_questions": evaluated_questions,
            "overall_average": round(overall_average, 1),
            "total_possible_score": evaluated_questions * 100,
            "total_achieved_score": round(total_score, 1)
        }
    }

def generate_detailed_report(results):
    """Generate a detailed evaluation report"""
    if not results:
        print("No results to generate report from.")
        return
    
    print("=" * 80)
    print("STUDENT ANSWER EVALUATION REPORT")
    print("=" * 80)
    
    # Summary statistics
    summary = results["summary"]
    print(f"\nOVERALL SUMMARY:")
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Questions Answered: {summary['answered_questions']}")
    print(f"Questions Evaluated: {summary['evaluated_questions']}")
    print(f"Overall Average: {summary['overall_average']}%")
    print(f"Total Score: {summary['total_achieved_score']}/{summary['total_possible_score']}")
    
    # Individual question results
    print(f"\nDETAILED RESULTS:")
    print("-" * 80)
    
    for question_id, result in results["individual_results"].items():
        score = result['percentage_score']
        score_str = f"{score}%" if isinstance(score, (int, float)) else str(score)
        print(f"\n{question_id.upper()}: {score_str}")


def main():
    """Main function to run the evaluation with command line arguments"""
    import sys
    import os
    
    print("Starting Answer Evaluator...")
    
    # Check if file paths are provided as command line arguments
    if len(sys.argv) == 3:
        student_file = sys.argv[1]
        answer_key_file = sys.argv[2]
    elif len(sys.argv) == 1:
        # Default file paths if no arguments provided
        student_file = "student_answer.json"
        answer_key_file = "answer_key.json"
    else:
        print("Usage: python answer_evaluator.py [student_answers.json] [answer_key.json]")
        print("Or run without arguments to use default files:")
        print("  - student_answer.json")
        print("  - student_answer_key.json")
        return
    
    # Check if files exist
    print(f"\nCurrent directory: {os.getcwd()}")
    print(f"Looking for files:")
    print(f"  - Student answers: {student_file}")
    print(f"  - Answer key: {answer_key_file}")
    
    if not os.path.exists(student_file):
        print(f"ERROR: {student_file} not found!")
        return
    
    if not os.path.exists(answer_key_file):
        print(f"ERROR: {answer_key_file} not found!")
        return
    
    # Run evaluation
    print(f"\nStarting evaluation...")
    print(f"Student answers file: {student_file}")
    print(f"Answer key file: {answer_key_file}")
    print("-" * 50)
    
    try:
        results = evaluate_from_json_files(student_file, answer_key_file)
        
        if results:
            generate_detailed_report(results)
            
            # Optional: Save results to JSON file
            output_file = "evaluation_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed results saved to '{output_file}'")
        else:
            print("Evaluation failed. Please check your JSON files.")
            
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
