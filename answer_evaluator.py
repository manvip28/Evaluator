# answer_evaluator.py

from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
def extract_keywords(text, domain_keywords=None):
    tokens = word_tokenize(text.lower())
    words = [word for word in tokens if word.isalnum() and word not in stop_words]
    if domain_keywords:
        matches = [kw for kw in domain_keywords if kw.lower() in text.lower()]
        return matches
    return words

def classify_bloom(question, answer):
    question = question.lower()
    answer = answer.lower()
    if re.search(r'\b(define|list|name|identify|recall|state|who|what|when|where)\b', question):
        question_level = "Remember"
    elif re.search(r'\b(explain|describe|compare|contrast|summarize|interpret|paraphrase)\b', question):
        question_level = "Understand"
    elif re.search(r'\b(apply|use|demonstrate|illustrate|solve|implement|design)\b', question):
        question_level = "Apply"
    elif re.search(r'\b(analyze|differentiate|organize|attribute|distinguish|examine)\b', question):
        question_level = "Analyze"
    elif re.search(r'\b(evaluate|assess|critique|judge|justify|recommend)\b', question):
        question_level = "Evaluate"
    elif re.search(r'\b(create|design|construct|plan|produce|develop|formulate)\b', question):
        question_level = "Create"
    else:
        question_level = None

    if re.search(r'\b(is|are|was|were|means)\b', answer) and len(answer.split()) < 20:
        answer_level = "Remember"
    elif re.search(r'\b(because|since|as|consists of|includes)\b', answer):
        answer_level = "Understand"
    elif re.search(r'\b(can be used to|applied|implemented|designed|built)\b', answer):
        answer_level = "Apply"
    elif re.search(r'\b(compared to|differs from|analysis|relationship|impact of|effect of)\b', answer):
        answer_level = "Analyze"
    elif re.search(r'\b(better|worse|more effective|less efficient|advantages|disadvantages|pros|cons)\b', answer):
        answer_level = "Evaluate"
    elif re.search(r'\b(new|novel|innovative|created|designed|developed|proposed)\b', answer):
        answer_level = "Create"
    else:
        answer_level = "Understand"

    bloom_hierarchy = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
    if question_level and answer_level:
        q_idx = bloom_hierarchy.index(question_level)
        a_idx = bloom_hierarchy.index(answer_level)
        return bloom_hierarchy[max(q_idx, a_idx)]
    elif answer_level:
        return answer_level
    elif question_level:
        return question_level
    else:
        return "Understand"

def keyword_coverage_score(student_text, reference_keywords):
    if not reference_keywords:
        return 1.0
    matches = sum(1 for keyword in reference_keywords if keyword.lower() in student_text.lower())
    return matches / len(reference_keywords)

def evaluate_answer(gt_question, gt_answer, stu_answer, bloom_gt=None, keywords=None):
    """
    Evaluate a single student answer against the ground truth.
    
    Args:
        gt_question: Ground truth question text
        gt_answer: Ground truth answer text
        stu_answer: Student's answer text
        bloom_gt: Expected Bloom's taxonomy level (optional)
        keywords: List of expected keywords (optional)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Handle empty or None answers
    if not stu_answer or not stu_answer.strip():
        return {
            "semantic_score": 0.0,
            "bleu": 0.0,
            "rouge_l": 0.0,
            "keyword_coverage": 0.0,
            "bloom_classified": "Remember",
            "bloom_expected": bloom_gt or "Understand",
            "bloom_penalty": 0.0,
            "raw_score": 0.0,
            "final_score": 0.0
        }

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

    # Handle Bloom's taxonomy penalty
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

    alpha, beta, gamma, delta = 0.4, 0.2, 0.3, 0.1
    traditional_score = (bleu_score + rouge_score) / 2
    final_score = (alpha * sem_score) + (beta * traditional_score) + (gamma * kw_coverage) - (delta * penalty)
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
        "raw_score": round(final_score, 4),
        "final_score": round(curved_score, 4)
    }

def evaluate_text_answer(student_answer, reference_answer, question_text="", bloom_level=None, keywords=None):
    """
    Simplified interface for text evaluation.
    Returns a score between 0 and 1.
    """
    result = evaluate_answer(question_text, reference_answer, student_answer, bloom_level, keywords)
    return result["final_score"]

# -------------------- Sample Data for Testing --------------------
def run_sample_test():
    """Run the original sample test for verification"""
    answer_key = {
        "questions": [
            {
                "id": 1,
                "question": "Explain the architecture of ARM processor.",
                "answer": "The ARM architecture includes components like the processor core, interrupt controller, memory controller, and peripheral controllers connected through AHB and APB buses.",
                "bloom": "Understand",
                "keywords": ["processor core", "interrupt controller", "memory controller", "peripheral", "AHB", "APB", "buses"]
            },
            {
                "id": 2,
                "question": "Design a simple block diagram for embedded systems.",
                "answer": "The design includes components such as CPU, memory, I/O devices, timers, and buses. Communication occurs through system buses like AHB and APB.",
                "bloom": "Apply",
                "keywords": ["CPU", "memory", "I/O", "timers", "buses", "AHB", "APB"]
            },
            {
                "id": 3,
                "question": "Analyze how AHB and APB improve embedded system performance.",
                "answer": "AHB enables high-speed communication between major components, while APB connects low-speed peripherals. This separation improves efficiency and reduces bottlenecks.",
                "bloom": "Analyze",
                "keywords": ["AHB", "APB", "high-speed", "low-speed", "efficiency", "bottlenecks", "separation"]
            }
        ]
    }

    student_answers = {
        "answers": [
            {
                "id": 1,
                "answer": "ARM includes the processor core, memory controller, interrupt controller and uses AHB and APB buses to connect peripherals."
            },
            {
                "id": 2,
                "answer": "A diagram has CPU, memory, timers, I/O devices all connected using buses like AHB and APB."
            },
            {
                "id": 3,
                "answer": "AHB connects fast components and APB connects slow devices, so performance is better."
            }
        ]
    }

    results = []
    for gt_q in answer_key["questions"]:
        stu_ans = next((a for a in student_answers["answers"] if a["id"] == gt_q["id"]), None)
        if stu_ans:
            evaluation = evaluate_answer(gt_q["question"], gt_q["answer"], stu_ans["answer"], gt_q["bloom"], gt_q.get("keywords", []))
            percentage_score = round(evaluation["final_score"] * 100, 1)
            results.append({
                "question_id": gt_q["id"],
                "question": gt_q["question"],
                "model_answer": gt_q["answer"],
                "student_answer": stu_ans["answer"],
                "evaluation": evaluation,
                "percentage_score": percentage_score
            })

    for result in results:
        print(f"Question {result['question_id']}: {result['question']}")
        print(f"Model answer: {result['model_answer']}")
        print(f"Student answer: {result['student_answer']}")
        print(f"Score: {result['percentage_score']}%")
        print(f"Bloom's level: Expected '{result['evaluation']['bloom_expected']}', Classified as '{result['evaluation']['bloom_classified']}'")
        print("-" * 80)

# -------------------- Main Execution --------------------
if __name__ == "__main__":
    run_sample_test()