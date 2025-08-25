# bloom_utils.py
import re

def classify_bloom(question, answer):
    question = question.lower()
    answer = answer.lower()

    # --- Detect Bloom level from question ---
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

    # --- Detect Bloom level from answer ---
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

    # --- Resolve final Bloom level ---
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
