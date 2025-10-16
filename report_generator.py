#!/usr/bin/env python3
"""
Generate human-readable feedback reports from evaluation_results.json
"""

import json
from datetime import datetime

def generate_question_feedback(q_num, result):
    """Generate detailed feedback for a single question."""
    question = result.get("question", "N/A")
    student_answer = result.get("student_answer", "No answer provided")
    expected_answer = result.get("expected_answer", "No reference answer")
    percentage_score = result.get("percentage_score", 0)

    # Include image info if available
    has_student_image = result.get("has_student_image", False)
    has_reference_image = result.get("has_reference_image", False)
    image_similarity = None
    if "evaluation_details" in result:
        image_similarity = result["evaluation_details"].get("image_similarity")

    feedback = f"""
## Question {q_num}
**Question:** {question}

**Your Answer:** {student_answer}
**Reference Answer:** {expected_answer}

**Score:** {percentage_score}%
"""
    if has_student_image or has_reference_image:
        feedback += f"**Image Provided:** Student: {has_student_image}, Reference: {has_reference_image}\n"
        if image_similarity is not None:
            feedback += f"**Image Similarity:** {image_similarity}\n"

    return feedback + "\n---\n"

def generate_summary(results):
    """Print console summary based on individual_results and summary."""
    individual_results = results.get("individual_results", {})
    summary = results.get("summary", {})

    print("=" * 50)
    print("           EVALUATION SUMMARY")
    print("=" * 50)

    total_score = 0
    total_questions = len(individual_results)

    for q_num, res in individual_results.items():
        score = res.get("percentage_score", 0)
        print(f"{q_num}: {score}%")
        total_score += score

    avg_score = total_score / total_questions if total_questions else 0
    print("-" * 50)
    print(f"Average Score: {avg_score:.1f}%")

    # Simple overall rating
    if avg_score >= 90:
        overall = "Excellent!"
    elif avg_score >= 80:
        overall = "Very Good!"
    elif avg_score >= 70:
        overall = "Good"
    elif avg_score >= 60:
        overall = "Satisfactory"
    else:
        overall = "Needs Improvement"
    print(f"Overall: {overall}")
    print("=" * 50)

def generate_full_report(results_json_path, output_path="feedback_report.md"):
    """Generate full Markdown report for all questions."""
    with open(results_json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    individual_results = results.get("individual_results", {})
    summary = results.get("summary", {})

    report = f"# Student Answer Evaluation Report\n"
    report += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += f"## Overall Summary\n"
    report += f"- Total Questions: {summary.get('total_questions', len(individual_results))}\n"
    report += f"- Questions Answered: {summary.get('answered_questions', 0)}\n"
    report += f"- Questions Evaluated: {summary.get('evaluated_questions', 0)}\n"
    report += f"- Overall Average Score: {summary.get('overall_average', 0)}%\n"
    report += f"- Total Achieved Score: {summary.get('total_achieved_score', 0)}\n"
    report += f"- Total Possible Score: {summary.get('total_possible_score', 0)}\n\n"

    report += "## Detailed Question Feedback\n"

    for q_num, res in individual_results.items():
        report += generate_question_feedback(q_num, res)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n📄 Detailed feedback report saved to: {output_path}")
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate feedback report from evaluation_results.json")
    parser.add_argument("--results", required=True, help="Path to evaluation_results.json")
    parser.add_argument("--output", default="feedback_report.md", help="Output Markdown report file")
    parser.add_argument("--summary-only", action="store_true", help="Only print console summary")
    args = parser.parse_args()

    results = generate_full_report(args.results, args.output) if not args.summary_only else None

    if args.summary_only:
        with open(args.results, 'r', encoding='utf-8') as f:
            results = json.load(f)
    generate_summary(results)
