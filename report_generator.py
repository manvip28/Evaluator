#!/usr/bin/env python3
"""
Generate human-readable feedback reports from evaluation results.
"""

import json
import os
from datetime import datetime
from pathlib import Path

def generate_question_feedback(q_num, result, student_data, key_data):
    """Generate detailed feedback for a single question."""
    explanation = result.get("Explanation", {})
    
    feedback = f"""
## Question {q_num}

**Question:** {key_data.get('Question', 'N/A')}

**Your Score:** {explanation.get('percentage', 0):.1f}% ({explanation.get('grade_recommendation', 'N/A')})
**Overall Assessment:** {explanation.get('overall_comment', 'No assessment available')}

### Your Answer:
**Text:** {student_data.get('Text', 'No text answer provided')[:200]}{"..." if len(student_data.get('Text', '')) > 200 else ""}
**Image:** {"Provided" if student_data.get('Image') else "Not provided"}

### Score Breakdown:
- **Text Score:** {result.get('TextScore', 0):.3f} (Weight: 60%)
- **Image Score:** {result.get('ImageScore', 0):.3f} (Weight: 40%)
- **Final Score:** {result.get('FinalScore', 0):.3f}

"""
    
    # Add strengths
    strengths = explanation.get('strengths', [])
    if strengths:
        feedback += "### ✅ What You Did Well:\n"
        for strength in strengths:
            feedback += f"- {strength}\n"
        feedback += "\n"
    
    # Add weaknesses
    weaknesses = explanation.get('weaknesses', [])
    if weaknesses:
        feedback += "### ❌ Areas for Improvement:\n"
        for weakness in weaknesses:
            feedback += f"- {weakness}\n"
        feedback += "\n"
    
    # Add suggestions
    suggestions = explanation.get('suggestions', [])
    if suggestions:
        feedback += "### 💡 Suggestions for Next Time:\n"
        for suggestion in suggestions:
            feedback += f"- {suggestion}\n"
        feedback += "\n"
    
    # Add detailed explanations
    detailed_explanations = explanation.get('detailed_explanations', [])
    if detailed_explanations:
        feedback += "### 📊 Detailed Analysis:\n"
        for detail in detailed_explanations:
            feedback += f"- {detail}\n"
        feedback += "\n"
    
    # Add reference answer
    if key_data.get('Text'):
        feedback += f"### 📚 Reference Answer:\n{key_data['Text']}\n\n"
    
    feedback += "---\n"
    
    return feedback

def generate_summary_report(results, student_answers, answer_key):
    """Generate overall summary report."""
    summary = results.get('_summary', {})
    
    report = f"""
# 📊 Overall Performance Summary

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Questions:** {summary.get('total_questions', 0)}
**Questions Evaluated:** {summary.get('evaluated_questions', 0)}
**Average Score:** {summary.get('average_score', 0):.1f}%

## 🎯 Grade Distribution
"""
    
    # Calculate grade distribution
    grades = {}
    total_score = 0
    valid_scores = 0
    
    for q_num, result in results.items():
        if q_num.startswith('_'):
            continue
            
        explanation = result.get('Explanation', {})
        grade = explanation.get('grade_recommendation')
        percentage = explanation.get('percentage', 0)
        
        if grade:
            grades[grade] = grades.get(grade, 0) + 1
        
        if result.get('FinalScore') is not None:
            total_score += percentage
            valid_scores += 1
    
    for grade, count in sorted(grades.items()):
        report += f"- **{grade}:** {count} question(s)\n"
    
    if valid_scores > 0:
        avg_percentage = total_score / valid_scores
        report += f"\n**Overall Average:** {avg_percentage:.1f}%\n"
        
        if avg_percentage >= 90:
            overall_grade = "A+ (Excellent)"
        elif avg_percentage >= 85:
            overall_grade = "A (Very Good)"
        elif avg_percentage >= 80:
            overall_grade = "A- (Good)"
        elif avg_percentage >= 75:
            overall_grade = "B+ (Above Average)"
        elif avg_percentage >= 70:
            overall_grade = "B (Satisfactory)"
        elif avg_percentage >= 65:
            overall_grade = "B- (Below Average)"
        elif avg_percentage >= 60:
            overall_grade = "C+ (Poor)"
        else:
            overall_grade = "C- or below (Very Poor)"
        
        report += f"**Overall Grade:** {overall_grade}\n\n"
    
    # Performance insights
    report += "## 🔍 Performance Insights\n\n"
    
    text_scores = []
    image_scores = []
    
    for q_num, result in results.items():
        if q_num.startswith('_'):
            continue
        
        if result.get('TextScore') is not None:
            text_scores.append(result['TextScore'])
        if result.get('ImageScore') is not None:
            image_scores.append(result['ImageScore'])
    
    if text_scores:
        avg_text = sum(text_scores) / len(text_scores) * 100
        report += f"**Text Performance:** {avg_text:.1f}% average\n"
        
        if avg_text < 60:
            report += "- ❌ **Written answers need significant improvement**\n"
            report += "- 💡 Focus on understanding key concepts and using proper terminology\n"
        elif avg_text < 80:
            report += "- ⚠️ **Written answers are adequate but could be better**\n"
            report += "- 💡 Include more details and technical terms in your explanations\n"
        else:
            report += "- ✅ **Strong written communication skills**\n"
    
    if image_scores:
        avg_image = sum(image_scores) / len(image_scores) * 100
        report += f"**Visual Performance:** {avg_image:.1f}% average\n"
        
        if avg_image < 60:
            report += "- ❌ **Diagrams and visual answers need improvement**\n"
            report += "- 💡 Practice drawing clear, accurate diagrams with proper labels\n"
        elif avg_image < 80:
            report += "- ⚠️ **Visual answers are adequate but could be clearer**\n"
            report += "- 💡 Add more detail and ensure diagrams match the expected format\n"
        else:
            report += "- ✅ **Excellent visual representation skills**\n"
    
    return report

def generate_full_report(results_json_path, student_json_path, key_json_path, output_path="feedback_report.md"):
    """Generate a complete feedback report."""
    
    # Load all data
    with open(results_json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    with open(student_json_path, 'r', encoding='utf-8') as f:
        student_answers = json.load(f)
    
    with open(key_json_path, 'r', encoding='utf-8') as f:
        answer_key = json.load(f)
    
    # Generate full report
    full_report = "# 📋 Student Answer Evaluation Report\n"
    
    # Add summary
    full_report += generate_summary_report(results, student_answers, answer_key)
    
    # Add individual question feedback
    full_report += "\n\n# 📝 Detailed Question-by-Question Feedback\n"
    
    question_nums = [q for q in results.keys() if not q.startswith('_')]
    question_nums.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or '0'))
    
    for q_num in question_nums:
        result = results[q_num]
        student_data = student_answers.get(q_num, {})
        key_data = answer_key.get(q_num, {})
        
        question_feedback = generate_question_feedback(q_num, result, student_data, key_data)
        full_report += question_feedback
    
    # Add study recommendations
    full_report += generate_study_recommendations(results, answer_key)
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_report)
    
    print(f"📄 Detailed feedback report saved to: {output_path}")
    return output_path

def generate_study_recommendations(results, answer_key):
    """Generate personalized study recommendations."""
    recommendations = "\n\n# 📚 Personalized Study Recommendations\n\n"
    
    # Analyze weak areas
    weak_topics = []
    bloom_levels_needed = []
    
    for q_num, result in results.items():
        if q_num.startswith('_'):
            continue
        
        explanation = result.get('Explanation', {})
        percentage = explanation.get('percentage', 0)
        
        if percentage < 70:  # Below satisfactory
            key_data = answer_key.get(q_num, {})
            question = key_data.get('Question', '')
            bloom_level = key_data.get('BloomLevel', '')
            keywords = key_data.get('Keywords', [])
            
            weak_topics.extend(keywords)
            if bloom_level:
                bloom_levels_needed.append(bloom_level)
    
    if weak_topics:
        # Count frequency of weak keywords
        keyword_freq = {}
        for keyword in weak_topics:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        # Sort by frequency
        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
        
        recommendations += "## 🎯 Focus Areas for Improvement:\n\n"
        for keyword, freq in sorted_keywords[:5]:  # Top 5 weak areas
            recommendations += f"- **{keyword.title()}**: Appeared in {freq} question(s) with low scores\n"
    
    # Bloom's taxonomy recommendations
    if bloom_levels_needed:
        level_freq = {}
        for level in bloom_levels_needed:
            level_freq[level] = level_freq.get(level, 0) + 1
        
        recommendations += "\n## 🧠 Cognitive Skills to Develop:\n\n"
        for level, freq in level_freq.items():
            recommendations += f"- **{level} Level Skills**: Practice more {level.lower()}-level questions\n"
            
            if level == "Remember":
                recommendations += "  - Focus on memorizing key facts and definitions\n"
            elif level == "Understand":
                recommendations += "  - Practice explaining concepts in your own words\n"
            elif level == "Apply":
                recommendations += "  - Work on applying concepts to solve problems\n"
            elif level == "Analyze":
                recommendations += "  - Practice breaking down complex topics into parts\n"
            elif level == "Evaluate":
                recommendations += "  - Work on making judgments and assessments\n"
            elif level == "Create":
                recommendations += "  - Practice designing and creating new solutions\n"
    
    # General study tips
    recommendations += "\n## 💡 General Study Tips:\n\n"
    
    # Calculate text vs image performance
    text_scores = []
    image_scores = []
    
    for q_num, result in results.items():
        if q_num.startswith('_'):
            continue
        if result.get('TextScore') is not None:
            text_scores.append(result['TextScore'])
        if result.get('ImageScore') is not None:
            image_scores.append(result['ImageScore'])
    
    if text_scores and image_scores:
        avg_text = sum(text_scores) / len(text_scores) * 100
        avg_image = sum(image_scores) / len(image_scores) * 100
        
        if avg_text < avg_image:
            recommendations += "- 📝 **Focus on written explanations**: Your visual skills are stronger than written skills\n"
            recommendations += "  - Practice writing detailed explanations for concepts\n"
            recommendations += "  - Use technical vocabulary correctly\n"
            recommendations += "  - Structure your answers with clear introduction, body, and conclusion\n"
        elif avg_image < avg_text:
            recommendations += "- 🎨 **Improve visual representation**: Your written skills are stronger than visual skills\n"
            recommendations += "  - Practice drawing clear, labeled diagrams\n"
            recommendations += "  - Follow standard conventions for technical drawings\n"
            recommendations += "  - Include all necessary components in your diagrams\n"
        else:
            recommendations += "- ⚖️ **Balanced improvement needed**: Work on both written and visual skills equally\n"
    
    recommendations += "- 📖 **Review fundamental concepts**: Go back to textbook chapters covering weak areas\n"
    recommendations += "- 👥 **Study with peers**: Discuss difficult concepts with classmates\n"
    recommendations += "- 🔄 **Practice regularly**: Set aside time each day for review and practice\n"
    recommendations += "- ❓ **Ask for help**: Don't hesitate to ask teachers or tutors for clarification\n"
    
    return recommendations

def generate_simple_summary(results_json_path):
    """Generate a simple text summary for quick viewing."""
    with open(results_json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("\n" + "="*50)
    print("           EVALUATION SUMMARY")
    print("="*50)
    
    total_questions = 0
    total_score = 0
    
    for q_num, result in results.items():
        if q_num.startswith('_'):
            continue
            
        total_questions += 1
        explanation = result.get('Explanation', {})
        percentage = explanation.get('percentage', 0)
        grade = explanation.get('grade_recommendation', 'N/A')
        
        total_score += percentage
        
        print(f"{q_num:>4}: {percentage:>5.1f}% ({grade:>2})")
    
    if total_questions > 0:
        avg_score = total_score / total_questions
        print("-" * 50)
        print(f"{'Average':>4}: {avg_score:>5.1f}%")
        
        if avg_score >= 90:
            overall = "Excellent! 🌟"
        elif avg_score >= 80:
            overall = "Very Good! 👍"
        elif avg_score >= 70:
            overall = "Good 👌"
        elif avg_score >= 60:
            overall = "Satisfactory ⚡"
        else:
            overall = "Needs Improvement 📚"
        
        print(f"Overall: {overall}")
    
    print("="*50)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate feedback reports from evaluation results")
    parser.add_argument("--results", required=True, help="Path to results.json file")
    parser.add_argument("--student", required=True, help="Path to student_answers.json")
    parser.add_argument("--key", required=True, help="Path to answer_key.json")
    parser.add_argument("--output", default="feedback_report.md", help="Output report file")
    parser.add_argument("--summary-only", action="store_true", help="Show only quick summary")
    
    args = parser.parse_args()
    
    if args.summary_only:
        generate_simple_summary(args.results)
    else:
        generate_full_report(args.results, args.student, args.key, args.output)
        generate_simple_summary(args.results)