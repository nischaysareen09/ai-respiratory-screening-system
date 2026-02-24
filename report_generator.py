def risk_label(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    else:
        return "High"


def generate_report(metadata, cough_count, wet_count, dry_count,
                    wetness_percentage, predictions):

    report = f"""
==============================
AI Respiratory Screening Report
==============================

Patient Information:
Age: {metadata['age']}
Gender: {metadata['gender']}

Cough Analysis:
Total Cough Events: {cough_count}
Wet Cough Count: {wet_count}
Dry Cough Count: {dry_count}
Wetness Percentage: {wetness_percentage:.2f}%

Disease Risk Assessment:
"""

    for disease, prob in predictions.items():
        report += f"{disease.upper()} Probability: {prob} ({risk_label(prob)} Risk)\n"

    report += """
--------------------------------
Clinical Note:
This is an AI-based screening tool.
Consult a healthcare professional for medical advice.
"""

    return report