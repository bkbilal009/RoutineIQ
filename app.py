import joblib
import json
import pandas as pd
import gradio as gr

# Load pipeline and metadata
pipe = joblib.load("exam_score_pipeline.joblib")
with open("metadata.json") as f:
    meta = json.load(f)

id_col = meta["id_col"]
target = meta["target"]
candidate_features = meta["features"]

# Simple recommender (same logic you used in Colab)
def recommend_actions(student_row: pd.Series):
    recs = []
    if student_row.get("study_hours_per_day", 0) < 3:
        recs.append("Increase study hours by 1–2 daily")
    if student_row.get("social_media_hours", 0) > 3:
        recs.append("Reduce social media usage below 2 hours")
    if student_row.get("sleep_hours", 0) < 6:
        recs.append("Improve sleep hygiene (7–8 hours)")
    return recs or ["No major changes suggested"]

def predict_custom(
    age, gender, study_hours_per_day, social_media_hours, netflix_hours,
    part_time_job, attendance_percentage, sleep_hours, diet_quality,
    exercise_frequency, parental_education_level, internet_quality,
    mental_health_rating, extracurricular_participation
):
    row = {
        "age": age,
        "gender": gender,
        "study_hours_per_day": study_hours_per_day,
        "social_media_hours": social_media_hours,
        "netflix_hours": netflix_hours,
        "part_time_job": 1 if part_time_job else 0,
        "attendance_percentage": attendance_percentage,
        "sleep_hours": sleep_hours,
        "diet_quality": diet_quality,
        "exercise_frequency": exercise_frequency,
        "parental_education_level": parental_education_level,
        "internet_quality": internet_quality,
        "mental_health_rating": mental_health_rating,
        "extracurricular_participation": 1 if extracurricular_participation else 0
    }
    row_series = pd.Series(row)
    pred = float(pipe.predict(pd.DataFrame([row]))[0])
    recs = recommend_actions(row_series)
    return f"Predicted score: {pred:.1f}", "\n".join(recs)

with gr.Blocks(title="StudentPathfinder") as demo:
    gr.Markdown("# 🎓 StudentPathfinder — Opportunity & Insights")
    gr.Markdown("Predict exam scores and discover improvement opportunities.")

    with gr.Row():
        age = gr.Slider(16, 30, value=20, step=1, label="Age")
        gender = gr.Dropdown(["Male", "Female", "Other"], value="Male", label="Gender")
        study_hours_per_day = gr.Slider(0, 10, value=3, step=0.1, label="Study hours/day")
        social_media_hours = gr.Slider(0, 8, value=2, step=0.1, label="Social media hours")
        netflix_hours = gr.Slider(0, 8, value=1, step=0.1, label="Netflix hours")
        part_time_job = gr.Checkbox(label="Part-time job?")
        attendance_percentage = gr.Slider(50, 100, value=85, step=0.1, label="Attendance %")
        sleep_hours = gr.Slider(3, 10, value=7, step=0.1, label="Sleep hours")
        diet_quality = gr.Dropdown(["Poor", "Fair", "Good"], value="Fair", label="Diet quality")
        exercise_frequency = gr.Slider(0, 10, value=3, step=1, label="Exercise frequency/week")
        parental_education_level = gr.Dropdown(["None", "High School", "Bachelor", "Master"], value="High School", label="Parental education")
        internet_quality = gr.Dropdown(["Poor", "Average", "Good"], value="Good", label="Internet quality")
        mental_health_rating = gr.Slider(1, 10, value=6, step=1, label="Mental health rating")
        extracurricular_participation = gr.Checkbox(label="Extracurricular participation?")

    btn = gr.Button("Predict & Recommend")
    out_pred = gr.Textbox(label="Predicted Score")
    out_recs = gr.Textbox(label="Recommendations")

    btn.click(
        fn=predict_custom,
        inputs=[
            age, gender, study_hours_per_day, social_media_hours, netflix_hours,
            part_time_job, attendance_percentage, sleep_hours, diet_quality,
            exercise_frequency, parental_education_level, internet_quality,
            mental_health_rating, extracurricular_participation
        ],
        outputs=[out_pred, out_recs]
    )

demo.launch()
