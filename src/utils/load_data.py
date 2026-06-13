import pandas as pd

from src.config import config


def load_toy_data():
    df = pd.read_csv(config.data_dir / "ai_student_impact_dataset.csv")
    df.describe()

    X_cols = [
        "Major_Category",
        "Year_of_Study",
        "Pre_Semester_GPA",
        "Weekly_GenAI_Hours",
        "Primary_Use_Case",
        "Prompt_Engineering_Skill",
        "Tool_Diversity",
        "Paid_Subscription",
        "Traditional_Study_Hours",
        "Perceived_AI_Dependency",
        "Institutional_Policy",
        "Anxiety_Level_During_Exams",
    ]
    # Post_Semester_GPA, Burnout_Risk_Level, Skill_Retention_Score
    y_cols = ["Burnout_Risk_Level"]

    X = df[X_cols]
    y = df[y_cols[0]]  # Series → 1D, not a DataFrame column-vector

    return X, y
