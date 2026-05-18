
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image, ImageStat, ImageFilter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Student Stress Analysis", page_icon="🧠", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {display: none;}

.stApp {
    background: linear-gradient(135deg, #f8fafc, #e0f2fe);
    color: #0f172a;
}

.block-container {
    padding-top: 0rem;
    padding-left: 1.6rem;
    padding-right: 1.6rem;
}

.top-nav {
    background: linear-gradient(90deg, #334155, #475569);
    padding: 22px 30px;
    border-radius: 0 0 28px 28px;
    margin-bottom: 24px;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.22);
}

.brand {
    color: #ffffff;
    font-size: 34px;
    font-weight: 900;
    margin-bottom: 18px;
}

.nav-buttons {
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
}

.nav-item {
    background: #ffffff;
    color: #0f172a;
    padding: 14px 23px;
    border-radius: 16px;
    font-weight: 800;
    text-align: center;
    box-shadow: 0 4px 12px rgba(15,23,42,0.18);
}

.hero-card {
    background: linear-gradient(120deg, #ffffff, #f1f5f9);
    color: #0f172a;
    padding: 38px;
    border-radius: 30px;
    margin-bottom: 24px;
    border: 1px solid #cbd5e1;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.14);
}

.hero-title {
    font-size: 44px;
    font-weight: 900;
    margin-bottom: 12px;
    color: #0f172a;
}

.hero-subtitle {
    font-size: 20px;
    line-height: 1.7;
    color: #334155;
    font-weight: 650;
}

.section-card {
    background: #ffffff;
    padding: 28px;
    border-radius: 24px;
    box-shadow: 0 8px 24px rgba(15,23,42,0.12);
    margin-bottom: 22px;
    color: #0f172a;
    border: 1px solid #e2e8f0;
}

.section-card h2, .section-card h3 {
    color: #0f172a;
    font-weight: 900;
}

.section-card p {
    color: #1f2937;
    font-size: 17px;
    font-weight: 600;
}

.module-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 18px;
    margin-top: 18px;
}

.module-box {
    background: #f8fafc;
    border-left: 6px solid #64748b;
    padding: 18px;
    border-radius: 18px;
    color: #0f172a;
    min-height: 145px;
    box-shadow: 0 4px 14px rgba(15,23,42,0.08);
}

.module-box h4 {
    color: #0f172a;
    font-size: 18px;
    font-weight: 900;
    margin-bottom: 8px;
}

.module-box p {
    color: #334155;
    font-size: 14px;
    font-weight: 700;
}

.metric-card {
    background: #ffffff;
    padding: 24px;
    border-radius: 22px;
    box-shadow: 0 8px 20px rgba(15,23,42,0.12);
    text-align: center;
    border-top: 6px solid #64748b;
}

.metric-value {
    font-size: 34px;
    font-weight: 900;
    color: #334155;
}

.metric-label {
    font-size: 16px;
    font-weight: 800;
    color: #111827;
}

.result-box {
    background: #ffffff;
    padding: 24px;
    border-radius: 22px;
    border: 1px solid #cbd5e1;
    box-shadow: 0 8px 22px rgba(15,23,42,0.12);
    text-align: center;
}

.result-percent {
    font-size: 48px;
    font-weight: 900;
    color: #0f172a;
}

.footer {
    text-align: center;
    color: #334155;
    font-weight: 800;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        return pd.read_csv("student_stress_cleaned.csv")
    except FileNotFoundError:
        return pd.read_csv("student_stress_dataset.csv")

df = load_data()

def image_based_stress(photo_file, age):
    image = Image.open(photo_file).convert("RGB")
    gray = image.convert("L")

    stat = ImageStat.Stat(gray)
    brightness = stat.mean[0]
    contrast = stat.stddev[0]

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    edge_level = edge_stat.mean[0]

    brightness_factor = abs(brightness - 135) / 135 * 28
    contrast_factor = min(contrast / 90 * 28, 28)
    edge_factor = min(edge_level / 45 * 24, 24)
    age_factor = max(0, min((age - 18) * 1.2, 10))

    raw_score = 22 + brightness_factor + contrast_factor + edge_factor + age_factor
    stress_percent = int(np.clip(raw_score, 5, 96))

    if stress_percent < 40:
        level_text = "Low"
    elif stress_percent < 70:
        level_text = "Medium"
    else:
        level_text = "High"

    return stress_percent, level_text, brightness, contrast, edge_level

st.markdown("""
<div class="top-nav">
    <div class="brand">🧠 Student Stress Analysis System</div>
    <div class="nav-buttons">
        <div class="nav-item">🏠 Overview</div>
        <div class="nav-item">📊 Analytics Dashboard</div>
        <div class="nav-item">🔮 Stress Prediction</div>
        <div class="nav-item">📷 Face Biometric Stress</div>
        <div class="nav-item">📄 Dataset</div>
    </div>
</div>
""", unsafe_allow_html=True)

page = st.radio(
    "Navigation",
    ["🏠 Overview", "📊 Analytics Dashboard", "🔮 Stress Prediction", "📷 Face Biometric Stress", "📄 Dataset"],
    horizontal=True,
    label_visibility="collapsed"
)

if page == "🏠 Overview":
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">Student Stress Analysis System</div>
        <div class="hero-subtitle">
            Analyze student lifestyle, sleep routine, screen time, academic pressure, and social support
            to understand stress patterns using data analytics and machine learning.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Records</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{round(df["Stress_Score"].mean(), 2)}</div><div class="metric-label">Average Stress Score</div></div>', unsafe_allow_html=True)
    with c3:
        high_count = df[df["Stress_Level"] == "High"].shape[0]
        st.markdown(f'<div class="metric-card"><div class="metric-value">{high_count}</div><div class="metric-label">High Stress Records</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <h3>Project Modules</h3>
        <div class="module-grid">
            <div class="module-box">
                <h4>🏠 Overview</h4>
                <p>Shows project summary and important KPI cards.</p>
            </div>
            <div class="module-box">
                <h4>📊 Analytics</h4>
                <p>Interactive charts, filters, and visual insights.</p>
            </div>
            <div class="module-box">
                <h4>🔮 Prediction</h4>
                <p>Classifies stress level using a machine learning model.</p>
            </div>
            <div class="module-box">
                <h4>📷 Biometric</h4>
                <p>Camera-based image analysis for stress percentage display.</p>
            </div>
            <div class="module-box">
                <h4>📄 Dataset</h4>
                <p>Displays complete dataset with download option.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "📊 Analytics Dashboard":
    st.markdown('<div class="section-card"><h2>📊 Analytics Dashboard</h2></div>', unsafe_allow_html=True)

    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        gender_filter = st.multiselect("Gender", df["Gender"].unique(), default=df["Gender"].unique())
    with colf2:
        stress_filter = st.multiselect("Stress Level", df["Stress_Level"].unique(), default=df["Stress_Level"].unique())
    with colf3:
        age_range = st.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))

    filtered_df = df[
        (df["Gender"].isin(gender_filter)) &
        (df["Stress_Level"].isin(stress_filter)) &
        (df["Age"].between(age_range[0], age_range[1]))
    ]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", len(filtered_df))
    c2.metric("Average Stress", round(filtered_df["Stress_Score"].mean(), 2))
    c3.metric("Average Sleep", round(filtered_df["Sleep_Hours"].mean(), 2))
    c4.metric("Average Screen Time", round(filtered_df["Screen_Time_Hours"].mean(), 2))

    left, right = st.columns(2)
    with left:
        fig1 = px.pie(filtered_df, names="Stress_Level", title="Stress Level Distribution", hole=0.45)
        st.plotly_chart(fig1, use_container_width=True)
    with right:
        avg_gender = filtered_df.groupby("Gender", as_index=False)["Stress_Score"].mean()
        fig2 = px.bar(avg_gender, x="Gender", y="Stress_Score", title="Average Stress Score by Gender", text_auto=True)
        st.plotly_chart(fig2, use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        fig3 = px.scatter(filtered_df, x="Sleep_Hours", y="Stress_Score", color="Stress_Level", title="Sleep Hours vs Stress Score", size="Academic_Pressure_Score")
        st.plotly_chart(fig3, use_container_width=True)
    with right2:
        fig4 = px.scatter(filtered_df, x="Screen_Time_Hours", y="Stress_Score", color="Stress_Level", title="Screen Time vs Stress Score", size="Study_Hours")
        st.plotly_chart(fig4, use_container_width=True)

    corr = filtered_df.select_dtypes(include="number").corr()
    fig5 = px.imshow(corr, text_auto=True, title="Correlation Heatmap", aspect="auto")
    st.plotly_chart(fig5, use_container_width=True)

elif page == "🔮 Stress Prediction":
    st.markdown('<div class="section-card"><h2>🔮 Stress Prediction</h2><p>Enter student lifestyle and academic details to classify stress level.</p></div>', unsafe_allow_html=True)

    features = ["Sleep_Hours", "Study_Hours", "Screen_Time_Hours", "Exercise_Days_Per_Week", "Social_Support_Score", "Academic_Pressure_Score", "Financial_Pressure_Score"]
    X = df[features]
    y = df["Stress_Level"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    st.success(f"Model Accuracy: {round(accuracy * 100, 2)}%")

    c1, c2, c3 = st.columns(3)
    with c1:
        input_sleep = st.slider("Sleep Hours", 3.0, 9.0, 6.0)
        input_study = st.slider("Study Hours", 0.5, 10.0, 4.0)
        input_screen = st.slider("Screen Time Hours", 1.0, 12.0, 5.0)
    with c2:
        input_exercise = st.slider("Exercise Days Per Week", 0, 6, 2)
        input_social = st.slider("Social Support Score", 1, 5, 3)
    with c3:
        input_academic = st.slider("Academic Pressure Score", 1, 5, 3)
        input_financial = st.slider("Financial Pressure Score", 1, 5, 3)

    if st.button("Predict Stress Level"):
        user_data = pd.DataFrame([[input_sleep, input_study, input_screen, input_exercise, input_social, input_academic, input_financial]], columns=features)
        result = model.predict(user_data)[0]

        if result == "High":
            st.error(f"Predicted Stress Level: {result}")
            st.write("Suggestion: Improve sleep routine, reduce screen time, take breaks, and maintain a healthy study routine.")
        elif result == "Medium":
            st.warning(f"Predicted Stress Level: {result}")
            st.write("Suggestion: Balance study, rest, exercise, and screen time.")
        else:
            st.success(f"Predicted Stress Level: {result}")
            st.write("Suggestion: Current routine looks balanced.")

elif page == "📷 Face Biometric Stress":
    st.markdown("""
    <div class="section-card">
        <h2>📷 Face Biometric Stress Module</h2>
        <p>Capture a student photo and generate a stress percentage using image brightness, contrast, edge level, and age factor.</p>
    </div>
    """, unsafe_allow_html=True)

    age = st.number_input("Enter Age", min_value=10, max_value=80, value=20)
    photo = st.camera_input("Take Student Photo")

    if photo is not None:
        stress_percent, stress_level, brightness, contrast, edge_level = image_based_stress(photo, age)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(photo, caption="Captured Image", use_container_width=True)
        with col2:
            st.markdown(f"""
            <div class="result-box">
                <h2>Stress Analysis Result</h2>
                <div class="result-percent">{stress_percent}%</div>
                <h3>Stress Level: {stress_level}</h3>
            </div>
            """, unsafe_allow_html=True)

            if stress_level == "High":
                st.error("Feedback: Take a short break, improve sleep routine, and reduce screen time.")
            elif stress_level == "Medium":
                st.warning("Feedback: Maintain balance between study, sleep, exercise, and relaxation.")
            else:
                st.success("Feedback: Routine looks balanced. Continue healthy habits.")

        with st.expander("View Image Analysis Values"):
            st.write("Brightness:", round(brightness, 2))
            st.write("Contrast:", round(contrast, 2))
            st.write("Edge Level:", round(edge_level, 2))
            st.write("Age Factor Included:", age)

elif page == "📄 Dataset":
    st.markdown('<div class="section-card"><h2>📄 Dataset</h2><p>View and download the student stress dataset.</p></div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download Dataset", data=csv, file_name="student_stress_dataset.csv", mime="text/csv")

st.markdown('<div class="footer">Student Stress Analysis Project</div>', unsafe_allow_html=True)
