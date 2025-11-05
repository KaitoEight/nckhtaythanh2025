import streamlit as st
import pandas as pd
import joblib

# Load mô hình, scaler và danh sách cột
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# CSS thêm ảnh nền + overlay tối + input đẹp
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    position: relative;
    background-image: url("https://res.cloudinary.com/dfclgysp0/image/upload/v1762341533/unnamed_1_am5xfj.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-size: 70% 100%;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.45);
    z-index: 0;
}
[data-testid="stAppViewContainer"] > div {
    position: relative;
    z-index: 1;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.85);
}
input, textarea, select, .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] * {
    color: #000 !important;
    background-color: rgba(255,255,255,0.9) !important;
    border: 1px solid #ccc !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}
::placeholder {
    color: #666 !important;
    opacity: 1 !important;
}
input:focus, textarea:focus, select:focus {
    border-color: #4CAF50 !important;
    box-shadow: 0 0 5px rgba(76,175,80,0.4) !important;
}
div.stButton > button:first-child {
    background: linear-gradient(90deg, #4CAF50, #2E7D32);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: bold;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    transition: 0.3s;
}
div.stButton > button:first-child:hover {
    background: linear-gradient(90deg, #66BB6A, #388E3C);
    box-shadow: 0px 6px 14px rgba(0,0,0,0.3);
    transform: translateY(-2px);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Giao diện chính
st.title("🧠 Bộ Trắc Nghiệm Trầm Cảm")

# Nhập liệu
age = st.number_input("Tuổi của bạn", min_value=10, max_value=100, value=20)
gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
city = st.selectbox("Thành phố", ["Thành phố 1", "Thành phố 2", "Thành phố 3"])
profession = st.selectbox("Nghề nghiệp", ["Học sinh/Sinh viên", "Làm thêm", "Làm toàn thời gian"])
academic_pressure = st.slider("Mức độ áp lực học tập", 0, 10, 5)
work_pressure = st.slider("Mức độ áp lực công việc", 0, 10, 5)
cgpa = st.number_input("Điểm trung bình (CGPA)", min_value=0.0, max_value=4.0, value=3.0)
study_satisfaction = st.slider("Mức độ hài lòng với việc học", 0, 10, 5)
job_satisfaction = st.slider("Mức độ hài lòng với công việc", 0, 10, 5)
sleep_duration = st.number_input("Thời gian ngủ trung bình mỗi ngày (giờ)", min_value=0.0, max_value=24.0, value=6.0)
dietary_habits = st.selectbox("Thói quen ăn uống", ["Lành mạnh", "Trung bình", "Không lành mạnh"])
degree = st.selectbox("Trình độ học vấn", ["Trung học", "Cử nhân", "Thạc sĩ"])
suicidal_thoughts = st.selectbox("Bạn đã từng có ý nghĩ tự tử?", ["Có", "Không"])
work_study_hours = st.number_input("Số giờ học/làm việc mỗi ngày", min_value=0.0, max_value=24.0, value=8.0)
financial_stress = st.slider("Mức độ áp lực tài chính", 0, 10, 5)
family_history = st.selectbox("Gia đình có tiền sử bệnh tâm lý không?", ["Có", "Không"])

# Xử lý input (mapping về giá trị gốc tiếng Anh để khớp với feature_columns)
input_data = {
    'Age': age,
    'Gender': 1 if gender == "Nam" else 0,
    'Work Pressure': work_pressure,
    'Academic Pressure': academic_pressure,
    'CGPA': cgpa,
    'Study Satisfaction': study_satisfaction,
    'Job Satisfaction': job_satisfaction,
    'Sleep Duration': sleep_duration,
    'Work/Study Hours': work_study_hours,
    'Financial Stress': financial_stress,
    'Have you ever had suicidal thoughts ?': 1 if suicidal_thoughts == "Có" else 0,
    'Family History of Mental Illness': 1 if family_history == "Có" else 0,
    f'City_City{["Thành phố 1","Thành phố 2","Thành phố 3"].index(city)+1}': 1,
    f'Profession_{"Student" if profession=="Học sinh/Sinh viên" else "Part-time" if profession=="Làm thêm" else "Full-time"}': 1,
    f'Dietary Habits_{"Healthy" if dietary_habits=="Lành mạnh" else "Moderate" if dietary_habits=="Trung bình" else "Unhealthy"}': 1,
    f'Degree_{"High School" if degree=="Trung học" else "Bachelor" if degree=="Cử nhân" else "Master"}': 1
}

input_df = pd.DataFrame([input_data])

# Điền các cột còn lại bằng 0
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Sắp xếp lại cột theo thứ tự
input_df = input_df[feature_columns]

# Chuẩn hóa dữ liệu
numeric_columns = ['Age', 'CGPA', 'Sleep Duration', 'Work Pressure', 'Academic Pressure',
                   'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])

# Dự đoán
if st.button("🩺 Dự đoán"):
    prediction = model.predict(input_df)
    result = "🟠 Có khả năng bị trầm cảm" if prediction[0] == 1 else "🟢 Không có dấu hiệu trầm cảm"
    st.subheader(f"Kết quả: {result}")
