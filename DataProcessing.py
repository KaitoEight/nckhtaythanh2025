import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
data = pd.read_csv('Student_Depression_Dataset.csv')

# Kiểm tra giá trị thiếu
print(data.isnull().sum())

# Xử lý giá trị thiếu (ví dụ: điền giá trị trung bình cho các cột số)
data['CGPA'].fillna(data['CGPA'].mean(), inplace=True)
data['Sleep Duration'].fillna(data['Sleep Duration'].mean(), inplace=True)

# Mã hóa biến phân loại
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Depression_Status'] = label_encoder.fit_transform(data['Depression_Status'])

# One-Hot Encoding cho các cột như City, Profession
data = pd.get_dummies(data, columns=['City', 'Profession', 'Dietary Habits'], drop_first=True)

# Chuẩn hóa các cột số
scaler = StandardScaler()
data[['Age', 'CGPA', 'Sleep Duration', 'Work Pressure', 'Academic Pressure', 'Study Satisfaction', 'Job Satisfaction']] = scaler.fit_transform(
    data[['Age', 'CGPA', 'Sleep Duration', 'Work Pressure', 'Academic Pressure', 'Study Satisfaction', 'Job Satisfaction']]
)

# Chia dữ liệu
X = data.drop(['ID', 'Depression_Status'], axis=1)  # Bỏ cột ID và mục tiêu
y = data['Depression_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)