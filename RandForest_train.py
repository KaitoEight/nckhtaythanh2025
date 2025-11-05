import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Đọc dữ liệu và kiểm tra
try:
    data = pd.read_csv('Student_Depression_Dataset.csv')
    print("Dataset loaded successfully. First 5 rows:")
    print(data.head())
    print("\nColumns in dataset:", data.columns.tolist())
except FileNotFoundError:
    print("Error: File 'Student_Depression_Dataset.csv' not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 2. Kiểm tra và xử lý giá trị không hợp lệ
print("\nChecking for invalid values in numeric columns...")
numeric_columns = ['Age', 'CGPA', 'Sleep Duration', 'Work Pressure', 'Academic Pressure', 'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
for col in numeric_columns:
    if col in data.columns:
        # Chuyển đổi các giá trị không phải số về NaN
        data[col] = pd.to_numeric(data[col], errors='coerce')
        print(f"Column {col} - Invalid values replaced with NaN: {data[col].isnull().sum()}")

# Xử lý giá trị thiếu
for col in numeric_columns:
    if col in data.columns:
        data[col].fillna(data[col].mean(), inplace=True)

print("\nMissing values after handling:")
print(data.isnull().sum())

# 3. Mã hóa biến phân loại
try:
    label_encoder = LabelEncoder()
    categorical_columns = ['Gender', 'City', 'Profession', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness', 'Depression']
    for col in categorical_columns:
        if col in data.columns:
            data[col] = label_encoder.fit_transform(data[col])
    print("\nEncoded categorical columns successfully.")
except KeyError as e:
    print(f"Error: Column {e} not found in dataset. Please check column names.")
    exit()

# One-Hot Encoding cho các cột phân loại
try:
    one_hot_columns = ['City', 'Profession', 'Dietary Habits', 'Degree']
    data = pd.get_dummies(data, columns=[col for col in one_hot_columns if col in data.columns], drop_first=True)
    print("\nOne-Hot Encoding completed. New columns:", data.columns.tolist())
except KeyError as e:
    print(f"Error: Column {e} not found for One-Hot Encoding. Please check column names.")
    exit()

# 4. Chuẩn hóa dữ liệu
scaler = StandardScaler()
try:
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    print("\nData standardized successfully.")
except KeyError as e:
    print(f"Error: Column {e} not found for standardization. Please check column names.")
    exit()

# 5. Chia dữ liệu thành X (features) và y (target)
try:
    X = data.drop(['id', 'Depression'], axis=1)
    y = data['Depression']
    print("\nFeatures and target separated successfully.")
except KeyError as e:
    print(f"Error: Column {e} not found when splitting data. Please check column names.")
    exit()

# 6. Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData split into train and test sets. Shapes:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

# 7. Khởi tạo và huấn luyện mô hình
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\nModel trained successfully.")

# 8. Dự đoán và đánh giá
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("Classification Report:\n", classification_report_str)

# 9. Lưu mô hình, scaler và danh sách cột
try:
    joblib.dump(model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(X.columns, 'feature_columns.pkl')
    print("\nModel, scaler, and feature columns saved successfully as .pkl files.")
except Exception as e:
    print(f"Error saving .pkl files: {e}")
    exit()