from src.preprocessing import get_preprocessed_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt

# Lấy dữ liệu đã được TV2 xử lý

X_train, X_test, y_train, y_test = get_preprocessed_data()

# Khởi tạo mô hình Logistic Regression

model = LogisticRegression(
max_iter=1000,
random_state=42
)

# Huấn luyện mô hình

model.fit(X_train, y_train)

# Dự đoán

y_pred = model.predict(X_test)

# Đánh giá mô hình

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("===== KẾT QUẢ LOGISTIC REGRESSION =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Phân tích trọng số

coef_df = pd.DataFrame({
"Feature": X_train.columns,
"Coefficient": model.coef_[0]
})

coef_df["AbsCoef"] = abs(coef_df["Coefficient"])

coef_df = coef_df.sort_values(
by="AbsCoef",
ascending=False
)

print("\n===== ĐỘ QUAN TRỌNG CỦA CÁC THUỘC TÍNH =====")
print(coef_df)

# Top 10 đặc trưng ảnh hưởng mạnh nhất

top10 = coef_df.head(10)

plt.figure(figsize=(10,6))
plt.barh(top10["Feature"], top10["Coefficient"])
plt.title("Top 10 Features - Logistic Regression")
plt.tight_layout()
plt.savefig("logistic_feature_importance.png")
plt.show()
