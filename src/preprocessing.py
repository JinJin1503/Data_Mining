import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path):
    """
    Đọc dữ liệu từ file CSV và xử lý các giá trị khuyết thiếu ẩn.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {file_path}")
        
    # Sử dụng utf-8-sig để tự động loại bỏ ký tự BOM (\ufeff) ở đầu file nếu có
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # Bản sao dữ liệu để tránh chỉnh sửa trên dữ liệu gốc
    df_clean = df.copy()
    
    # 1. Xử lý giá trị khuyết ở cột 'ca' (Number of major vessels)
    # Trong tập dữ liệu Cleveland gốc, giá trị '4' là ký hiệu của dữ liệu bị khuyết.
    # Ta thay thế giá trị khuyết '4' bằng giá trị yếu vị (mode) hoặc trung vị (median) của cột.
    ca_mode = df_clean[df_clean['ca'] != 4]['ca'].mode()[0]
    df_clean['ca'] = df_clean['ca'].replace(4, ca_mode)
    
    # 2. Xử lý giá trị khuyết ở cột 'thal' (Thalassemia)
    # Giá trị '0' đại diện cho dữ liệu bị khuyết.
    # Ta thay thế giá trị khuyết '0' bằng giá trị yếu vị (mode) của cột.
    thal_mode = df_clean[df_clean['thal'] != 0]['thal'].mode()[0]
    df_clean['thal'] = df_clean['thal'].replace(0, thal_mode)
    
    return df_clean

def preprocess_pipeline(data_path='data/heart.csv', test_size=0.2, random_state=42):
    """
    Quy trình tiền xử lý dữ liệu hoàn chỉnh:
    1. Đọc và làm sạch dữ liệu khuyết.
    2. Mã hóa One-Hot Encoding cho các biến phân loại đa trị.
    3. Chia tập Train/Test theo phương pháp phân tầng (Stratified Split).
    4. Chuẩn hóa (Scaling) các biến số liên tục bằng StandardScaler sau khi chia tập.
    """
    # Bước 1: Load và Clean dữ liệu
    df = load_and_clean_data(data_path)
    
    # Tách biến độc lập (X) và biến mục tiêu (y)
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Xác định các loại cột để áp dụng xử lý khác nhau
    # - Cột số liên tục cần chuẩn hóa (Scaling)
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    # - Cột phân loại đa trị cần mã hóa One-Hot Encoding
    cat_cols = ['cp', 'restecg', 'slope', 'thal']
    # - Cột phân loại nhị phân (0/1) giữ nguyên
    bin_cols = ['sex', 'fbs', 'exang', 'ca'] # ca đã là số mạch máu (0, 1, 2, 3), giữ nguyên tính chất đếm
    
    # Bước 2: One-Hot Encoding cho các cột phân loại đa trị
    # Sử dụng drop_first=True để tránh bẫy đa cộng tuyến (Dummy Variable Trap), 
    # rất quan trọng cho mô hình Logistic Regression của TV4.
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
    
    # Bước 3: Chia tập dữ liệu Train/Test
    # Sử dụng stratify=y để đảm bảo tỷ lệ nhãn target cân bằng giữa cả 2 tập.
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # Bước 4: Chuẩn hóa dữ liệu (Scaling)
    # Chỉ fit StandardScaler trên tập Train để tránh rò rỉ dữ liệu (Data Leakage).
    scaler = StandardScaler()
    
    # Tạo bản sao để tránh cảnh báo SettingWithCopyWarning
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Áp dụng scaling lên các cột số liên tục
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def get_preprocessed_data():
    """
    Hàm tiện ích để TV3 và TV4 gọi trực tiếp từ các file code khác.
    """
    # Lấy đường dẫn tuyệt đối đến file dữ liệu để chạy ổn định ở mọi thư mục làm việc
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    data_path = os.path.join(project_dir, 'data', 'heart.csv')
    
    return preprocess_pipeline(data_path=data_path)

if __name__ == '__main__':
    # Kiểm tra thử hoạt động của module
    try:
        X_train, X_test, y_train, y_test = get_preprocessed_data()
        print("--- KIỂM TRA PIPELINE TIỀN XỬ LÝ DỮ LIỆU ---")
        print(f"Kích thước tập Train: {X_train.shape}")
        print(f"Kích thước tập Test : {X_test.shape}")
        print("\nCác cột dữ liệu sau mã hóa:")
        print(list(X_train.columns))
        print("\nKiểm tra giá trị trung bình sau chuẩn hóa trên tập Train (xấp xỉ 0):")
        print(X_train[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].mean().round(4))
        print("\nKiểm tra tỷ lệ phân tầng target (Train vs Test):")
        print(f"Train target=1: {y_train.mean():.4f}")
        print(f"Test target=1: {y_test.mean():.4f}")
        print("\nPipeline hoạt động thành công!")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi chạy thử: {e}")
