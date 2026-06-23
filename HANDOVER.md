# TÀI LIỆU BÀN GIAO - PHẦN TIỀN XỬ LÝ DỮ LIỆU (TV2)

> **Người bàn giao**: Thành viên 2 (TV2)  
> **Người nhận**: TV3 (Decision Tree), TV4 (Logistic Regression), TV5 (Review & Git)  
> **Ngày bàn giao**: 16/06/2026

---

## 1. Tổng quan: Tôi đã làm gì?

Tôi đã xây dựng hoàn chỉnh **module tiền xử lý dữ liệu** trong file `src/preprocessing.py`. Module này thực hiện 4 bước xử lý tuần tự trên bộ dữ liệu Heart Disease UCI gốc (`data/heart.csv`) và trả về dữ liệu sạch, sẵn sàng để các bạn đưa vào huấn luyện mô hình.

### Sơ đồ quy trình xử lý:

```
data/heart.csv (303 dòng, 14 cột gốc)
        │
        ▼
[Bước 1] Làm sạch dữ liệu khuyết ẩn
        │   • ca = 4  → thay bằng 0 (mode) — 5 dòng bị lỗi
        │   • thal = 0 → thay bằng 2 (mode) — 2 dòng bị lỗi
        ▼
[Bước 2] One-Hot Encoding (drop_first=True)
        │   • cp (4 giá trị)     → cp_1, cp_2, cp_3
        │   • restecg (3 giá trị) → restecg_1, restecg_2
        │   • slope (3 giá trị)   → slope_1, slope_2
        │   • thal (3 giá trị)    → thal_2, thal_3
        │   Tổng cột: 14 gốc → 18 cột đặc trưng + 1 cột target
        ▼
[Bước 3] Chia Train/Test (80/20, stratify theo target)
        │   • Train: 242 dòng
        │   • Test:  61 dòng
        ▼
[Bước 4] Chuẩn hóa StandardScaler (chỉ fit trên Train)
        │   Chỉ chuẩn hóa 5 cột số: age, trestbps, chol, thalach, oldpeak
        │   Các cột nhị phân (0/1) giữ nguyên
        ▼
Đầu ra: X_train, X_test, y_train, y_test
```

---

## 2. Cách sử dụng (QUAN TRỌNG - Đọc kỹ)

### 2.1. Cài đặt môi trường lần đầu
Sau khi clone repo về máy, mở terminal tại thư mục gốc dự án và chạy:

```powershell
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Cài thư viện
pip install -r requirements.txt
```

### 2.2. Lấy dữ liệu đã xử lý trong code của bạn
Các bạn **chỉ cần 2 dòng code** để nhận dữ liệu sạch:

```python
from src.preprocessing import get_preprocessed_data

X_train, X_test, y_train, y_test = get_preprocessed_data()
```

**Giải thích các biến trả về:**

| Biến | Kiểu dữ liệu | Kích thước | Mô tả |
|---|---|---|---|
| `X_train` | pandas DataFrame | (242, 18) | Dữ liệu đặc trưng tập huấn luyện (đã mã hóa + chuẩn hóa) |
| `X_test` | pandas DataFrame | (61, 18) | Dữ liệu đặc trưng tập kiểm thử |
| `y_train` | pandas Series | (242,) | Nhãn mục tiêu tập huấn luyện (0 hoặc 1) |
| `y_test` | pandas Series | (61,) | Nhãn mục tiêu tập kiểm thử (0 hoặc 1) |

### 2.3. Ví dụ code mẫu cho TV3 (Decision Tree)

```python
from src.preprocessing import get_preprocessed_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Lấy dữ liệu
X_train, X_test, y_train, y_test = get_preprocessed_data()

# Huấn luyện mô hình
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = dt.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

### 2.4. Ví dụ code mẫu cho TV4 (Logistic Regression)

```python
from src.preprocessing import get_preprocessed_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Lấy dữ liệu
X_train, X_test, y_train, y_test = get_preprocessed_data()

# Huấn luyện mô hình
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = lr.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Trích xuất trọng số để phân tích chỉ số y tế nào ảnh hưởng mạnh nhất
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr.coef_[0]
}).sort_values('Coefficient', ascending=False)
print(coef_df)
```

---

## 3. Danh sách 18 cột đặc trưng sau xử lý

| STT | Tên cột | Loại | Giải thích |
|---|---|---|---|
| 1 | `age` | Số (đã chuẩn hóa) | Tuổi bệnh nhân |
| 2 | `sex` | Nhị phân (0/1) | 1 = Nam, 0 = Nữ |
| 3 | `trestbps` | Số (đã chuẩn hóa) | Huyết áp lúc nghỉ (mm Hg) |
| 4 | `chol` | Số (đã chuẩn hóa) | Cholesterol huyết thanh (mg/dl) |
| 5 | `fbs` | Nhị phân (0/1) | Đường huyết đói > 120 mg/dl |
| 6 | `thalach` | Số (đã chuẩn hóa) | Nhịp tim tối đa đạt được |
| 7 | `exang` | Nhị phân (0/1) | Đau thắt ngực khi gắng sức |
| 8 | `oldpeak` | Số (đã chuẩn hóa) | Độ trầm cảm đoạn ST |
| 9 | `ca` | Số nguyên (0-3) | Số mạch máu lớn bị chặn |
| 10 | `cp_1` | One-Hot (0/1) | Đau ngực không điển hình |
| 11 | `cp_2` | One-Hot (0/1) | Đau không do thắt ngực |
| 12 | `cp_3` | One-Hot (0/1) | Không triệu chứng |
| 13 | `restecg_1` | One-Hot (0/1) | ECG: Bất thường sóng ST-T |
| 14 | `restecg_2` | One-Hot (0/1) | ECG: Phì đại thất trái |
| 15 | `slope_1` | One-Hot (0/1) | Đoạn ST đi ngang (Flat) |
| 16 | `slope_2` | One-Hot (0/1) | Đoạn ST dốc xuống |
| 17 | `thal_2` | One-Hot (0/1) | Thalassemia: Khuyết tật cố định |
| 18 | `thal_3` | One-Hot (0/1) | Thalassemia: Khuyết tật phục hồi |

**Cột mục tiêu (target)**: `1` = Mắc bệnh tim, `0` = Không mắc bệnh tim.

---

## 4. Lưu ý quan trọng cho TV5 (Review code)

### 4.1. Về Data Leakage
Code tiền xử lý đã được thiết kế để **tránh rò rỉ dữ liệu**:
- StandardScaler chỉ gọi `fit_transform()` trên `X_train` (dòng 80 trong `preprocessing.py`).
- Tập Test chỉ được `transform()` bằng tham số đã học từ tập Train (dòng 81).
- Việc One-Hot Encoding được thực hiện **trước** khi chia Train/Test, nhưng điều này không gây data leakage vì One-Hot chỉ chuyển đổi cấu trúc cột chứ không học thông tin thống kê nào từ dữ liệu.

### 4.2. Tham số cố định
- `random_state=42`: Đảm bảo mọi thành viên chạy code đều nhận được cùng một kết quả phân chia Train/Test.
- `test_size=0.2`: Tỷ lệ 80% Train / 20% Test.
- `stratify=y`: Tỷ lệ nhãn target cân bằng giữa Train (~54.55%) và Test (~54.10%).

### 4.3. Cấu trúc file cần quan tâm

```
heart-disease-prediction/
├── data/
│   ├── heart.csv                 ← Dữ liệu gốc (KHÔNG CHỈNH SỬA)
│   └── processed/
│       ├── train_processed.csv   ← File CSV tập Train (để xem bằng Excel)
│       └── test_processed.csv    ← File CSV tập Test (để xem bằng Excel)
├── src/
│   ├── __init__.py               ← Đánh dấu src là package Python
│   ├── preprocessing.py          ← MODULE CHÍNH (import từ đây)
│   ├── decision_tree.py          ← TV3 code ở đây
│   └── logistic_regression.py    ← TV4 code ở đây
├── export_data.py                ← Script xuất CSV (chạy: python export_data.py)
└── requirements.txt              ← Danh sách thư viện cần cài
```

---

## 5. Dữ liệu đã xuất sẵn (nếu không muốn chạy code)

Nếu các bạn chỉ muốn xem dữ liệu đã xử lý mà không cần chạy code Python, tôi đã xuất sẵn 2 file CSV:
- `data/processed/train_processed.csv` — 242 dòng × 19 cột
- `data/processed/test_processed.csv` — 61 dòng × 19 cột

Các bạn có thể mở bằng Excel để xem trực quan.

---

## 6. Liên hệ

Nếu TV3, TV4 hoặc TV5 gặp lỗi khi import hoặc chạy code, hãy kiểm tra:
1. Đã kích hoạt môi trường ảo `venv` chưa?
2. Đã cài đặt thư viện bằng `pip install -r requirements.txt` chưa?
3. Đang chạy lệnh Python từ **thư mục gốc dự án** (nơi chứa thư mục `src/`) chứ không phải từ bên trong thư mục `src/`?

Nếu vẫn lỗi, liên hệ trực tiếp TV2 qua nhóm chat.
