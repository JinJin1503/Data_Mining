# BÁO CÁO THỰC NGHIỆM THUẬT TOÁN CÂY QUYẾT ĐỊNH (DECISION TREE)

Dự án: Dự đoán khả năng mắc bệnh tim mạch bằng các thuật toán phân lớp

| Thông tin | Giá trị |
|-----------|----------|
| Người thực hiện | Thành viên 3 (TV3) |
| Người review | Thành viên 5 (TV5) và các thành viên trong nhóm |
| Ngày hoàn thành | 23/06/2026 |

---

## 1. Tổng quan công việc

Thành viên 3 chịu trách nhiệm triển khai và đánh giá mô hình Cây quyết định (Decision Tree) trên bộ dữ liệu Heart Disease UCI đã được tiền xử lý bởi Thành viên 2. Công việc bao gồm:

- Xây dựng mô hình Decision Tree bằng thư viện Scikit-learn.
- Khảo sát ảnh hưởng của tham số `max_depth`.
- Phân tích hiện tượng Overfitting và lựa chọn độ sâu phù hợp.
- Tối ưu mô hình bằng GridSearchCV.
- Đánh giá mô hình bằng các độ đo Accuracy, Precision, Recall và F1-score.
- Trích xuất độ quan trọng của các thuộc tính (Feature Importance).
- Sinh các biểu đồ phục vụ cho báo cáo và thuyết trình.

---

## 2. Danh sách kết quả đầu ra

### 2.1. Mã nguồn

| File | Mô tả |
|--------|--------|
| `src/decision_tree.py` | Chương trình huấn luyện và đánh giá mô hình Decision Tree |

### 2.2. File kết quả

| File | Mô tả |
|--------|--------|
| `results/decision_tree_output.txt` | Kết quả chạy chương trình và các chỉ số đánh giá |

### 2.3. Biểu đồ

| File | Nội dung |
|--------|--------|
| `depth_vs_accuracy.png` | Biểu đồ khảo sát độ sâu cây và hiện tượng Overfitting |
| `confusion_matrix_dt.png` | Ma trận nhầm lẫn của mô hình tối ưu |
| `feature_importance.png` | Biểu đồ độ quan trọng của các thuộc tính |
| `decision_tree.png` | Hình ảnh cấu trúc cây quyết định |

Đường dẫn lưu trữ:

```text
results/figures/