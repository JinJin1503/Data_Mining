import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reconfigure stdout to support UTF-8 for Vietnamese characters in console output
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import GridSearchCV

# Import module của TV2
from preprocessing import get_preprocessed_data

def train_baseline(X_train, y_train):
    """
    Huấn luyện mô hình Decision Tree baseline với max_depth=5.
    """
    baseline_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    baseline_model.fit(X_train, y_train)
    return baseline_model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Đánh giá mô hình trên tập Test và in các chỉ số: Accuracy, Precision, Recall, F1.
    """
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"--- ĐÁNH GIÁ MÔ HÌNH: {model_name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))
    print("-" * 40)
    
    return y_pred, {
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1
    }

def find_best_depth(X_train, X_test, y_train, y_test):
    """
    Quét max_depth từ 1 đến 10 để theo dõi hiện tượng Overfitting.
    Trả về depths, train_scores, test_scores và best_depth (Phương án A).
    """
    depths = range(1, 11)
    train_scores = []
    test_scores = []
    
    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        
        train_pred = dt.predict(X_train)
        test_pred = dt.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        train_scores.append(train_acc)
        test_scores.append(test_acc)
        
    # Chọn best_depth theo Phương án A:
    # 1. Tìm Test Accuracy cao nhất.
    # 2. Nếu có nhiều chiều sâu đạt Test Accuracy cao nhất, chọn chiều sâu nhỏ hơn để cây đơn giản.
    max_test_acc = max(test_scores)
    best_depth = 1
    for depth, score in zip(depths, test_scores):
        if score == max_test_acc:
            best_depth = depth
            break # Lấy phần tử đầu tiên gặp (nhỏ nhất)
            
    return list(depths), train_scores, test_scores, best_depth

def plot_overfitting_curve(depths, train_scores, test_scores, best_depth):
    """
    Vẽ biểu đồ Overfitting Curve và lưu vào kết quả.
    """
    plt.figure(figsize=(9, 5))
    plt.plot(depths, train_scores, marker='o', color='#e056fd', linewidth=2, label='Train Accuracy')
    plt.plot(depths, test_scores, marker='s', color='#0984e3', linewidth=2, label='Test Accuracy')
    
    # Đánh dấu best_depth bằng đường nét đứt
    plt.axvline(x=best_depth, color='#2d3436', linestyle='--', alpha=0.7, 
                label=f'Best Depth ({best_depth})')
    
    plt.xlabel("Max Depth (Độ sâu tối đa)", fontsize=11, fontweight='bold', labelpad=8)
    plt.ylabel("Accuracy (Độ chính xác)", fontsize=11, fontweight='bold', labelpad=8)
    plt.title("Biểu đồ Phân tích Quá khớp (Overfitting Curve) - Decision Tree", 
              fontsize=12, fontweight='bold', pad=15)
    plt.xticks(depths)
    plt.legend(frameon=True, facecolor='white', edgecolor='lightgray')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    filepath = os.path.join("results", "figures", "depth_vs_accuracy.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ Overfitting tại: {filepath}")

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """
    Vẽ ma trận nhầm lẫn đẹp mắt bằng Seaborn Heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    
    # Sử dụng Seaborn Heatmap để biểu đồ hóa ma trận nhầm lẫn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"],
                cbar=False,
                annot_kws={"size": 12, "weight": "bold"})
    
    plt.title(title, fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Nhãn dự đoán (Predicted Label)", fontsize=10, labelpad=10)
    plt.ylabel("Nhãn thực tế (True Label)", fontsize=10, labelpad=10)
    plt.tight_layout()
    
    filepath = os.path.join("results", "figures", filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Đã lưu ma trận nhầm lẫn tại: {filepath}")

def plot_feature_importance(model, feature_names, filename):
    """
    Trích xuất và vẽ biểu đồ thanh ngang Top 10 đặc trưng quan trọng nhất.
    """
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True) # Sắp xếp tăng dần để vẽ barh từ dưới lên trên
    
    # Chỉ lấy Top 10 đặc trưng quan trọng nhất (hoặc tất cả nếu ít hơn 10)
    top_n = importance_df.tail(10)
    
    plt.figure(figsize=(9, 6))
    
    # Tạo danh sách màu chuyển sắc nhẹ từ xám đến xanh
    colors = sns.color_palette("GnBu_d", len(top_n))
    
    plt.barh(top_n["Feature"], top_n["Importance"], color=colors, height=0.6, edgecolor='none')
    plt.xlabel("Mức độ quan trọng (Gini Importance)", fontsize=10, fontweight='bold', labelpad=8)
    plt.ylabel("Đặc trưng (Features)", fontsize=10, fontweight='bold', labelpad=8)
    plt.title("Top 10 Đặc trưng Ảnh hưởng Nhất đến Bệnh tim mạch", fontsize=12, fontweight='bold', pad=15)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    filepath = os.path.join("results", "figures", filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ Feature Importance tại: {filepath}")

def plot_decision_tree(model, feature_names, filename):
    """
    Vẽ và lưu cấu trúc cây quyết định dưới dạng sơ đồ cây phân nhánh độ phân giải cao.
    """
    plt.figure(figsize=(22, 11), dpi=300)
    plot_tree(
        model,
        feature_names=list(feature_names),
        class_names=["No Disease", "Disease"],
        filled=True,
        rounded=True,
        fontsize=9
    )
    plt.title("Sơ đồ Cây Quyết định Phân lớp Bệnh Tim mạch (Tối ưu)", fontsize=15, fontweight='bold', pad=20)
    
    filepath = os.path.join("results", "figures", filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu sơ đồ cây quyết định tại: {filepath}")

def main():
    # Khởi tạo thư mục chứa kết quả hình ảnh
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    
    # ----------------------------------------------------
    # 1. Load Data
    # ----------------------------------------------------
    print("=== BƯỚC 1: LOAD DỮ LIỆU ĐÃ TIỀN XỬ LÝ ===")
    X_train, X_test, y_train, y_test = get_preprocessed_data()
    print(f"Kích thước tập Train: {X_train.shape}")
    print(f"Kích thước tập Test : {X_test.shape}\n")
    
    # ----------------------------------------------------
    # 2. Huấn luyện mô hình Baseline (max_depth=5)
    # ----------------------------------------------------
    print("=== BƯỚC 2: HUẤN LUYỆN MÔ HÌNH BASELINE ===")
    baseline_model = train_baseline(X_train, y_train)
    evaluate_model(baseline_model, X_test, y_test, model_name="Baseline Tree (max_depth=5)")
    
    # ----------------------------------------------------
    # 3. Quét max_depth (Phase 1: Depth Sweep)
    # ----------------------------------------------------
    print("=== BƯỚC 3: QUÉT CHIỀU SÂU CÂY (DEPTH SWEEP) ===")
    depths, train_scores, test_scores, best_depth = find_best_depth(X_train, X_test, y_train, y_test)
    print(f"Chiều sâu tối ưu tự động chọn (Phương án A): {best_depth}")
    for d, train_acc, test_acc in zip(depths, train_scores, test_scores):
        print(f"Depth: {d:2d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    print()
    
    # Vẽ và lưu biểu đồ overfitting
    plot_overfitting_curve(depths, train_scores, test_scores, best_depth)
    print()
    
    # ----------------------------------------------------
    # 4. Tối ưu hóa siêu tham số (Phase 2: GridSearchCV)
    # ----------------------------------------------------
    print("=== BƯỚC 4: TỐI ƯU SIÊU THAM SỐ BẰNG GRIDSEARCHCV ===")
    # Lưới tìm kiếm tham số vừa phải cho Decision Tree
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 10],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 5, 8, 10]
    }
    
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    print("Các siêu tham số tốt nhất:")
    print(grid_search.best_params_)
    print(f"Độ chính xác Cross-Validation tốt nhất: {grid_search.best_score_:.4f}\n")
    
    # ----------------------------------------------------
    # 5. Đánh giá mô hình tối ưu (GridSearchCV best_model)
    # ----------------------------------------------------
    y_pred_tuned, metrics = evaluate_model(best_model, X_test, y_test, model_name="Tuned Tree (GridSearchCV)")
    
    # ----------------------------------------------------
    # 6. Vẽ và Lưu các biểu đồ kết quả cho mô hình tối ưu
    # ----------------------------------------------------
    print("=== BƯỚC 5: XUẤT CÁC BIỂU ĐỒ BÁO CÁO CỦA MÔ HÌNH TỐI ƯU ===")
    
    # Vẽ Ma trận nhầm lẫn
    plot_confusion_matrix(
        y_test, 
        y_pred_tuned, 
        title="Ma trận Nhầm lẫn - Decision Tree (Tuned)", 
        filename="confusion_matrix_dt.png"
    )
    
    # Vẽ Feature Importance
    plot_feature_importance(
        best_model, 
        X_train.columns, 
        filename="feature_importance.png"
    )
    
    # Vẽ cấu trúc Cây quyết định
    plot_decision_tree(
        best_model, 
        X_train.columns, 
        filename="decision_tree.png"
    )
    
    print("\nQuá trình xử lý và lưu kết quả hoàn tất thành công!")

if __name__ == "__main__":
    main()
