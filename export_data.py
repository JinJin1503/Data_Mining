"""
Script xuất dữ liệu sau khi tiền xử lý ra file CSV để kiểm tra trực quan.
Chạy lệnh: python export_data.py
"""
import os
import sys

# Thêm thư mục gốc của dự án vào đường dẫn để import được module src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import get_preprocessed_data
import pandas as pd

def export():
    X_train, X_test, y_train, y_test = get_preprocessed_data()

    # Tạo thư mục output nếu chưa có
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)

    # Ghép X và y lại thành bảng hoàn chỉnh để dễ xem
    train_full = pd.concat([X_train, y_train], axis=1)
    test_full = pd.concat([X_test, y_test], axis=1)

    # Xuất ra file CSV
    train_full.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
    test_full.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)

    print(f"Da xuat thanh cong vao thu muc: {output_dir}")
    print(f"  - train_processed.csv  ({train_full.shape[0]} dong x {train_full.shape[1]} cot)")
    print(f"  - test_processed.csv   ({test_full.shape[0]} dong x {test_full.shape[1]} cot)")
    print(f"\nCac cot du lieu: {list(train_full.columns)}")
    print(f"\n--- 5 dong dau cua tap Train ---")
    print(train_full.head().to_string())

if __name__ == '__main__':
    export()
