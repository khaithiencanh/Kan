import json
import os
import numpy as np

json_file_path = 'results_summary.json'

if not os.path.exists(json_file_path):
    print("Chưa có file kết quả. Hãy chạy test.py trước!")
else:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"{'BẢNG TỔNG HỢP ĐIỂM TRUNG BÌNH CÁC MỐC DỮ LIỆU':^60}")
    print(f"{'='*60}")
    print(f"{'Mốc Data':<15} | {'Accuracy TB':<12} | {'F1-Macro TB':<12} | {'AUC TB':<12}")
    print("-" * 60)

    # Duyệt qua từng mốc (mix100, mix300...)
    for mix_key, folds_data in data.items():
        acc_list = []
        f1_list = []
        auc_list = []
        
        # Gom điểm của tất cả các fold đã chạy trong mốc đó
        for fold_key, metrics in folds_data.items():
            acc_list.append(metrics['accuracy'])
            f1_list.append(metrics['f1_macro'])
            auc_list.append(metrics['auc'])
            
        # Tính trung bình
        avg_acc = np.mean(acc_list)
        avg_f1 = np.mean(f1_list)
        avg_auc = np.mean(auc_list)
        
        # In ra số lượng Fold đã chạy để kiểm soát
        num_folds = len(acc_list)
        
        print(f"{mix_key} ({num_folds} folds) | {avg_acc:<12.2f} | {avg_f1:<12.2f} | {avg_auc:<12.2f}")
    
    print("=" * 60)