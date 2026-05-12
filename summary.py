import json
import os
import numpy as np

json_file_path = 'results_summary.json'

if not os.path.exists(json_file_path):
    print("[!] Chưa có file kết quả. Hãy chạy test_kansformer.py trước!")
else:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    COL1, COL2, COL3 = 22, 12, 16
    TOTAL = COL1 + COL2 + COL3 * 3 + 10  # 10 = số ký tự " | " * 4

    print(f"\n{'='*TOTAL}")
    print(f"{'BẢNG TỔNG HỢP KẾT QUẢ KANSFORMER (MEAN ± STD)':^{TOTAL}}")
    print(f"{'='*TOTAL}")
    print(f"{'Dataset (Mix)':<{COL1}} | {'Split':<{COL2}} | {'Accuracy':<{COL3}} | {'F1 Macro':<{COL3}} | {'AUC':<{COL3}}")
    print("-" * TOTAL)

    try:
        sorted_mix_keys = sorted(data.keys(), key=lambda x: int(x.replace('mix', '')))
    except ValueError:
        sorted_mix_keys = sorted(data.keys())

    def format_metric(metric_list):
        valid_metrics = [m for m in metric_list if m != 0]
        if not valid_metrics:
            return "N/A"
        mean = np.mean(valid_metrics)
        std = np.std(valid_metrics)
        return f"{(mean/100):.4f} ± {(std/100):.4f}"

    for mix_key in sorted_mix_keys:
        folds_data = data[mix_key]

        val  = {'acc': [], 'f1': [], 'auc': []}
        test = {'acc': [], 'f1': [], 'auc': []}

        num_folds = len(folds_data)

        for fold_key, splits in folds_data.items():
            if "Validation" in splits:
                v = splits["Validation"]
                val['acc'].append(v.get('accuracy', v.get('acc', 0)))
                val['f1'].append(v.get('f1_macro', v.get('f1', 0)))
                val['auc'].append(v.get('auc', 0))

            if "Test" in splits:
                t = splits["Test"]
                test['acc'].append(t.get('accuracy', t.get('acc', 0)))
                test['f1'].append(t.get('f1_macro', t.get('f1', 0)))
                test['auc'].append(t.get('auc', 0))

        mix_label = f"{mix_key} ({num_folds} folds)"

        print(f"{mix_label:<{COL1}} | {'Validation':<{COL2}} | {format_metric(val['acc']):<{COL3}} | {format_metric(val['f1']):<{COL3}} | {format_metric(val['auc']):<{COL3}}")
        print(f"{'':<{COL1}} | {'Test':<{COL2}} | {format_metric(test['acc']):<{COL3}} | {format_metric(test['f1']):<{COL3}} | {format_metric(test['auc']):<{COL3}}")
        print("-" * TOTAL)