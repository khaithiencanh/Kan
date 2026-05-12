import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

def get_metric(splits, split_name, metric_key):
    """Lấy metric từ split, trả về None nếu không có"""
    if split_name not in splits:
        return None
    d = splits[split_name]
    # Tương thích cả format cũ lẫn mới (giống bảng summary)
    val = d.get('accuracy' if metric_key == 'acc' else
                'f1_macro'  if metric_key == 'f1'  else 'auc', None)
    return val if val and val != 0 else None

def load_and_aggregate_data(json_file='results_summary.json'):
    if not os.path.exists(json_file):
        print(f"[!] Không tìm thấy file {json_file}")
        return None

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    mix_keys = sorted(data.keys(), key=lambda x: int(x.replace('mix', '')))

    result = {
        'mix_sizes': [],
        'val_acc':  [], 'test_acc':  [],
        'val_f1':   [], 'test_f1':   [],
        'val_auc':  [], 'test_auc':  [],
    }

    for key in mix_keys:
        size = int(key.replace('mix', ''))
        result['mix_sizes'].append(size)

        val  = {'acc': [], 'f1': [], 'auc': []}
        test = {'acc': [], 'f1': [], 'auc': []}

        for fold_key, splits in data[key].items():
            for metric in ['acc', 'f1', 'auc']:
                v = get_metric(splits, 'Validation', metric)
                t = get_metric(splits, 'Test', metric)
                if v is not None: val[metric].append(v)
                if t is not None: test[metric].append(t)

        # Tính mean + std — giống hệt bảng summary
        for split_name, bucket in [('val', val), ('test', test)]:
            for metric in ['acc', 'f1', 'auc']:
                values = bucket[metric]
                result[f'{split_name}_{metric}'].append(
                    (np.mean(values), np.std(values)) if values else (0, 0)
                )

    return result

def plot_metric(data, metric_key, metric_label, filename):
    """Vẽ bar chart cho 1 metric, có error bar = std"""
    x = np.arange(len(data['mix_sizes']))
    width = 0.35

    val_means  = [v[0] for v in data[f'val_{metric_key}']]
    val_stds   = [v[1] for v in data[f'val_{metric_key}']]
    test_means = [v[0] for v in data[f'test_{metric_key}']]
    test_stds  = [v[1] for v in data[f'test_{metric_key}']]

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width/2, val_means, width, yerr=val_stds,
                   label='Validation', color='#5DBB63',
                   edgecolor='black', linewidth=0.8,
                   capsize=4, error_kw={'linewidth': 1.2})

    bars2 = ax.bar(x + width/2, test_means, width, yerr=test_stds,
                   label='Test', color='#1D6A96',
                   edgecolor='black', linewidth=0.8,
                   capsize=4, error_kw={'linewidth': 1.2})

    # Ghi mean lên đỉnh cột
    for bars, means in [(bars1, val_means), (bars2, test_means)]:
        for bar, mean in zip(bars, means):
            ax.annotate(f'{mean:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 6), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_ylabel(f'{metric_label} (%)', fontweight='bold', fontsize=12)
    ax.set_title(f'Validation vs Test — {metric_label} across Mix Sizes',
                 fontweight='bold', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Mix\n{s}' for s in data['mix_sizes']],
                       fontweight='bold', fontsize=9)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    all_vals = val_means + test_means
    ax.set_ylim([max(0, min(all_vals) - 8), min(100, max(all_vals) + 10)])

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[+] Đã lưu: {filename}")
    plt.close()

if __name__ == '__main__':
    data = load_and_aggregate_data()
    if data:
        plot_metric(data, 'acc', 'Accuracy',  'figure_accuracy.png')
        plot_metric(data, 'f1',  'F1-Macro',  'figure_f1.png')
        plot_metric(data, 'auc', 'AUC',       'figure_auc.png')