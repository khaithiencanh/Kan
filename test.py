import os
import argparse
import numpy as np
import torch
import json
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

import classic_models
from dataset_wbc import DatasetMarr, labels_map

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("\n" + "="*70)
    print(f"[*] ĐANG KHỞI ĐỘNG KIỂM THỬ TRÊN {str(device).upper()} CHO FOLD SỐ: {args.fold}")
    print("="*70 + "\n")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = DatasetMarr(dataroot=args.data_path,
                               dataset_selection="AML_LMU",
                               labels_map=labels_map,
                               fold=args.fold,
                               transform=data_transform,
                               state='test')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                              pin_memory=True, num_workers=0)

    model = classic_models.find_model_using_name(args.model, num_classes=args.num_classes).to(device)
    

    mix_val = os.path.basename(os.path.normpath(args.data_path))
    weights_name = f"kansformer_AML_mix{mix_val}_fold{args.fold}_best.pth"
    weights_path = os.path.join(os.getcwd(), 'weights', weights_name)
    assert os.path.exists(weights_path), f"Không tìm thấy file trọng số tại: {weights_path}"
    
    print(f"[*] Đang nạp: {weights_name}...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    true_labels = []
    pred_labels = []
    pred_probs = []

    print(f"[*] Bắt đầu chấm điểm {len(test_dataset)} ảnh...")
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc=f"Đánh giá Test Set Fold {args.fold}", colour='green')
        for data in test_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)
            pred = torch.max(outputs, dim=1)[1]

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(pred.cpu().numpy())
            pred_probs.append(probs.cpu().numpy())

    pred_probs = np.concatenate(pred_probs, axis=0)

    acc = accuracy_score(true_labels, pred_labels) * 100
    f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0) * 100
    
    try:
        auc_macro = roc_auc_score(true_labels, pred_probs, multi_class='ovr', average='macro') * 100
    except ValueError as e:
        auc_macro = 0.0

    print("\n")
    print(f"{f'TABLE: SCKansformer evaluation on the AML dataset (Fold {args.fold})':^75}")
    print("-" * 75)
    print(f"{'Method':<30} | {'Accuracy':<10} | {'F1 Macro':<10} | {'AUC':<10}")
    print("-" * 75)
    print(f"{'SCKansformer (Ours)':<30} | {acc:<10.2f} | {f1_macro:<10.2f} | {auc_macro:<10.2f}")
    print("-" * 75)

    print("\n[*] Chi tiết F1-Score từng Class:")
    
    target_names = [name for name, idx in sorted(labels_map.items(), key=lambda x: x[1])]
    print(classification_report(true_labels, pred_labels, target_names=target_names, zero_division=0))
    json_file_path = os.path.join(os.getcwd(), 'results_summary.json')

    # Đọc dữ liệu cũ nếu đã có file
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    else:
        results_data = {}

    # Tạo key theo mốc Mix và Fold hiện tại (VD: "mix100", "fold0")
    mix_key = f"mix{mix_val}"
    fold_key = f"fold{args.fold}"

    # Đảm bảo cấu trúc key tồn tại
    if mix_key not in results_data:
        results_data[mix_key] = {}

    # Ghi đè hoặc thêm mới kết quả của Fold hiện tại
    results_data[mix_key][fold_key] = {
        "accuracy": round(acc, 2),
        "f1_macro": round(f1_macro, 2),
        "auc": round(auc_macro, 2)
    }

    # Lưu lại vào file JSON
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    print(f"\n[+] Đã cập nhật kết quả tự động vào file: {json_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_path', type=str, default="csv_files/mix/100")
    parser.add_argument('--model', type=str, default="kansformer1")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fold', type=int, default=0, help='Chấm điểm Fold số mấy?')
    
    opt = parser.parse_args()
    main(opt)