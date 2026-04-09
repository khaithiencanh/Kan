import os
import argparse
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

import classic_models
# 1. GỌI DATASET MỚI CHUẨN Y TẾ
from dataset_wbc import DatasetMarr, labels_map

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("\n" + "="*70)
    print(f"[*] ĐANG KHỞI ĐỘNG KIỂM THỬ TRÊN {str(device).upper()} CHO FOLD SỐ: {args.fold}")
    print("="*70 + "\n")

    # 2. Transform ảnh (Giữ nguyên như lúc Train/Val)
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Load Dữ liệu Test bằng DatasetMarr (Nhặt những ảnh được đánh chữ 'test' trong CSV)
    test_dataset = DatasetMarr(dataroot=args.data_path,
                               dataset_selection="AML_LMU",
                               labels_map=labels_map,
                               fold=args.fold,
                               transform=data_transform,
                               state='test')

    # Xóa collate_fn cũ đi để tránh báo lỗi
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                              pin_memory=True, num_workers=0)

    # 4. Khởi tạo mô hình & Load file .pth tương ứng với Fold đang chấm
    model = classic_models.find_model_using_name(args.model, num_classes=args.num_classes).to(device)
    
    # Đổi tên file để nó nạp đúng Fold (VD: kansformer_AML_fold0.pth)
    weights_name = f"kansformer_AML_fold{args.fold}.pth"
    weights_path = os.path.join(os.getcwd(), 'results/weights', args.model, weights_name)
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

    # ==============================================================================
    # 5. TÍNH TOÁN CÁC CHỈ SỐ (Accuracy, F1 Macro, AUC)
    # ==============================================================================
    acc = accuracy_score(true_labels, pred_labels) * 100
    f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0) * 100
    
    try:
        auc_macro = roc_auc_score(true_labels, pred_probs, multi_class='ovr', average='macro') * 100
    except ValueError as e:
        auc_macro = 0.0

    # ==============================================================================
    # 6. IN BÁO CÁO CỰC ĐẸP ĐỂ COPY VÀO BÁO CÁO
    # ==============================================================================
    print("\n")
    print(f"{f'TABLE: SCKansformer evaluation on the AML dataset (Fold {args.fold})':^75}")
    print("-" * 75)
    print(f"{'Method':<30} | {'Accuracy':<10} | {'F1 Macro':<10} | {'AUC':<10}")
    print("-" * 75)
    print(f"{'SCKansformer (Ours)':<30} | {acc:<10.2f} | {f1_macro:<10.2f} | {auc_macro:<10.2f}")
    print("-" * 75)

    print("\n[*] Chi tiết F1-Score từng Class:")
    
    # Hiển thị tên Tế bào thay vì số 0-14 cho dễ nhìn
    target_names = [name for name, idx in sorted(labels_map.items(), key=lambda x: x[1])]
    print(classification_report(true_labels, pred_labels, target_names=target_names, zero_division=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_path', type=str, default="./data/AML_LMU")
    parser.add_argument('--model', type=str, default="kansformer1")
    parser.add_argument('--device', default='cuda')
    
    # THÊM THAM SỐ CHỌN FOLD ĐỂ TEST
    parser.add_argument('--fold', type=int, default=0, help='Chấm điểm Fold số mấy?')
    
    opt = parser.parse_args()
    main(opt)