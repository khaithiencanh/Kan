import os
import argparse
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

import classic_models
from dataload.dataload_five_flower import Five_Flowers_Load

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[*] Đang khởi động tiến trình kiểm thử trên: {device}")

    # 1. Transform ảnh
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Dữ liệu
    test_path = os.path.join(args.data_path, 'test')
    test_dataset = Five_Flowers_Load(test_path, transform=data_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                              pin_memory=True, num_workers=0, collate_fn=test_dataset.collate_fn)

    # 3. Khởi tạo mô hình & Load Weights
    model = classic_models.find_model_using_name(args.model, num_classes=args.num_classes).to(device)
    weights_path = os.path.join(os.getcwd(), 'results/weights', args.model, "kansformer_flower.pth")
    assert os.path.exists(weights_path), f"Không tìm thấy file trọng số tại: {weights_path}"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    model.eval()

    true_labels = []
    pred_labels = []
    pred_probs = []  # Cần thu thập thêm Xác suất để tính AUC

    print(f"[*] Bắt đầu chấm điểm {len(test_dataset)} ảnh...")
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Đang đánh giá Test Set", colour='green')
        for data in test_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            
            # Tính xác suất (softmax) cho AUC
            probs = torch.softmax(outputs, dim=1)
            # Lấy nhãn dự đoán cho Accuracy và F1
            pred = torch.max(outputs, dim=1)[1]

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(pred.cpu().numpy())
            pred_probs.append(probs.cpu().numpy())

    # Gộp list xác suất thành 1 ma trận numpy [Số lượng ảnh, Số class]
    pred_probs = np.concatenate(pred_probs, axis=0)

    # ==============================================================================
    # 6. TÍNH TOÁN 3 CHỈ SỐ CỐT LÕI (Nhân 100 để hiển thị %)
    # ==============================================================================
    
    # 1. Accuracy
    acc = accuracy_score(true_labels, pred_labels) * 100
    
    # 2. F1-Score (Macro)
    f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0) * 100
    
    # 3. AUC (Macro, One-vs-Rest) 
    # Lưu ý: Tính AUC đa lớp (15 classes) thường dùng chiến lược One-vs-Rest (ovr)
    try:
        auc_macro = roc_auc_score(true_labels, pred_probs, multi_class='ovr', average='macro') * 100
    except ValueError as e:
        print(f"\n[Cảnh báo] Không thể tính AUC: {e}")
        print("Lý do: Có thể tập Test bị thiếu hoàn toàn một loại tế bào nào đó.")
        auc_macro = 0.0

    # ==============================================================================
    # 7. IN BẢNG BÁO CÁO GỌN GÀNG THEO CHUẨN MỚI
    # ==============================================================================
    print("\n")
    print(f"{'TABLE: SCKansformer evaluation experiments on the AML dataset.':^75}")
    print("-" * 75)
    print(f"{'Method':<30} | {'Accuracy':<10} | {'F1 Macro':<10} | {'AUC':<10}")
    print("-" * 75)
    print(f"{'SCKansformer (Ours)':<30} | {acc:<10.2f} | {f1_macro:<10.2f} | {auc_macro:<10.2f}")
    print("-" * 75)

    # In chi tiết báo cáo F1 từng Class để rà soát
    print("\n[*] Chi tiết F1-Score từng Class:")
    print(classification_report(true_labels, pred_labels, zero_division=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_path', type=str, default="./data/AML_LMU")
    parser.add_argument('--model', type=str, default="kansformer1")
    parser.add_argument('--device', default='cuda')
    
    opt = parser.parse_args()
    main(opt)