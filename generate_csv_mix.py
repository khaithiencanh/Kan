import os
import pandas as pd

def create_reproduce_mix(base_csv, fake_dir, milestones, output_base_dir, num_splits=5):
    print(f"[*] Đang nạp danh bạ ảnh thật từ: {base_csv}")
    # Đọc file ảnh thật đã có sẵn 5 Fold chuẩn
    df_real = pd.read_csv(base_csv)

    for m in milestones:
        print(f"\n[*] Đang xử lý mốc: {m} ảnh giả/class (Xoay vòng Train/Val/Test)...")
        fake_data_rows = []
        
        for class_name in os.listdir(fake_dir):
            class_folder = os.path.join(fake_dir, class_name)
            if not os.path.isdir(class_folder): continue
                
            # Lấy toàn bộ ảnh giả và cắt đúng số lượng mốc
            all_imgs = [f for f in os.listdir(class_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
            selected_imgs = all_imgs[:m]
            total_imgs = len(selected_imgs)
            
            if total_imgs == 0:
                continue
                
            # Duyệt qua từng ảnh giả và tính toán chunk_id bằng toán học cơ bản
            for idx, img_file in enumerate(selected_imgs):
                # Chia đều thành 5 phần (0, 1, 2, 3, 4) một cách tự động
                chunk_id = (idx * num_splits) // total_imgs
                
                img_path = os.path.join(class_folder, img_file).replace("\\", "/")
                
                row_data = {
                    'image': img_path,
                    'label': class_name,
                    'dataset': 'synthetic'
                }
                
                # Áp dụng công thức xoay vòng y hệt file kfold của ảnh thật
                for i in range(num_splits):
                    fold_col = f'kfold{i}'
                    test_chunk = (i + num_splits - 1) % num_splits
                    val_chunk  = (i + num_splits - 2) % num_splits
                    
                    if chunk_id == test_chunk:
                        row_data[fold_col] = 'test'
                    elif chunk_id == val_chunk:
                        row_data[fold_col] = 'val'
                    else:
                        row_data[fold_col] = 'train'
                        
                fake_data_rows.append(row_data)

        # Gộp ảnh thật và ảnh giả đã xoay vòng
        df_fake = pd.DataFrame(fake_data_rows)
        df_mix = pd.concat([df_real, df_fake], ignore_index=True)
        
        # Tạo thư mục đích (Ví dụ: csv_files/mix/1000)
        out_dir = os.path.join(output_base_dir, str(m))
        os.makedirs(out_dir, exist_ok=True)
        
        # Lưu file CSV
        out_csv_path = os.path.join(out_dir, "AML_metadata.csv")
        df_mix.to_csv(out_csv_path, index=False)
        
        print(f"    -> Đã tạo xong: {out_csv_path}")
        print(f"    -> Ảnh thật: {len(df_real)} | Ảnh giả: {len(df_fake)} | Tổng: {len(df_mix)}")

if __name__ == "__main__":
    # 1. Đường dẫn file danh bạ ảnh thật
    BASE_CSV = "AML_metadata.csv" 
    
    # 2. Thư mục chứa 15 folder ảnh giả
    SYNTHETIC_DIRECTORY = "../data/results/synthetic2/matek/sd2.1/gs2.0_nis50/shot16_seed0_template1_lr0.0001_ep300/train" 
    
    # 3. Thư mục mẹ chứa các file mix
    OUTPUT_DIR = "csv_files/mix"
    
    # 4. Các mốc muốn tạo
    MILESTONES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    create_reproduce_mix(BASE_CSV, SYNTHETIC_DIRECTORY, MILESTONES, OUTPUT_DIR)