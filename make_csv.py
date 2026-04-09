import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def create_kfold_csv(data_dir, output_csv, num_splits=5):
    print(f"[*] Đang quét thư mục ảnh: {data_dir}...")
    
    data = []
    
    # 1. Quét toàn bộ ảnh trong các thư mục con
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for label in classes:
        folder_path = os.path.join(data_dir, label)
        images = [img for img in os.listdir(folder_path) if img.endswith(('.tiff', '.jpg', '.png', '.jpeg'))]
        
        for img_name in images:
            # Lưu đường dẫn tương đối (hoặc tuyệt đối tùy ý bạn)
            # Ở đây dùng đường dẫn tương đối cho dễ bê sang máy khác
            img_path = os.path.join(data_dir, label, img_name).replace("\\", "/")
            data.append([img_path, label, 'AML_LMU'])

    # Chuyển thành DataFrame
    df = pd.DataFrame(data, columns=['image', 'label', 'dataset'])
    print(f"[*] Đã tìm thấy tổng cộng {len(df)} ảnh thuộc {len(classes)} loại tế bào.")

    # 2. Chia 5 Chunk cân bằng (Stratified) theo đúng tỷ lệ tế bào
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    df['chunk_id'] = -1
    
    for chunk_idx, (train_index, test_index) in enumerate(skf.split(df, df['label'])):
        df.loc[test_index, 'chunk_id'] = chunk_idx

    # 3. Phân bổ Train (60%), Val (20%), Test (20%) xoay vòng cho 5 Fold
    for i in range(num_splits):
        fold_col = f'kfold{i}'
        df[fold_col] = 'train'  # Mặc định là train
        
        # Công thức xoay vòng chuẩn của file kfold_splits.py
        test_chunk = (i + num_splits - 1) % num_splits
        val_chunk  = (i + num_splits - 2) % num_splits

        df.loc[df['chunk_id'] == val_chunk, fold_col] = 'val'
        df.loc[df['chunk_id'] == test_chunk, fold_col] = 'test'

    # Xóa cột tạm
    df.drop(columns=['chunk_id'], inplace=True)

    # 4. Lưu ra file CSV
    df.to_csv(output_csv, index=False)
    print(f"[+] Đã tạo file danh bạ thành công: {output_csv}")

    # In thử thống kê của Fold 0 xem chuẩn 60-20-20 chưa
    print("\nThống kê phân bổ dữ liệu tại Fold 0:")
    print(df['kfold0'].value_counts())

if __name__ == "__main__":
    # Thay đường dẫn thư mục gốc chứa 15 thư mục tế bào của bạn vào đây
    DATA_DIRECTORY = "./data/AML_LMU" 
    OUTPUT_FILE = "AML_metadata.csv"
    
    create_kfold_csv(DATA_DIRECTORY, OUTPUT_FILE)