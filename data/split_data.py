import os
import random
import shutil

def split_dataset_3_way(data_dir, train_ratio=0.7, val_ratio=0.15):
    # 1. Tạo 3 thư mục: train, val, test
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 2. Quét tất cả các thư mục con (các loại tế bào)
    classes = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d)) 
               and d not in ['train', 'val', 'test']]

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        images = os.listdir(cls_path)
        random.shuffle(images) # Xáo trộn ngẫu nhiên để đảm bảo khách quan

        total_images = len(images)
        
        # 3. Tính toán 2 mốc cắt
        train_idx = int(total_images * train_ratio)               # Mốc cắt Train (70%)
        val_idx = train_idx + int(total_images * val_ratio)       # Mốc cắt Val (15%)

        # 4. Phân chia mảng ảnh thành 3 phần
        train_images = images[:train_idx]
        val_images = images[train_idx:val_idx]
        test_images = images[val_idx:]

        # 5. Tạo thư mục con tương ứng trong train, val, test
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        # 6. Chuyển ảnh về đúng nhà của nó
        for img in train_images:
            shutil.move(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
        for img in val_images:
            shutil.move(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))
        for img in test_images:
            shutil.move(os.path.join(cls_path, img), os.path.join(test_dir, cls, img))
        
        # 7. Xóa thư mục gốc sau khi chuyển xong
        os.rmdir(cls_path)
        print(f"Lớp {cls}: {len(train_images)} Train | {len(val_images)} Val | {len(test_images)} Test")

# GỌI HÀM THỰC THI (Tự động chia 70 - 15 - 15)
split_dataset_3_way("./AML_LMU", train_ratio=0.7, val_ratio=0.15)