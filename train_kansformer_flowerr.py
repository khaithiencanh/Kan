import os
import argparse
import math
import shutil
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

import classic_models
from utils.lr_methods import warmup
from utils.train_engin import train_one_epoch, evaluate
import datetime
# ==========================================
# NHÚNG DATASET MỚI CỦA BẠN VÀO ĐÂY
from dataset_wbc import DatasetMarr, labels_map
# ==========================================

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    if args.tensorboard:
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()

    save_path = './weights'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Transform bóp ảnh về chuẩn Kansformer
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    dataset_name = 'AML_LMU' 
    
    train_dataset = DatasetMarr(dataroot=args.data_path, dataset_selection=dataset_name, labels_map=labels_map, fold=args.fold, transform=data_transform["train"], state='train')
    val_dataset = DatasetMarr(dataroot=args.data_path, dataset_selection=dataset_name, labels_map=labels_map, fold=args.fold, transform=data_transform["val"], state='validation')

    nw = 4
    print(f'Using {nw} dataloader workers every process')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=nw)

    # Định nghĩa Model của tác giả
    model = getattr(classic_models, args.model)(num_classes=args.num_classes)
    model.to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1E-4)

    # Scheduler của tác giả
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    best_val_acc = 0.0
    mix_val = os.path.basename(os.path.normpath(args.data_path))
    best_weight_path = os.path.join(save_path, f"kansformer_AML_mix{mix_val}_fold{args.fold}_{current_time}_best.pth")
    for epoch in range(args.epochs):
        mean_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                            device=device, epoch=epoch, use_amp=args.use_amp, lr_method=warmup)
        scheduler.step()
        
        val_acc = evaluate(model=model, data_loader=val_loader, device=device)

        print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' % (
            epoch + 1, mean_loss, train_acc, val_acc))
        print(f"  VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

        # Dọn VRAM cache sau mỗi epoch
        torch.cuda.empty_cache()

        # Cập nhật log file
        log_file = os.path.join(save_path, f"kansformer_AML_fold{args.fold}_log.txt")
        with open(log_file, 'a') as f:
            f.writelines('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' % (
                epoch + 1, mean_loss, train_acc, val_acc) + '\n')

        if args.tensorboard:
            tags = ["train_loss", "train_acc", "val_accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_acc, epoch)
            tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weight_path)
            print(f"[+] Đã lưu Best Model tại epoch {epoch+1} (Val Acc: {val_acc:.3f})")

    print("="*50)
    print(f"[*] HOÀN THÀNH HUẤN LUYỆN! Model tốt nhất được lưu tại: {best_weight_path}")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=15, help='the number of classes')
    parser.add_argument('--epochs', type=int, default=50, help='the number of training epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size for training')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='learning rate factor')
    parser.add_argument('--data_path', type=str, default="csv_files/mix/100", help='dataset path')
    parser.add_argument('--model', type=str, default="kansformer1", help='model name')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--use_amp', action='store_true', help='Use torch.cuda.amp for mixed precision training')
    parser.add_argument('--fold', type=int, default=0, help='Fold number for cross validation')
    parser.add_argument('--tensorboard', default=False, action='store_true', help='Use tensorboard for logging')

    args = parser.parse_args()
    main(args)