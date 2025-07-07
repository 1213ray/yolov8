# benchmark_dataloader.py
import time
import argparse
from torch.utils.data import DataLoader
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

def main(args):
    # 检查数据集
    data = check_det_dataset(args.data)
    # 使用 ultralytics 内部函数创建数据集
    train_ds = build_yolo_dataset(args, data['train'], args.batch_size, data, mode='train')

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,  # Only use persistent workers if num_workers > 0
        collate_fn=getattr(train_ds, 'collate_fn', None)
    )
    
    print(f"Starting benchmark with batch_size={args.batch_size}, num_workers={args.num_workers}...")
    
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= args.batches:
            break
    t1 = time.time()
    
    duration = t1 - t0
    batches_per_sec = args.batches / duration if duration > 0 else 0
    
    print(f'{args.batches} batches in {duration:.2f}s, ~{batches_per_sec:.2f} batches/s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark YOLO Dataloader")
    parser.add_argument('--data', type=str, default='yaml/person.yaml', help='path to data.yaml')
    parser.add_argument('--batch-size', type=int, default=12, help='batch size')
    parser.add_argument('--num-workers', type=int, default=12, help='number of workers')
    parser.add_argument('--batches', type=int, default=200, help='number of batches to run')
    
    # Add all necessary arguments found from trial and error, with defaults matching the user's training script
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--cache', type=bool, default=False)
    parser.add_argument('--rect', type=bool, default=False)
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--single-cls', action='store_true')
    parser.add_argument('--task', type=str, default='detect')
    parser.add_argument('--classes', nargs='+', type=int, default=None)
    
    # Augmentation arguments (mostly disabled to match 01.train.py)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--mosaic', type=float, default=0.0)
    parser.add_argument('--mixup', type=float, default=0.0)
    parser.add_argument('--cutmix', type=float, default=0.0)
    parser.add_argument('--copy-paste', type=float, default=0.0)
    parser.add_argument('--degrees', type=float, default=0.0)
    parser.add_argument('--translate', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--shear', type=float, default=0.0)
    parser.add_argument('--perspective', type=float, default=0.0)
    parser.add_argument('--flipud', type=float, default=0.0)
    parser.add_argument('--fliplr', type=float, default=0.5)
    parser.add_argument('--hsv_h', type=float, default=0.015)
    parser.add_argument('--hsv_s', type=float, default=0.7)
    parser.add_argument('--hsv_v', type=float, default=0.4)
    parser.add_argument('--copy_paste_mode', type=str, default='flip')
    parser.add_argument('--mask_ratio', type=int, default=4)
    parser.add_argument('--overlap_mask', action='store_true', default=True)
    parser.add_argument('--bgr', action='store_true', default=False, help='bgr to rgb conversion')


    args = parser.parse_args()
    main(args)
    main(args)
