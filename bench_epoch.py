# bench_epoch.py  --  測 n_batch 的資料載入與 GPU 前向
import time, torch, argparse, yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--batches", type=int, default=50)
parser.add_argument("--model",   default="models/yolov8s.pt")
parser.add_argument("--data",    default="yaml/person.yaml")
parser.add_argument("--imgsz",   type=int, default=640)
parser.add_argument("--batch",   type=int, default=12)
parser.add_argument("--workers", type=int, default=12)
opt = parser.parse_args()

# Manually parse the data YAML to get the actual image path
with open(opt.data, 'r') as f:
    data_config = yaml.safe_load(f)
train_img_path = data_config['train']

model_obj = YOLO(opt.model)
trainer = DetectionTrainer(overrides={
    "model": opt.model, "data": opt.data,
    "imgsz": opt.imgsz, "batch": opt.batch, "workers": opt.workers,
})
trainer.model = model_obj.model # Explicitly set trainer's model to the loaded DetectionModel
loader = trainer.get_dataloader(train_img_path, batch_size=opt.batch, rank=-1, mode="train")
model  = model_obj.model.cuda().eval()

dl, gpu = 0.0, 0.0
for i, batch in zip(range(opt.batches), loader):
    t0 = time.perf_counter()
    imgs = batch["img"]
    dl += time.perf_counter() - t0

    t1 = time.perf_counter()
    with torch.inference_mode():
        _ = model(imgs.cuda(non_blocking=True).float() / 255)
    gpu += time.perf_counter() - t1

print(f"{{\"dl_time\": {dl/opt.batches:.6f}, \"gpu_time\": {gpu/opt.batches:.6f}}}")