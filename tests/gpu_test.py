"""Compare the energy of 200 VGG16 inferences on the CPU and on the first NVIDIA GPU.

Needs torch, torchvision, timm and pillow (see requirements.txt). The model weights are
downloaded by timm on the first run; the input image is generated locally, since its content
does not matter for energy and remote image links tend to break.
"""
from energymeter import EnergyMeter
import time
import numpy as np
from PIL import Image
import timm
import torch
import torchvision

N_INFERENCES = 200

transforms = torchvision.transforms.Compose([
    torchvision.transforms.transforms.Resize(size=256),
    torchvision.transforms.transforms.CenterCrop(size=(224, 224)),
    torchvision.transforms.transforms.ToTensor(),
    torchvision.transforms.transforms.Normalize(mean=torch.Tensor([0.4850, 0.4560, 0.4060]),
                                                 std=torch.Tensor([0.2290, 0.2240, 0.2250])),
])


def predict(model, img):
    with torch.no_grad():
        output = model(img)
    top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
    return top5_probabilities, top5_class_indices.tolist()[0]


def preprocess_data(img):
    # unsqueeze single image into batch of 1.
    return transforms(img).unsqueeze(0)


def summary(meter):
    joules = {k: float(np.sum(v)) for k, v in meter.get_total_joules_per_component().items()}
    parts = ", ".join(f"{k} {v:.1f} J" for k, v in joules.items())
    return f"{meter.label}: {meter.duration:.1f} s, {sum(joules.values()):.1f} J ({parts})"


if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU available: this test compares CPU and NVIDIA GPU inference.")

model = timm.create_model('vgg16.tv_in1k', pretrained=True)
model = model.eval()

# A fixed random 640x480 RGB image.
rng = np.random.default_rng(42)
img = Image.fromarray(rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8))

# CPU
# Disk in LSD15 has an avg i/o speed of 1600MB/s and
# a power consumption of 6W when active and 1.42W when idle.
meter_cpu = EnergyMeter(disk_avg_speed=1600*1e6,
                        disk_active_power=6,
                        disk_idle_power=1.42,
                        label="CPU Inference")
model = model.to("cpu")

meter_cpu.begin()
for _ in range(N_INFERENCES):
    p_img = preprocess_data(img).to("cpu")
    probs, labels = predict(model, p_img)
meter_cpu.end()

# To ensure samples from GPU power stats are not mixed up.
time.sleep(1)

meter_gpu = EnergyMeter(disk_avg_speed=1600*1e6,
                        disk_active_power=6,
                        disk_idle_power=1.42,
                        label="GPU Inference")
model = model.to("cuda:0")
meter_gpu.begin()
for _ in range(N_INFERENCES):
    p_img = preprocess_data(img).to("cuda:0")
    probs, labels = predict(model, p_img)
torch.cuda.synchronize()  # wait for queued GPU work, so it falls inside the measurement
meter_gpu.end()

print(summary(meter_cpu))
print(summary(meter_gpu))
meter_cpu.cleanup()
meter_gpu.cleanup()
