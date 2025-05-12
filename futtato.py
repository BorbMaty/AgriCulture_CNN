import torch
from pathlib import Path

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/brassai_detector3/weights/best.pt')

# Directory of test images
test_folder = '/home/lucy/Desktop/brassai_projekt/test_tractors'
image_paths = list(Path(test_folder).glob("*.jpg")) + list(Path(test_folder).glob("*.png"))

# Loop over images
for image_path in image_paths:
    results = model(str(image_path))
    print(f"📸 Processed {image_path.name}")
    results.print()
    results.save()   # saves to 'runs/detect/exp'
