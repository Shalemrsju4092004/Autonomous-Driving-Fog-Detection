import os
import torch
import torchvision
from PIL import Image, ImageDraw
import torchvision.transforms as T

# === Paths ===
images_dir = r"input datafile "
output_dir = r"output data file"

os.makedirs(output_dir, exist_ok=True)

# === Load Pretrained Faster R-CNN Model ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.to(device).eval()

# === Transform ===
transform = T.ToTensor()

# === Process All Images ===
for img_name in os.listdir(images_dir):
    if not img_name.lower().endswith(".png"):
        continue

    img_path = os.path.join(images_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Run Inference
    with torch.no_grad():
        outputs = model(img_tensor)[0]

    # Draw boxes with probability only
    draw = ImageDraw.Draw(image)
    for box, score in zip(outputs["boxes"], outputs["scores"]):
        if score > 0.4:  # confidence threshold
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            probability = score.item() * 100
            draw.rectangle([xmin, ymin, xmax, ymax], outline="lime", width=3)
            draw.text((xmin, ymin - 10), f"{probability:.1f}%", fill="lime")

    # Save output image
    save_path = os.path.join(output_dir, img_name.replace(".png", "_out.png"))
    image.save(save_path)
    print(f"✅ Saved: {save_path}")

print("🎉 All foggy images processed and saved with detection probabilities only.")
