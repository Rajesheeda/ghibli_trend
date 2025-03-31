import torch
import cv2
import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image
from model import Generator

# Load AnimeGANv2 model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Generator().to(device)
model.load_state_dict(torch.load("weights/face_paint_512_v2.pt", map_location=device))
model.eval()

def transform_to_ghibli(image_path, output_path="ghibli_output.jpg"):
    # Load image
    try:
        img = Image.open(image_path).convert("RGB")
    except OSError as e:
        print(f"Error opening image: {e}")
    # Handle the error, e.g., skip the image or display a message
    img = img.resize((512, 512))  # Resize for model input
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)

    # Apply Ghibli transformation
    with torch.no_grad():
        output = model(img_tensor).squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
        output = (output * 255).astype(np.uint8)

    # Save the transformed image
    cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print(f"Ghibli-style image saved as {output_path}")

# Convert your image (replace 'input.jpg' with your actual file)
transform_to_ghibli("/content/IMG_20180518_124734.jpg")
