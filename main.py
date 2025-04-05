import cv2
import torch
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms

# Load NanoDet ONNX model
NANODET_MODEL_PATH = "nanodet/models/nanodet-plus-m-1.5x_416.onnx"
nanodet_session = ort.InferenceSession(NANODET_MODEL_PATH, providers=['CPUExecutionProvider'])

# Load MiDaS model
midas = torch.hub.load("isl-org/MiDaS", "MiDaS_small")
midas.eval()

# MiDaS image transformation
midas_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_nanodet(image):
    """Prepare image for NanoDet ONNX model"""
    input_size = (416, 416)
    resized_image = cv2.resize(image, input_size)
    blob = resized_image.astype(np.float32) / 255.0  # Normalize to [0,1]
    blob = np.transpose(blob, (2, 0, 1))  # Change to (C, H, W)
    blob = np.expand_dims(blob, axis=0).astype(np.float32)  # Add batch dimension
    return blob

def run_nanodet(image):
    """Run NanoDet and return detected objects"""
    input_data = preprocess_nanodet(image)
    outputs = nanodet_session.run(None, {"input": input_data})
    return outputs  # Process outputs as needed

def run_midas(image):
    """Run MiDaS for depth estimation"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = midas_transform(img).unsqueeze(0)

    with torch.no_grad():
        depth_map = midas(img_tensor)

    depth_map = depth_map.squeeze().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return depth_map

# Load input image
image_path = "test.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

# Run NanoDet for object detection
nanodet_results = run_nanodet(image)

# Run MiDaS for depth estimation
depth_map = run_midas(image)

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Depth Map (MiDaS)", depth_map)

cv2.waitKey(0)
cv2.destroyAllWindows()
