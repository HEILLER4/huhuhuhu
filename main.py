import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import os


class DepthDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 + MiDaS Detector")
        self.root.geometry("1200x800")

        # Model initialization
        self.yolo_model = None
        self.midas_model = None
        self.load_models()

        # GUI Setup
        self.setup_gui()
        self.image = None
        self.depth_colormap = None
        self.detections = None

    def load_models(self):
        """Load both models with offline fallback"""
        try:
            # YOLOv8 (will auto-download if not found)
            self.yolo_model = YOLO('yolov8n.pt')

            # MiDaS (will use local cache if available)
            self.midas_model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid', trust_repo=True)
            self.midas_model.eval()

            # Cache models for offline use
            torch.save(self.midas_model.state_dict(), 'midas_hybrid.pt')
        except Exception as e:
            print(f"Model loading error: {e}")
            self.load_offline_models()

    def load_offline_models(self):
        """Attempt to load pre-downloaded models"""
        try:
            # YOLOv8 offline
            if os.path.exists('yolov8n.pt'):
                self.yolo_model = YOLO('yolov8n.pt')

            # MiDaS offline
            if os.path.exists('midas_hybrid.pt'):
                self.midas_model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid', pretrained=False)
                self.midas_model.load_state_dict(torch.load('midas_hybrid.pt'))
                self.midas_model.eval()
        except Exception as e:
            print(f"Offline load failed: {e}")

    def setup_gui(self):
        """Create the interface"""
        # Control Panel
        control_frame = tk.Frame(self.root, bd=2, relief=tk.RIDGE)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Model selection
        tk.Label(control_frame, text="YOLOv8 Model:").grid(row=0, column=0, padx=5)
        self.yolo_var = tk.StringVar(value='yolov8n.pt')
        yolo_options = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
        ttk.Combobox(control_frame, textvariable=self.yolo_var, values=yolo_options).grid(row=0, column=1, padx=5)

        # Confidence threshold
        tk.Label(control_frame, text="Confidence:").grid(row=0, column=2, padx=5)
        self.conf_var = tk.DoubleVar(value=0.4)
        tk.Scale(control_frame, variable=self.conf_var, from_=0.1, to=0.9,
                 resolution=0.05, orient=tk.HORIZONTAL, length=150).grid(row=0, column=3, padx=5)

        # Buttons
        tk.Button(control_frame, text="Load Image", command=self.load_image).grid(row=0, column=4, padx=5)
        tk.Button(control_frame, text="Run Detection", command=self.run_detection).grid(row=0, column=5, padx=5)

        # Display Area
        display_frame = tk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True)

        # Original Image
        self.original_label = tk.Label(display_frame, bd=2, relief=tk.SUNKEN)
        self.original_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # YOLO Detections
        self.detection_label = tk.Label(display_frame, bd=2, relief=tk.SUNKEN)
        self.detection_label.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # Depth Map
        self.depth_label = tk.Label(display_frame, bd=2, relief=tk.SUNKEN)
        self.depth_label.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Combined View
        self.combined_label = tk.Label(display_frame, bd=2, relief=tk.SUNKEN)
        self.combined_label.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        # Configure grid weights
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_rowconfigure(1, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)
        display_frame.grid_columnconfigure(1, weight=1)

    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is not None:
                self.show_image(self.image, self.original_label)
                # Clear previous results
                for label in [self.detection_label, self.depth_label, self.combined_label]:
                    label.config(image='')
                    label.image = None

    def run_detection(self):
        """Run both detection and depth estimation"""
        if self.image is None:
            return

        # Update YOLO model if changed
        if self.yolo_var.get() != getattr(self, 'current_yolo_model', None):
            self.yolo_model = YOLO(self.yolo_var.get())
            self.current_yolo_model = self.yolo_var.get()

        # YOLOv8 Detection
        self.yolo_model.conf = self.conf_var.get()
        results = self.yolo_model(self.image)
        self.detections = results[0].plot()
        self.show_image(self.detections, self.detection_label)

        # MiDaS Depth Estimation
        depth_map = self.estimate_depth(self.image)
        self.depth_colormap = self.colorize_depth(depth_map)
        self.show_image(self.depth_colormap, self.depth_label)

        # Combined View
        combined = cv2.addWeighted(self.detections, 0.7, self.depth_colormap, 0.3, 0)
        self.show_image(combined, self.combined_label)

    def estimate_depth(self, image):
        """Run MiDaS depth estimation"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (384, 384))  # MiDaS input size

        # Convert to tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        # Normalize (MiDaS specific)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img_tensor = (img_tensor - mean[:, None, None]) / std[:, None, None]

        # Inference
        with torch.no_grad():
            depth = self.midas_model(img_tensor.unsqueeze(0))

        # Post-process
        depth = depth.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]))
        return depth

    def colorize_depth(self, depth):
        """Convert depth map to color visualization"""
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
        return depth_colored

    def show_image(self, img, label_widget):
        """Display an OpenCV image in a Tkinter label"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Calculate aspect ratio preserving dimensions
        max_size = (400, 400)
        img_pil.thumbnail(max_size, Image.LANCZOS)

        img_tk = ImageTk.PhotoImage(img_pil)
        label_widget.config(image=img_tk)
        label_widget.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = DepthDetectionApp(root)
    root.mainloop()