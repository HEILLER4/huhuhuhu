import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load NanoDet ONNX model
net = cv2.dnn.readNetFromONNX("nanodet-plus-m-1.5x_416.onnx")
print("Model loaded successfully")


def run_nanodet_onnx(image, conf_threshold=0.4):
    input_size = (416, 416)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, input_size,
                                 mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Debug: Print input blob shape
    print(f"Input blob shape: {blob.shape}")

    try:
        output = net.forward()
        print(f"Raw output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return []

    detections = []
    for i, det in enumerate(output[0]):
        class_id = int(det[0])
        score = float(det[1])

        # Debug: Print all raw detections
        if i < 5:  # Print first 5 for inspection
            print(f"Raw detection {i}: class={class_id}, score={score:.2f}, coords={det[2:6]}")

        if score > conf_threshold:
            cx, cy, w, h = det[2:6]
            x1 = int((cx - w / 2) * image.shape[1])
            y1 = int((cy - h / 2) * image.shape[0])
            x2 = int((cx + w / 2) * image.shape[1])
            y2 = int((cy + h / 2) * image.shape[0])

            # Debug: Print valid detections
            print(f"Valid detection: class={class_id}, score={score:.2f}, box=({x1},{y1},{x2},{y2})")

            detections.append((class_id, score, (x1, y1, x2, y2)))

    print(f"Total detections: {len(detections)}")
    return detections


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("NanoDet Debugger")

        # Debug console
        self.console = tk.Text(root, height=10, state='disabled')
        self.console.pack(fill=tk.X)

        # Image display
        self.canvas = tk.Canvas(root, width=800, height=600, bg='gray')
        self.canvas.pack()

        # Controls
        self.btn = tk.Button(root, text="Open Image", command=self.load_image)
        self.btn.pack(pady=10)

    def log(self, message):
        self.console.config(state='normal')
        self.console.insert(tk.END, message + "\n")
        self.console.config(state='disabled')
        self.console.see(tk.END)
        print(message)  # Also print to terminal

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.log(f"\nProcessing: {file_path}")
            image = cv2.imread(file_path)
            if image is None:
                self.log("Error: Failed to load image")
                return

            self.log(f"Image shape: {image.shape}")
            detections = run_nanodet_onnx(image)

            if not detections:
                self.log("No detections found!")
            else:
                self.log(f"Found {len(detections)} objects")

            # Convert to RGB and draw boxes
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for class_id, score, (x1, y1, x2, y2) in detections:
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"{class_id}:{score:.2f}"
                cv2.putText(image_rgb, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                self.log(f"Drew box: {label} at ({x1},{y1})-({x2},{y2})")

            # Display
            img_pil = Image.fromarray(image_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()