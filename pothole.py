from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# ✅ Load the trained pothole model
from ultralytics import YOLO
model = YOLO("models/pothole_yolov8n.pt")
print("Model classes:", model.names)


# ✅ Load the same image used in Colab
image_path = "test_images/pothole.jpg"  # <-- replace with your exact image path
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ✅ Run inference
results = model(img_rgb)[0]
print("Detections:", results.names)
for box in results.boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = results.names[cls_id]
    print(f"Detected {label} with confidence {conf:.2f}")


    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img_rgb, f"{label} {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# ✅ Show image with detection
plt.imshow(img_rgb)
plt.axis("off")
plt.title("YOLOv8 Pothole Detection")
plt.show()
