from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64
import io
import os

app = Flask(__name__)

# Load model once when app starts
model = YOLO("yolov8n.pt")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    try:
        image_data = request.json["image"]

        image_bytes = base64.b64decode(image_data.split(",")[1])
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = np.array(img)

        results = model(frame)[0]
        person_count = 0

        for box in results.boxes:
            cls = int(box.cls[0])

            if cls == 0:  # Person class
                person_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    f"Person {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        _, buffer = cv2.imencode(".jpg", frame)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "image": encoded_image,
            "count": person_count
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
