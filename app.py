from flask import Flask, render_template, Response, request
from ultralytics import YOLO
import cv2
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load models
helmet_model = YOLO("ppe.pt")
plate_model = YOLO("numberplate.pt")

print("✅ Models loaded successfully!")

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ----------- LIVE DETECTION -----------
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run helmet detection
        results = helmet_model(frame)
        names = helmet_model.names
        detected_no_helmet = False

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = names[cls].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if label in ["hardhat", "helmet"]:
                color = (0, 255, 0)
                text = f"Helmet ({conf:.2f})"
            elif label in ["no-hardhat", "no_helmet"]:
                color = (0, 0, 255)
                text = f"No Helmet ({conf:.2f})"
                detected_no_helmet = True
            elif label == "person":
                color = (255, 255, 0)
                text = "Person"
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # If no helmet → run number plate detection
        if detected_no_helmet:
            plate_results = plate_model(frame)
            for pbox in plate_results[0].boxes:
                x1, y1, x2, y2 = map(int, pbox.xyxy[0])
                conf = float(pbox.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f"Plate ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ----------- IMAGE UPLOAD DETECTION -----------
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded image
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Run helmet detection
    frame = cv2.imread(filepath)
    results = helmet_model(frame)
    names = helmet_model.names
    detected_no_helmet = False

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = names[cls].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        if label in ["hardhat", "helmet"]:
            color = (0, 255, 0)
            text = f"Helmet ({conf:.2f})"
        elif label in ["no-hardhat", "no_helmet"]:
            color = (0, 0, 255)
            text = f"No Helmet ({conf:.2f})"
            detected_no_helmet = True
        elif label == "person":
            color = (255, 255, 0)
            text = "Person"
        else:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # If no helmet detected → number plate detection
    if detected_no_helmet:
        plate_results = plate_model(frame)
        for pbox in plate_results[0].boxes:
            x1, y1, x2, y2 = map(int, pbox.xyxy[0])
            conf = float(pbox.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, f"Plate ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"out_{filename}")
    cv2.imwrite(output_path, frame)
    return render_template('index.html', uploaded_image=output_path)

# ----------- ROUTES -----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
