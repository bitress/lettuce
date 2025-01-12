from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from ultralytics import YOLO
import uuid
from flask_socketio import SocketIO
import mysql.connector
from mysql.connector import Error
import base64

# Flask application setup
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

socketio = SocketIO(app, cors_allowed_origins="*")

DEFAULT_MODEL_PATH = 'my_model.pt'
if os.path.exists(DEFAULT_MODEL_PATH):
    model = YOLO(DEFAULT_MODEL_PATH)
else:
    raise FileNotFoundError(f"Default model not found at {DEFAULT_MODEL_PATH}")


# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Perform YOLO detection and log results to MySQL
def perform_yolo_detection(model, input_path, output_path, resolution=None, thresh=0.5):
    try:
        # Check if the input file is a supported image type
        if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Read the image
            frame = cv2.imread(input_path)

            # Resize the image if resolution is specified
            if resolution:
                resW, resH = map(int, resolution.lower().split('x'))
                frame = cv2.resize(frame, (resW, resH))

            # Perform YOLO detection
            results = model(frame, conf=thresh, verbose=False)
            annotated_frame = results[0].plot()

            # Convert the original image to Base64
            with open(input_path, "rb") as raw_file:
                raw_image_b64 = base64.b64encode(raw_file.read()).decode('utf-8')

            # Convert the annotated frame to Base64
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')

            # Save annotated image to output path
            cv2.imwrite(output_path, annotated_frame)

            # Insert Base64-encoded images into the database
            insert_detection_log(raw_image_b64, annotated_image_b64)

            return {'status': 'success', 'output_path': output_path}
        else:
            return {'status': 'error', 'message': 'Unsupported file type.'}

    except Exception as e:
        return {'status': 'error', 'message': str(e)}


# Insert detection log into MySQL database
def insert_detection_log(raw_image, annotated_image):
    try:
        # Establish a connection to the database
        connection = mysql.connector.connect(
            host='139.99.97.250',
            user='hydroponics',
            password=')[ZEy032Zy_oe8C8',
            database='hydroponics'
        )

        if connection.is_connected():
            cursor = connection.cursor()

            # Insert query
            query = """
            INSERT INTO camera_logs (raw_image, annotated_image)
            VALUES (%s, %s)
            """
            cursor.execute(query, (raw_image, annotated_image))
            connection.commit()
            print("Record inserted successfully into camera_logs table.")

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed.")


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part in the request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected for uploading.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_id + "_" + filename)
        file.save(input_path)

        model_path = request.form.get('model', DEFAULT_MODEL_PATH)
        resolution = request.form.get('resolution')
        thresh = float(request.form.get('thresh', 0.5))

        if not os.path.exists(model_path):
            return jsonify({'status': 'error', 'message': 'Model path does not exist.'}), 400

        output_filename = f"result_{unique_id}_{filename}"
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)

        if model_path != DEFAULT_MODEL_PATH:
            detection_model = YOLO(model_path)
        else:
            detection_model = model

        result = perform_yolo_detection(detection_model, input_path, output_path, resolution, thresh)

        if result['status'] == 'success':
            return send_file(result['output_path'], mimetype='image/jpeg')
        else:
            return jsonify({'status': 'error', 'message': result['message']}), 500
    else:
        return jsonify({'status': 'error', 'message': 'Allowed file types are jpg, jpeg, png, bmp.'}), 400


@app.route('/api/detect', methods=['POST'])
def api_detect():
    return upload_file()


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8086)
