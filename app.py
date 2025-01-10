from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from ultralytics import YOLO
import uuid
from flask_socketio import SocketIO, emit
import threading

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'avi', 'mov', 'mp4', 'mkv', 'wmv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

socketio = SocketIO(app, cors_allowed_origins="*")

# Load the YOLO model once at startup
DEFAULT_MODEL_PATH = 'yolov8s.pt'  # Change to your default model
if os.path.exists(DEFAULT_MODEL_PATH):
    model = YOLO(DEFAULT_MODEL_PATH)
else:
    raise FileNotFoundError(f"Default model not found at {DEFAULT_MODEL_PATH}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def perform_yolo_detection(model, input_path, output_path, resolution=None, thresh=0.5):
    # Load the image or video
    if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        frame = cv2.imread(input_path)
        if resolution:
            resW, resH = map(int, resolution.lower().split('x'))
            frame = cv2.resize(frame, (resW, resH))
        
        results = model(frame, conf=thresh, verbose=False)
        annotated_frame = results[0].plot()
        
        # Save the result
        cv2.imwrite(output_path, annotated_frame)
        return {'status': 'success', 'output_path': output_path}
    
    elif input_path.lower().endswith(('.avi', '.mov', '.mp4', '.mkv', '.wmv')):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return {'status': 'error', 'message': 'Cannot open video file.'}
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if resolution:
            width, height = map(int, resolution.lower().split('x'))
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if resolution:
                frame = cv2.resize(frame, (width, height))
            results = model(frame, conf=thresh, verbose=False)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        return {'status': 'success', 'output_path': output_path}
    
    else:
        return {'status': 'error', 'message': 'Unsupported file type.'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check for file in request
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part in the request.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected for uploading.'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_id + "_" + filename)
        file.save(input_path)
        
        # Get parameters
        model_path = request.form.get('model', DEFAULT_MODEL_PATH)
        resolution = request.form.get('resolution')
        thresh = float(request.form.get('thresh', 0.5))
        
        # Validate model path
        if not os.path.exists(model_path):
            return jsonify({'status': 'error', 'message': 'Model path does not exist.'}), 400
        
        # Define output path
        output_filename = f"result_{unique_id}_{filename}"
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        
        # Load the specified model if different from default
        if model_path != DEFAULT_MODEL_PATH:
            detection_model = YOLO(model_path)
        else:
            detection_model = model
        
        # Perform detection
        result = perform_yolo_detection(detection_model, input_path, output_path, resolution, thresh)
        
        if result['status'] == 'success':
            return send_file(result['output_path'], mimetype='image/jpeg')
        else:
            return jsonify({'status': 'error', 'message': result['message']}), 500
    else:
        return jsonify({'status': 'error', 'message': 'Allowed file types are jpg, jpeg, png, bmp, avi, mov, mp4, mkv, wmv.'}), 400

@app.route('/api/detect', methods=['POST'])
def api_detect():
    return upload_file()  # Reuse the upload_file function

@app.route('/api/detect/multiple', methods=['POST'])
def api_detect_multiple():
    if 'files' not in request.files:
        return jsonify({'status': 'error', 'message': 'No files part in the request.'}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({'status': 'error', 'message': 'No files selected for uploading.'}), 400
    
    model_path = request.form.get('model', DEFAULT_MODEL_PATH)
    resolution = request.form.get('resolution')
    thresh = float(request.form.get('thresh', 0.5))
    
    # Validate model path
    if not os.path.exists(model_path):
        return jsonify({'status': 'error', 'message': 'Model path does not exist.'}), 400
    
    results_list = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_id + "_" + filename)
            file.save(input_path)
            
            # Define output path
            output_filename = f"result_{unique_id}_{filename}"
            output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
            
            # Use default or specified model
            if model_path != DEFAULT_MODEL_PATH:
                detection_model = YOLO(model_path)
            else:
                detection_model = model
            
            # Perform detection
            result = perform_yolo_detection(detection_model, input_path, output_path, resolution, thresh)
            
            if result['status'] == 'success':
                results_list.append({
                    'input_file': filename,
                    'output_file': output_filename
                })
            else:
                results_list.append({
                    'input_file': filename,
                    'status': 'error',
                    'message': result['message']
                })
        else:
            results_list.append({
                'input_file': file.filename,
                'status': 'error',
                'message': 'Unsupported file type.'
            })
    
    return jsonify({'status': 'success', 'results': results_list}), 200

# WebSocket for streaming (Optional)
@app.route('/stream')
def stream():
    return render_template('stream.html')

@socketio.on('start_stream')
def handle_start_stream(data):
    model_path = data.get('model', DEFAULT_MODEL_PATH)
    resolution = data.get('resolution')
    thresh = float(data.get('thresh', 0.5))
    
    if not os.path.exists(model_path):
        emit('error', {'message': 'Model path does not exist.'})
        return
    
    # Load the specified model if different from default
    if model_path != DEFAULT_MODEL_PATH:
        detection_model = YOLO(model_path)
    else:
        detection_model = model
    
    def generate_frames():
        cap = cv2.VideoCapture(0)  # Change index if multiple cameras
        if not cap.isOpened():
            emit('error', {'message': 'Cannot open camera.'})
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if resolution:
                resW, resH = map(int, resolution.lower().split('x'))
                frame = cv2.resize(frame, (resW, resH))
            results = detection_model(frame, conf=thresh, verbose=False)
            annotated_frame = results[0].plot()
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            # Emit the frame
            emit('frame', {'image': frame_bytes.hex()})
        
        cap.release()
    
    thread = threading.Thread(target=generate_frames)
    thread.start()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
