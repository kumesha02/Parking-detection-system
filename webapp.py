########------initial web interface------######
#
# import json
# import argparse
# import io
# from PIL import Image
# import datetime
#
# import torch
# import cv2
# import numpy as np
# import tensorflow as tf
# from re import DEBUG, sub
# from flask import Flask, render_template, request, redirect, send_file, url_for, Response
# from werkzeug.utils import secure_filename, send_from_directory
# import os
# import subprocess
# from subprocess import Popen
# import re
# import requests
# import shutil
# import time
# import glob
#
#
# from ultralytics import YOLO
#
#
# app = Flask(__name__)
#
#
# @app.route("/")
# def hello_world():
#     return render_template('index.html')
#
#
# @app.route("/", methods=["GET", "POST"])
# def predict_img():
#     if request.method == "POST":
#         if 'file' in request.files:
#             f = request.files['file']
#             basepath = os.path.dirname(__file__)
#             filepath = os.path.join(basepath,'uploads',f.filename)
#             print("upload folder is ", filepath)
#             f.save(filepath)
#             global imgpath
#             predict_img.imgpath = f.filename
#             print("printing predict_img :::::: ", predict_img)
#
#             file_extension = f.filename.rsplit('.', 1)[1].lower()
#
#             if file_extension == 'jpg':
#                 img = cv2.imread(filepath)
#
#                 # Perform the detection
#                 model = YOLO('yolov9c.pt')
#                 detections =  model(img, save=True)
#                 return display(f.filename)
#
#             elif file_extension == 'mp4':
#                 video_path = filepath  # replace with your video path
#                 cap = cv2.VideoCapture(video_path)
#
#                 # get video dimensions
#                 frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#                 # Define the codec and create VideoWriter object
#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
#
#                 # initialize the YOLOv8 model here
#                 model = YOLO('yolov9c.pt')
#
#                 while cap.isOpened():
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#
#                     # do YOLOv9 detection on the frame here
#                     #model = YOLO('yolov9c.pt')
#                     results = model(frame, save=True)  #working
#                     print(results)
#                     cv2.waitKey(1)
#
#                     res_plotted = results[0].plot()
#                     cv2.imshow("result", res_plotted)
#
#                     # write the frame to the output video
#                     out.write(res_plotted)
#
#                     if cv2.waitKey(1) == ord('q'):
#                         break
#
#                 return video_feed()
#
#
#
#     folder_path = 'runs/detect'
#     subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
#     latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
#     image_path = folder_path+'/'+latest_subfolder+'/'+f.filename
#     return render_template('index.html', image_path=image_path)
#     #return "done"
#
#
#
# # #The display function is used to serve the image or video from the folder_path directory.
# @app.route('/<path:filename>')
# def display(filename):
#     folder_path = 'runs/detect'
#     subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
#     latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
#     directory = folder_path+'/'+latest_subfolder
#     print("printing directory: ",directory)
#     files = os.listdir(directory)
#     latest_file = files[0]
#
#     print(latest_file)
#
#     filename = os.path.join(folder_path, latest_subfolder, latest_file)
#
#     file_extension = filename.rsplit('.', 1)[1].lower()
#
#     environ = request.environ
#     if file_extension == 'jpg':
#         return send_from_directory(directory,latest_file,environ) #shows the result in seperate tab
#
#     else:
#         return "Invalid file format"
#
#
#
#
# def get_frame():
#     folder_path = os.getcwd()
#     mp4_files = 'output.mp4'
#     video = cv2.VideoCapture(mp4_files)  # detected video path
#     while True:
#         success, image = video.read()
#         if not success:
#             break
#         ret, jpeg = cv2.imencode('.jpg', image)
#
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
#         time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds:
#
#
# # function to display the detected objects video on html page
# @app.route("/video_feed")
# def video_feed():
#     print("function called")
#
#     return Response(get_frame(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
#
#
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Flask app exposing yolov9 models")
#     parser.add_argument("--port", default=5000, type=int, help="port number")
#     args = parser.parse_args()
#     model = YOLO('yolov9c.pt')
#     app.run(host="0.0.0.0", port=args.port)


# import argparse
# from flask import Flask, render_template, request, Response, jsonify
# import cv2
# import os
# import time
# from ultralytics import YOLO
# from datetime import datetime
# import sqlite3
#
# app = Flask(__name__)
#
# # Database setup
# DB_FILE = 'vehicle_data.db'
#
#
# def init_db():
#     conn = sqlite3.connect(DB_FILE)
#     c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS vehicle_count (
#                     id INTEGER PRIMARY KEY AUTOINCREMENT,
#                     date TEXT NOT NULL,
#                     total_count INTEGER NOT NULL
#                 )''')
#     conn.commit()
#     conn.close()
#
#
# init_db()
#
# # Model initialization
# model = YOLO('yolov9c.pt')
#
# # Global variables
# total_vehicle_count = 0
#
#
# @app.route("/")
# def home():
#     conn = sqlite3.connect(DB_FILE)
#     c = conn.cursor()
#     c.execute("SELECT SUM(total_count) FROM vehicle_count WHERE date = ?", (datetime.now().strftime('%Y-%m-%d'),))
#     daily_count = c.fetchone()[0] or 0
#     conn.close()
#     return render_template('index.html', daily_count=daily_count)
#
#
# @app.route("/start_detection", methods=["POST"])
# def start_detection():
#     global total_vehicle_count
#     rtsp_url = request.form.get('rtsp_url')
#     if not rtsp_url:
#         return jsonify({'error': 'Invalid RTSP URL'}), 400
#
#     cap = cv2.VideoCapture(rtsp_url)
#
#     if not cap.isOpened():
#         return jsonify({'error': 'Unable to open video stream'}), 500
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         results = model(frame)
#         detections = results[0].boxes
#
#         # Count vehicles based on YOLO classes
#         for box in detections:
#             if box.cls in [2, 3, 5, 7]:  # Example: car, truck, bus, motorcycle
#                 total_vehicle_count += 1
#
#         # Display or log detection (optional)
#         res_plotted = results[0].plot()
#         cv2.imshow("Detection", res_plotted)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     # Store the daily count in the database
#     conn = sqlite3.connect(DB_FILE)
#     c = conn.cursor()
#     c.execute("INSERT INTO vehicle_count (date, total_count) VALUES (?, ?)",
#               (datetime.now().strftime('%Y-%m-%d'), total_vehicle_count))
#     conn.commit()
#     conn.close()
#
#     return jsonify({'message': 'Detection completed', 'total_count': total_vehicle_count})
#
#
# @app.route("/video_feed")
# def video_feed():
#     def generate_frames():
#         cap = cv2.VideoCapture(rtsp_url)
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 break
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         cap.release()
#
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Flask app for YOLO vehicle detection")
#     parser.add_argument("--port", default=5000, type=int, help="Port number")
#     args = parser.parse_args()
#     app.run(host="0.0.0.0", port=args.port)
# import os
# import sqlite3
# import cv2
# from datetime import datetime
# from flask import Flask, render_template, request, Response
# from ultralytics import YOLO
#
# app = Flask(__name__)
# model = YOLO('yolov9c.pt')  # Pre-trained YOLOv9 model
#
# # SQLite database setup
# DB_PATH = "vehicles.db"
#
#
# def init_db():
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS vehicle_log (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             timestamp TEXT,
#             category TEXT
#         )
#     """)
#     conn.commit()
#     conn.close()
#
#
# init_db()
#
#
# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         input_type = request.form.get("input_type")
#         if input_type == "folder":
#             folder_path = request.form.get("folder_path")
#             video_path = os.path.join(folder_path, "input_video.mp4")  # Ensure file exists
#             return process_video(video_path)
#         elif input_type == "rtsp":
#             rtsp_url = request.form.get("rtsp_url")
#             return process_stream(rtsp_url)
#     return render_template("index.html")
#
#
# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     return Response(detect_and_log(cap), mimetype="multipart/x-mixed-replace; boundary=frame")
#
#
# def process_stream(rtsp_url):
#     cap = cv2.VideoCapture(rtsp_url)
#     return Response(detect_and_log(cap), mimetype="multipart/x-mixed-replace; boundary=frame")
#
#
# def detect_and_log(cap):
#     vehicle_categories = {"car": 0, "motorbike": 0, "three_wheeler": 0, "van": 0}
#
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # YOLO detection
#         results = model(frame)
#         for result in results:
#             for box in result.boxes:
#                 category = box.name.lower()
#                 if category in vehicle_categories:
#                     vehicle_categories[category] += 1
#                     cursor.execute(
#                         "INSERT INTO vehicle_log (timestamp, category) VALUES (?, ?)",
#                         (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), category)
#                     )
#
#         conn.commit()
#
#         # Render the frame with detected results
#         res_plotted = results[0].plot()
#         _, buffer = cv2.imencode('.jpg', res_plotted)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#
#     cap.release()
#     conn.close()
#
#
# @app.route("/vehicle_stats")
# def vehicle_stats():
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute("""
#         SELECT category, COUNT(*) FROM vehicle_log GROUP BY category
#     """)
#     stats = cursor.fetchall()
#     conn.close()
#     return render_template("stats.html", stats=stats)
#
#
# if __name__ == "__main__":
#     app.run(debug=True)
# from flask import Flask, render_template, request, Response, jsonify, stream_with_context
# import cv2
# import sqlite3
# import threading
# from ultralytics import YOLO
# from datetime import datetime
# import time
#
# app = Flask(__name__)
#
# # Database setup
# DB_FILE = 'vehicle_data.db'
#
# def init_db():
#     conn = sqlite3.connect(DB_FILE)
#     c = conn.cursor()
#     # Create table for vehicle logs
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS vehicle_log (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             timestamp TEXT NOT NULL,
#             category TEXT NOT NULL
#         )
#     ''')
#     conn.commit()
#     conn.close()
#
# init_db()
#
# # YOLO model setup
# model = YOLO('yolov9c.pt')
#
# # Global variables
# lock = threading.Lock()
# current_frame = None
# vehicle_categories = {
#     0: 'car',
#     1: 'motorbike',
#     2: 'three-wheeler',
#     3: 'van'
# }
# vehicle_count = 0
#
# @app.route("/")
# def home():
#     return render_template("index.html")
#
# @app.route("/start_detection", methods=["POST"])
# def start_detection():
#     global current_frame, vehicle_count
#
#     input_type = request.form.get("input_type")
#     input_value = request.form.get("input_value")
#
#     if input_type not in ["video", "rtsp"]:
#         return jsonify({"error": "Invalid input type"}), 400
#
#     if not input_value:
#         return jsonify({"error": "No input provided"}), 400
#
#     # Start video capture
#     cap = cv2.VideoCapture(input_value if input_type == "video" else input_value)
#
#     if not cap.isOpened():
#         return jsonify({"error": "Unable to open video stream"}), 500
#
#     def process_video():
#         global current_frame, vehicle_count
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             # Run YOLO model on frame
#             results = model(frame)
#             detections = results[0].boxes
#
#             # Process detections
#             conn = sqlite3.connect(DB_FILE)
#             c = conn.cursor()
#             for box in detections:
#                 cls = int(box.cls)
#                 if cls in vehicle_categories:
#                     category = vehicle_categories[cls]
#                     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                     c.execute("INSERT INTO vehicle_log (timestamp, category) VALUES (?, ?)", (timestamp, category))
#                     vehicle_count += 1
#             conn.commit()
#             conn.close()
#
#             # Update current frame
#             with lock:
#                 current_frame = results[0].plot()
#
#         cap.release()
#
#     # Run the processing thread
#     threading.Thread(target=process_video).start()
#
#     return jsonify({"message": "Detection started"}), 200
#
# @app.route("/video_feed")
# def video_feed():
#     def generate_frames():
#         global current_frame
#         while True:
#             with lock:
#                 if current_frame is not None:
#                     _, buffer = cv2.imencode('.jpg', current_frame)
#                     frame = buffer.tobytes()
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#
#     return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
#
# @app.route("/live_data")
# def live_data():
#     def generate_data():
#         while True:
#             conn = sqlite3.connect(DB_FILE)
#             c = conn.cursor()
#             c.execute("SELECT timestamp, category FROM vehicle_log ORDER BY id DESC LIMIT 5")
#             rows = c.fetchall()
#             conn.close()
#
#             data = {
#                 "vehicle_count": vehicle_count,
#                 "vehicles": [{"timestamp": row[0], "category": row[1]} for row in rows]
#             }
#             yield f"data: {jsonify(data).get_data(as_text=True)}\n\n"
#             time.sleep(1)
#
#     return Response(stream_with_context(generate_data()), content_type="text/event-stream")
#
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
# from flask import Flask, render_template, request, Response, jsonify, stream_with_context
# import cv2
# import sqlite3
# import threading
# from ultralytics import YOLO
# from datetime import datetime
# import time
# import torch
# # print(torch.__version__)
# # print(torch.version.cuda)
# # print(torch.backends.cudnn.enabled)
# # print("CUDA available:", torch.cuda.is_available())
# app = Flask(__name__)
#
# # Database setup
# DB_FILE = 'vehicle_data.db'
#
# def init_db():
#     conn = sqlite3.connect(DB_FILE)
#     c = conn.cursor()
#     # Create table for vehicle logs
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS vehicle_log (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             timestamp TEXT NOT NULL,
#             category TEXT NOT NULL
#         )
#     ''')
#     conn.commit()
#     conn.close()
#
# init_db()
#
# ##### YOLO model setup
# # Model initialization with GPU support
# model = YOLO('yolov9c.pt')  # Ensure you're using the correct model path
# if torch.cuda.is_available():
#     model.to('cuda')  # Ensure model runs on GPU
#     print("CUDA is available. Running on GPU.")
# else:
#     print("CUDA is not available. Running on CPU.")
#
#
# # Global variables
# lock = threading.Lock()
# current_frame = None
# vehicle_categories = {
#     2: 'car',
#     3: 'motorbike',
#     5: 'three_wheeler',  # Placeholder for Three-Wheeler
#     7: 'van',  # Placeholder for Van
#     8: 'truck'
# }
# vehicle_count = 0
# category_counts = {
#     'car': 0,
#     'motorbike': 0,
#     'three_wheeler': 0,
#     'van': 0,
#     'truck': 0
# }
#
# @app.route("/")
# def home():
#     return render_template("index.html")
#
# @app.route("/start_detection", methods=["POST"])
# def start_detection():
#     global current_frame, vehicle_count, category_counts
#
#     input_type = request.form.get("input_type")
#     input_value = request.form.get("input_value")
#
#     if input_type not in ["video", "rtsp"]:
#         return jsonify({"error": "Invalid input type"}), 400
#
#     if not input_value:
#         return jsonify({"error": "No input provided"}), 400
#
#     # Start video capture
#     cap = cv2.VideoCapture(input_value if input_type == "video" else input_value)
#
#     if not cap.isOpened():
#         return jsonify({"error": "Unable to open video stream"}), 400
#
#     def detect_vehicles():
#         global current_frame, vehicle_count, category_counts
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             results = model(frame)
#             detections = results[0].boxes
#
#             # Reset category counts
#             category_counts = {key: 0 for key in category_counts}
#
#             # Detect and count vehicles
#             for box in detections:
#                 cls = int(box.cls)
#                 if cls in vehicle_categories:
#                     category_name = vehicle_categories[cls]
#                     category_counts[category_name] += 1
#                     vehicle_count += 1
#                     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#                     # Store the detection in the database
#                     store_detection(timestamp, category_name)
#
#             current_frame = frame
#             time.sleep(0.1)
#
#     detection_thread = threading.Thread(target=detect_vehicles)
#     detection_thread.daemon = True
#     detection_thread.start()
#
#     return jsonify({"message": "Detection started"}), 200
#
# @app.route("/live_data")
# def live_data():
#     # This route will provide real-time data via Server-Sent Events
#     def generate():
#         while True:
#             with lock:
#                 data = {
#                     "vehicle_count": vehicle_count,
#                     "counts": category_counts,
#                     "vehicles": [{"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "category": category} for category, count in category_counts.items() if count > 0]
#                 }
#             yield f"data: {json.dumps(data)}\n\n"
#             time.sleep(1)
#     return Response(generate(), content_type="text/event-stream")
#
# @app.route("/video_feed")
# def video_feed():
#     def generate_frames():
#         global current_frame
#         while True:
#             if current_frame is not None:
#                 ret, buffer = cv2.imencode('.jpg', current_frame)
#                 if ret:
#                     frame = buffer.tobytes()
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#             time.sleep(0.1)
#
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
# def store_detection(timestamp, category):
#     conn = sqlite3.connect(DB_FILE)
#     c = conn.cursor()
#     c.execute("INSERT INTO vehicle_log (timestamp, category) VALUES (?, ?)", (timestamp, category))
#     conn.commit()
#     conn.close()
#
# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000)
# from flask import Flask, render_template, request, Response, jsonify, stream_with_context
# import cv2
# import sqlite3
# import threading
# from ultralytics import YOLO
# from datetime import datetime
# import time
# import json
# import logging
# import torch
#
# # Initialize Flask app
# app = Flask(__name__)
#
# # Database setup
# DB_FILE = 'vehicle_data.db'
#
# def init_db():
#     """Initialize the database."""
#     conn = sqlite3.connect(DB_FILE)
#     c = conn.cursor()
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS vehicle_log (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             timestamp TEXT NOT NULL,
#             category TEXT NOT NULL
#         )
#     ''')
#     conn.commit()
#     conn.close()
#
# init_db()
#
# # YOLO model setup
# model = YOLO('yolov9c.pt')
# if torch.cuda.is_available():
#     model.to('cuda')
#     print("CUDA is available. Running on GPU.")
# else:
#     print("CUDA is not available. Running on CPU.")
#
# # Globals
# lock = threading.Lock()
# current_frame = None
# vehicle_categories = {
#     2: 'car',
#     3: 'motorbike',
#     5: 'three_wheeler',
#     7: 'van',
#     8: 'truck'
# }
# vehicle_count = 0
# category_counts = {key: 0 for key in vehicle_categories.values()}
#
# def store_detection(timestamp, category):
#     """Store detection in the database."""
#     conn = sqlite3.connect(DB_FILE)
#     c = conn.cursor()
#     c.execute("INSERT INTO vehicle_log (timestamp, category) VALUES (?, ?)", (timestamp, category))
#     conn.commit()
#     conn.close()
#
# @app.route("/")
# def home():
#     return render_template("index.html")
#
# @app.route("/start_detection", methods=["POST"])
# def start_detection():
#     global current_frame, vehicle_count, category_counts
#
#     input_type = request.form.get("input_type")
#     input_value = request.form.get("input_value")
#
#     if input_type not in ["video", "rtsp"] or not input_value:
#         return jsonify({"error": "Invalid input"}), 400
#
#     cap = cv2.VideoCapture(input_value)
#     if not cap.isOpened():
#         return jsonify({"error": "Unable to open video stream"}), 400
#
#     def detect_vehicles():
#         global current_frame, vehicle_count, category_counts
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             try:
#                 results = model(frame)
#                 detections = results[0].boxes
#
#                 category_counts = {key: 0 for key in category_counts}
#                 for box in detections:
#                     cls = int(box.cls)
#                     if cls in vehicle_categories:
#                         category_name = vehicle_categories[cls]
#                         category_counts[category_name] += 1
#                         vehicle_count += 1
#                         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#                         store_detection(timestamp, category_name)
#                 current_frame = frame
#             except Exception as e:
#                 logging.error(f"Error during detection: {e}")
#             time.sleep(0.1)
#
#     threading.Thread(target=detect_vehicles, daemon=True).start()
#     return jsonify({"message": "Detection started"}), 200
#
# @app.route("/live_data")
# def live_data():
#     def generate():
#         while True:
#             with lock:
#                 data = {
#                     "vehicle_count": vehicle_count,
#                     "counts": category_counts
#                 }
#             yield f"data: {json.dumps(data)}\n\n"
#             time.sleep(1)
#     return Response(generate(), content_type="text/event-stream")
#
# @app.route("/video_feed")
# def video_feed():
#     def generate_frames():
#         global current_frame
#         while True:
#             if current_frame is not None:
#                 ret, buffer = cv2.imencode('.jpg', current_frame)
#                 if ret:
#                     frame = buffer.tobytes()
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#             time.sleep(0.1)
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     app.run(debug=True, host="0.0.0.0", port=5000)
###########-----changes after adding  stop button----#########
# from flask import Flask, render_template, request, Response, jsonify, stream_with_context
# import cv2
# import sqlite3
# import threading
# from ultralytics import YOLO
# from datetime import datetime
# import time
# import json
# import logging
# import torch
#
# # Initialize Flask app
# app = Flask(__name__)
#
# # Database setup
# DB_FILE = 'vehicle_data.db'
#
# def init_db():
#     """Initialize the database."""
#     conn = sqlite3.connect(DB_FILE)
#     c = conn.cursor()
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS vehicle_log (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             timestamp TEXT NOT NULL,
#             category TEXT NOT NULL
#         )
#     ''')
#     conn.commit()
#     conn.close()
#
# init_db()
#
# # YOLO model setup
# model = YOLO('yolov9c.pt')
# if torch.cuda.is_available():
#     model.to('cuda')
#     print("CUDA is available. Running on GPU.")
# else:
#     print("CUDA is not available. Running on CPU.")
#
# # Globals
# lock = threading.Lock()
# current_frame = None
# vehicle_categories = {
#     2: 'car',
#     3: 'motorbike',
#     5: 'three_wheeler',
#     7: 'van',
#     8: 'truck'
# }
# vehicle_count = 0
# category_counts = {key: 0 for key in vehicle_categories.values()}
# is_detection_running = False  # To control the detection process
# detection_thread = None       # Thread for the detection process
# cap = None                    # VideoCapture object
#
# def store_detection(timestamp, category):
#     """Store detection in the database."""
#     conn = sqlite3.connect(DB_FILE)
#     c = conn.cursor()
#     c.execute("INSERT INTO vehicle_log (timestamp, category) VALUES (?, ?)", (timestamp, category))
#     conn.commit()
#     conn.close()
#
# @app.route("/")
# def home():
#     return render_template("index.html")
#
# @app.route("/start_detection", methods=["POST"])
# def start_detection():
#     global current_frame, vehicle_count, category_counts, is_detection_running, detection_thread, cap
#
#     input_type = request.json.get("input_type")
#     input_value = request.json.get("input_value")
#
#     if input_type not in ["video", "rtsp"] or not input_value:
#         return jsonify({"error": "Invalid input"}), 400
#
#     # Open video capture
#     cap = cv2.VideoCapture(input_value)
#     if not cap.isOpened():
#         return jsonify({"error": "Unable to open video stream"}), 400
#
#     is_detection_running = True
#
#     def detect_vehicles():
#         global current_frame, vehicle_count, category_counts, is_detection_running, cap
#
#         # Define unique colors for each vehicle category
#         category_colors = {
#             'car': (0, 255, 0),  # Green
#             'motorbike': (255, 0, 0),  # Blue
#             'three_wheeler': (0, 0, 255),  # Red
#             'van': (255, 255, 0),  # Cyan
#             'truck': (255, 0, 255)  # Magenta
#         }
#
#         # Define the line position for crossing detection (y-coordinate)
#         line_y = 300  # Adjust this based on the video resolution
#         crossing_vehicles = set()  # To track already counted vehicles
#
#         while is_detection_running and cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             try:
#                 # Run YOLO model on the current frame
#                 results = model(frame)
#                 detections = results[0].boxes
#
#                 # Reset category counts
#                 category_counts = {key: 0 for key in category_counts}
#
#                 # Draw the line for crossing detection
#                 line_color = (0, 255, 255)  # Yellow
#                 line_thickness = 2
#                 cv2.line(frame, (0, line_y), (frame.shape[1], line_y), line_color, line_thickness)
#
#                 # Loop through detections and process each bounding box
#                 for box in detections:
#                     cls = int(box.cls)
#                     if cls in vehicle_categories:
#                         category_name = vehicle_categories[cls]
#
#                         # Extract bounding box coordinates
#                         x1, y1, x2, y2 = map(int, box.xyxy[0])  # x1, y1: top-left; x2, y2: bottom-right
#                         box_center = (x1 + x2) // 2, y2  # Bottom center of the bounding box
#
#                         # Get the color for the current category
#                         color = category_colors.get(category_name, (255, 255, 255))  # Default to white if not found
#
#                         # Draw bounding box
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#
#                         # Put category label near the bounding box
#                         label = f"{category_name}"
#                         font = cv2.FONT_HERSHEY_SIMPLEX
#                         font_scale = 0.5
#                         font_color = color
#                         thickness = 2
#                         cv2.putText(frame, label, (x1, y1 - 10), font, font_scale, font_color, thickness)
#
#                         # Check if the vehicle has crossed the line
#                         if box_center[1] > line_y:  # Bottom edge crosses the line
#                             vehicle_id = (x1, y1, x2, y2)  # Use bounding box as a temporary ID
#                             if vehicle_id not in crossing_vehicles:
#                                 crossing_vehicles.add(vehicle_id)  # Mark this vehicle as counted
#                                 category_counts[category_name] += 1
#                                 vehicle_count += 1
#                                 timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#                                 store_detection(timestamp, category_name)
#
#                 # Update the global frame to include bounding boxes and line
#                 current_frame = frame
#             except Exception as e:
#                 logging.error(f"Error during detection: {e}")
#             time.sleep(0.1)
#
#         # Release resources when stopping
#         if cap.isOpened():
#             cap.release()
#         is_detection_running = False
#
#     # Start the detection thread
#     detection_thread = threading.Thread(target=detect_vehicles)
#     detection_thread.daemon = True
#     detection_thread.start()
#
#     return jsonify({"message": "Detection started"}), 200
#
# @app.route("/stop_detection", methods=["POST"])
# def stop_detection():
#     global is_detection_running, cap
#
#     if not is_detection_running:
#         return jsonify({"error": "Detection is not running"}), 400
#
#     is_detection_running = False  # Stop the detection process
#
#     # Wait for resources to be cleaned up
#     if cap and cap.isOpened():
#         cap.release()
#
#     return jsonify({"message": "Detection stopped"}), 200
#
# @app.route("/live_data")
# def live_data():
#     def generate():
#         while True:
#             with lock:
#                 data = {
#                     "vehicle_count": vehicle_count,
#                     "counts": category_counts
#                 }
#             yield f"data: {json.dumps(data)}\n\n"
#             time.sleep(1)
#     return Response(generate(), content_type="text/event-stream")
#
# @app.route("/video_feed")
# def video_feed():
#     def generate_frames():
#         global current_frame
#         while True:
#             if current_frame is not None:
#                 ret, buffer = cv2.imencode('.jpg', current_frame)
#                 if ret:
#                     frame = buffer.tobytes()
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#             time.sleep(0.1)
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     app.run(debug=True, host="0.0.0.0", port=5000)
########--------changes after adding customized horizontal line and automatic lane detection----#####
# from flask import Flask, render_template, request, Response, jsonify
# import cv2
# import sqlite3
# import threading
# from ultralytics import YOLO
# from datetime import datetime
# import time
# import json
# import logging
# import torch
# import numpy as np
#
#
# # Initialize Flask app
# app = Flask(__name__)
#
# # Database setup
# DB_FILE = 'vehicle_data.db'
#
# def init_db():
#     """Initialize the database."""
#     conn = sqlite3.connect(DB_FILE)
#     c = conn.cursor()
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS vehicle_log (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             timestamp TEXT NOT NULL,
#             category TEXT NOT NULL
#         )
#     ''')
#     conn.commit()
#     conn.close()
#
# init_db()
#
# # YOLO model setup
# model = YOLO('yolov9c.pt')
# if torch.cuda.is_available():
#     model.to('cuda')
#     print("CUDA is available. Running on GPU.")
# else:
#     print("CUDA is not available. Running on CPU.")
#
# # Globals
# lock = threading.Lock()
# current_frame = None
# vehicle_categories = {
#     2: 'car',
#     3: 'motorbike',
#     5: 'bus',
#     7: 'van',
#     8: 'truck'
# }
# vehicle_count = 0
# category_counts = {key: 0 for key in vehicle_categories.values()}
# is_detection_running = False  # To control the detection process
# detection_thread = None       # Thread for the detection process
# cap = None                    # VideoCapture object
#
# def store_detection(timestamp, category):
#     """Store detection in the database."""
#     conn = sqlite3.connect(DB_FILE)
#     c = conn.cursor()
#     c.execute("INSERT INTO vehicle_log (timestamp, category) VALUES (?, ?)", (timestamp, category))
#     conn.commit()
#     conn.close()
#
# @app.route("/")
# def home():
#     """Serve the homepage."""
#     return render_template("index.html")
#
# @app.route("/start_detection", methods=["POST"])
# def start_detection():
#     """Start vehicle detection."""
#     global current_frame, vehicle_count, category_counts, is_detection_running, detection_thread, cap
#
#     input_type = request.json.get("input_type")
#     input_value = request.json.get("input_value")
#
#     if input_type not in ["video", "rtsp"] or not input_value:
#         return jsonify({"error": "Invalid input"}), 400
#
#     # Open video capture
#     cap = cv2.VideoCapture(input_value)
#     if not cap.isOpened():
#         return jsonify({"error": "Unable to open video stream"}), 400
#
#     is_detection_running = True
#
#     def detect_vehicles():
#         global current_frame, vehicle_count, category_counts, is_detection_running, cap
#
#         # Define unique colors for each vehicle category
#         category_colors = {
#             'car': (0, 255, 0),  # Green
#             'motorbike': (255, 0, 0),  # Blue
#             'bus': (0, 0, 255),  # Red
#             'van': (255, 255, 0),  # Cyan
#             'truck': (255, 0, 255)  # Magenta
#         }
#
#         # Define the line position for crossing detection (y-coordinate)
#         line_y = 300  # Adjust this based on the video resolution
#         crossing_vehicles = set()  # To track already counted vehicles
#
#         while is_detection_running and cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             try:
#                 # Run YOLO model on the current frame
#                 results = model(frame)
#                 detections = results[0].boxes
#
#                 # Reset category counts
#                 category_counts = {key: 0 for key in category_counts}
#
#                 # Draw the line for crossing detection
#                 line_color = (0, 255, 255)  # Yellow
#                 line_thickness = 2
#                 cv2.line(frame, (0, line_y), (frame.shape[1], line_y), line_color, line_thickness)
#
#                 # Loop through detections and process each bounding box
#                 for box in detections:
#                     cls = int(box.cls)
#                     if cls in vehicle_categories:
#                         category_name = vehicle_categories[cls]
#
#                         # Extract bounding box coordinates
#                         x1, y1, x2, y2 = map(int, box.xyxy[0])  # x1, y1: top-left; x2, y2: bottom-right
#                         box_center = (x1 + x2) // 2, y2  # Bottom center of the bounding box
#
#                         # Get the color for the current category
#                         color = category_colors.get(category_name, (255, 255, 255))  # Default to white if not found
#
#                         # Draw bounding box
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#
#                         # Put category label near the bounding box
#                         label = f"{category_name}"
#                         font = cv2.FONT_HERSHEY_SIMPLEX
#                         font_scale = 0.5
#                         font_color = color
#                         thickness = 2
#                         cv2.putText(frame, label, (x1, y1 - 10), font, font_scale, font_color, thickness)
#
#                         # Check if the vehicle has crossed the line
#                         if box_center[1] > line_y:  # Bottom edge crosses the line
#                             vehicle_id = (x1, y1, x2, y2)  # Use bounding box as a temporary ID
#                             if vehicle_id not in crossing_vehicles:
#                                 crossing_vehicles.add(vehicle_id)  # Mark this vehicle as counted
#                                 category_counts[category_name] += 1
#                                 vehicle_count += 1
#                                 timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#                                 store_detection(timestamp, category_name)
#
#                 # Update the global frame to include bounding boxes and line
#                 current_frame = frame
#             except Exception as e:
#                 logging.error(f"Error during detection: {e}")
#             time.sleep(0.1)
#
#         # Release resources when stopping
#         if cap.isOpened():
#             cap.release()
#         is_detection_running = False
#
#         def detect_road_and_calculate_margin(frame, fallback_line_y):
#             """Detect road boundaries and calculate a crossing margin dynamically."""
#             try:
#                 # Convert the frame to grayscale
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#                 # Apply Canny edge detection
#                 edges = cv2.Canny(gray, 50, 150)
#
#                 # Use Hough Line Transform to detect lines
#                 lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=200)
#                 if lines is not None:
#                     # Identify the average y-coordinates of the detected lines
#                     lane_y_coords = [line[0][1] for line in lines] + [line[0][3] for line in lines]
#                     road_y = int(np.mean(lane_y_coords))  # Average y-coordinate of road lanes
#
#                     # Calculate a crossing margin proportional to the road's size
#                     margin = int(frame.shape[0] * 0.1)  # 10% of the frame height as margin
#                     crossing_line_y = road_y + margin
#
#                     return crossing_line_y
#             except Exception as e:
#                 logging.error(f"Road detection failed: {e}")
#                 return fallback_line_y  # Return fallback value if detection fails
#
#     # Start the detection thread
#     detection_thread = threading.Thread(target=detect_vehicles)
#     detection_thread.daemon = True
#     detection_thread.start()
#
#     return jsonify({"message": "Detection started"}), 200
#
# @app.route("/stop_detection", methods=["POST"])
# def stop_detection():
#     """Stop vehicle detection."""
#     global is_detection_running, cap
#
#     if not is_detection_running:
#         return jsonify({"error": "Detection is not running"}), 400
#
#     is_detection_running = False  # Stop the detection process
#
#     # Wait for resources to be cleaned up
#     if cap and cap.isOpened():
#         cap.release()
#
#     return jsonify({"message": "Detection stopped"}), 200
#
# @app.route("/live_data")
# def live_data():
#     """Stream live data about vehicles."""
#     def generate():
#         while True:
#             with lock:
#                 data = {
#                     "vehicle_count": vehicle_count,
#                     "counts": category_counts
#                 }
#             yield f"data: {json.dumps(data)}\n\n"
#             time.sleep(1)
#     return Response(generate(), content_type="text/event-stream")
#
# @app.route("/video_feed")
# def video_feed():
#     """Stream the video feed with bounding boxes."""
#     def generate_frames():
#         global current_frame
#         while True:
#             if current_frame is not None:
#                 ret, buffer = cv2.imencode('.jpg', current_frame)
#                 if ret:
#                     frame = buffer.tobytes()
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#             time.sleep(0.1)
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     app.run(debug=True, host="0.0.0.0", port=5000)
# #########---------Updated Python Code for Proportional Crossing Line

import csv
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from io import StringIO
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, Response, jsonify, make_response
from ultralytics import YOLO


# Initialize Flask app
app = Flask(__name__)

# Database setup
DB_FILE = 'vehicle_data.db'

def init_db():
    """Initialize the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            category TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# YOLO model setup
model = YOLO('yolov8l.pt')  # Use the correct YOLOv model path
if torch.cuda.is_available():
    model.to('cuda')
    print("CUDA is available. Running on GPU.")
else:
    print("CUDA is not available. Running on CPU.")

# Globals
lock = threading.Lock()
current_frame = None
vehicle_categories = {
    2: 'car',
    3: 'motorbike',
    5: 'bus',
    7: 'van',
    8: 'truck'
}
vehicle_count = 0
category_counts = {key: 0 for key in vehicle_categories.values()}
is_detection_running = False  # To control the detection process
detection_thread = None       # Thread for the detection process
cap = None                    # VideoCapture object

# Set to track vehicles already counted
crossing_vehicles = set()


def store_detection(timestamp, category):
    """Store detection in the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO vehicle_log (timestamp, category) VALUES (?, ?)", (timestamp, category))
    conn.commit()
    conn.close()


@app.route("/")
def home():
    """Serve the homepage."""
    return render_template("index.html")

@app.route('/download_csv', methods=['POST'])
def download_csv():
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    # Fetch data from the database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    query = """
        SELECT timestamp, category
        FROM vehicle_log
        WHERE timestamp BETWEEN ? AND ?
    """
    cursor.execute(query, (start_date, end_date))
    rows = cursor.fetchall()
    conn.close()

    # Generate CSV file in memory
    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(['Timestamp', 'Category'])  # CSV headers
    writer.writerows(rows)
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=vehicle_data_{start_date}_to_{end_date}.csv"
    output.headers["Content-type"] = "text/csv"
    return output


@app.route("/start_detection", methods=["POST"])
def start_detection():
    """Start vehicle detection."""
    global current_frame, vehicle_count, category_counts, is_detection_running, detection_thread, cap, crossing_vehicles

    input_type = request.json.get("input_type")
    input_value = request.json.get("input_value")

    if input_type not in ["video", "rtsp"] or not input_value:
        return jsonify({"error": "Invalid input"}), 400

    cap = cv2.VideoCapture(input_value)
    if not cap.isOpened():
        return jsonify({"error": "Unable to open video stream"}), 400

    is_detection_running = True
    crossing_vehicles.clear()  # Clear the set when starting detection

    def detect_road_and_calculate_margin(frame, fallback_line_y):
        """Detect road boundaries and calculate a crossing margin dynamically."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=200)

            if lines is not None:
                lane_y_coords = [line[0][1] for line in lines] + [line[0][3] for line in lines]
                road_y = int(np.mean(lane_y_coords))  # Average y-coordinate of road lanes
                margin = int(frame.shape[0] * 0.1)  # 10% of frame height as margin
                return road_y + margin
        except Exception as e:
            logging.error(f"Road detection failed: {e}")
        return fallback_line_y

    def detect_vehicles():
        global current_frame, vehicle_count, category_counts, is_detection_running, cap

        # Line position and buffer zone
        fallback_line_y = 300
        buffer_zone = 10

        # Category color mapping
        category_colors = {
            'car': (0, 255, 0),  # Green
            'motorbike': (255, 0, 0),  # Blue
            'bus': (0, 0, 255),  # Red
            'van': (255, 255, 0),  # Cyan
            'truck': (255, 0, 255)  # Magenta
        }

        while is_detection_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # Calculate the dynamic crossing line position
                left_edge, right_edge = 0, frame.shape[1]  # Assume full width as fallback
                line_y = detect_road_and_calculate_margin(frame, fallback_line_y)

                # Run YOLO detection
                results = model(frame)
                detections = results[0].boxes

                for box in detections:
                    cls = int(box.cls)
                    if cls in vehicle_categories:
                        category_name = vehicle_categories[cls]

                        # Extract bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        box_center = (x1 + x2) // 2, y2  # Bottom-center of the bounding box

                        # Vehicle unique ID
                        vehicle_id = (x1, y1, x2, y2)

                        # Check if vehicle crosses the line
                        if line_y < box_center[1] < line_y + buffer_zone:
                            if vehicle_id not in crossing_vehicles:
                                crossing_vehicles.add(vehicle_id)
                                category_counts[category_name] += 1
                                vehicle_count += 1
                                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                store_detection(timestamp, category_name)

                        # Draw bounding box and label
                        color = category_colors.get(category_name, (255, 255, 255))  # Default to white
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, category_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw the crossing line
                cv2.line(frame, (left_edge, line_y), (right_edge, line_y), (0, 255, 255), 2)

                # Update the global frame for video streaming
                current_frame = frame

            except Exception as e:
                logging.error(f"Error during detection: {e}")

            time.sleep(0.1)

        if cap.isOpened():
            cap.release()
        is_detection_running = False

    detection_thread = threading.Thread(target=detect_vehicles)
    detection_thread.daemon = True
    detection_thread.start()

    return jsonify({"message": "Detection started"}), 200


@app.route("/stop_detection", methods=["POST"])
def stop_detection():
    """Stop vehicle detection."""
    global is_detection_running, cap

    if not is_detection_running:
        return jsonify({"error": "Detection is not running"}), 400

    is_detection_running = False
    if cap and cap.isOpened():
        cap.release()

    return jsonify({"message": "Detection stopped"}), 200


@app.route("/live_data")
def live_data():
    """Stream live data about vehicles."""
    def generate():
        while True:
            with lock:
                data = {
                    "vehicle_count": vehicle_count,
                    "counts": category_counts
                }
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(1)
    return Response(generate(), content_type="text/event-stream")


@app.route("/video_feed")
def video_feed():
    """Stream the video feed with bounding boxes."""
    def generate_frames():
        global current_frame
        while True:
            if current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', current_frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            time.sleep(0.1)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host="0.0.0.0", port=5000)














