from collections import defaultdict
from moviepy.editor import *

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.solutions import heatmap

from flask import Flask, render_template_string, request, session
from flask import send_from_directory
import shutil
import os
from pathlib import Path

x = 0
y = 0
print(Path.cwd())
app = Flask(__name__, static_url_path='/static')
# HTML Code
TPL = '''
<!doctype html>
<html lang="en">
<head>
	<meta charset="utf-8">
	<meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">

	<link rel="shortcut icon" href="https://ih0.redbubble.net/image.4636513578.7670/raf,360x360,075,t,fafafa:ca443f4786.jpg" type="image/x-icon">
	<link rel="icon" href="https://ih0.redbubble.net/image.4636513578.7670/raf,360x360,075,t,fafafa:ca443f4786.jpg" type="image/x-icon">

	<title>Web Page Controlled Servo Crowd Management System</title>

	<link rel="preconnect" href="https://fonts.googleapis.com">
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
	<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
	<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" />

    <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='css/style.css') }}"> 
	<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.1/jquery.min.js"></script>
</head>
<body>
	<div class="nav" role="nav">
		<div class="nav-top">
			<svg class="logo" id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 452.38 105.4">
				<path d="M73.03,105.27C12.78,105.27,0,82.55,0,59.16c0-4.62.41-9.11.82-11.7C3.81,28.15,14.96,0,81.33,0h4.62c67.86,0,73.3,29.65,71.26,42.43l-.41,2.58h-50.86c-.14-2.31-1.9-12.78-24.21-12.78s-27.61,10.61-28.97,17.41c-.41,2.04-.68,4.22-.68,6.12,0,8.98,5.17,17,24.34,17,22.85,0,26.25-11.42,27.06-13.87h51.14l-.27,1.77c-2.18,13.74-8.43,44.61-76.71,44.61h-4.62Z" fill="#23489b" stroke-width="0"/>
				<path d="M148.77,2.04h65.69l17,45.02,30.06-45.02h64.19l-16.05,101.19h-50.19l7.89-50.05-33.59,50.05h-24.07l-18.9-50.05-7.89,50.05h-50.18L148.77,2.04Z" fill="#2f60ce" stroke-width="0"/>
				<path d="M298.37,68.82h59.16c0,4.35.54,10.88,14.82,10.88,13.19,0,15.64-3.94,15.64-7.21,0-3.94-2.04-6.53-18.5-7.48l-16.73-.95c-37.26-2.18-46.1-15.64-46.1-31.28,0-2.04.41-4.9.81-6.66C311.56,8.3,329.65.27,374.53.27h16.87c57.94,0,62.15,18.77,60.79,32.5h-58.07c0-3.67-1.09-8.16-14.28-8.16-11.83,0-15.1,1.63-15.1,5.3s3.13,5.44,18.63,6.26l16.59.82c36.99,1.77,49.5,10.2,49.5,28.56,0,2.58-.14,5.17-.41,7.07-2.45,18.09-15.64,32.64-71.4,32.64h-17.41c-60.38,0-63.24-23.8-61.88-36.45Z" fill="#05e3ab" stroke-width="0"/>
			  </svg>
			  <h1 style="display: none;">Crowd Monitoring System</h1>
		</div>
		<hr>
		<div class="nav-middle">
			<div class="upload-video-form">
				<h2>Upload Video File</h2>
				<form method="POST" action="/process_video" enctype="multipart/form-data">
					<div class="switches-container">
						<input type="radio" id="switchObject" name="switchType" value="Object" checked="checked" />
						<input type="radio" id="switchHeatmap" name="switchType" value="Heatmap" />
						<label for="switchObject">Object Detection</label>
						<label for="switchHeatmap">Heatmap</label>
						<div class="switch-wrapper">
							<div class="switch">
							<div>Object Counting</div>
							<div>Heatmap</div>
							</div>
						</div>
						</div>
					<label for="file-upload" class="custom-file-upload choose-file">
						<div class="custom-file-button">
							<span class="material-symbols-outlined icon">
								add
							</span>
							Choose Video
						</div>
					</label>
					<input id="file-upload" name='video_file' type="file" accept=".mp4, .mov, .avi, .asf, .m4v" style="display:none;">

					<div class="slidecontainer">
						<p><span class="bold">Confidence Level Value:</span> <span id="demo"></span> <span class="material-symbols-outlined" data-tooltip="Confidence level is a measure of percentage on how sure the model is about its predictions. A high confidence score means the model is very confident, while a low confidence score means it's less certain." data-tooltip-location="top">
							help
							</span></p>
						<input type="range" min="0" max="100" value="25" class="slider" id="myRange" name="slider">
					</div>

					<label for="upload" class="custom-upload-button choose-file">
						<div class="custom-file-button">
							<span class="material-symbols-outlined">
								upload
							</span>
							Upload
						</div>
					</label>
					<input id="upload" type="submit" name="action" value="Upload" style="display:none;" />
					<script>
						$('#file-upload').change(function() {
							var i = $(this).prev('label').clone();
							var file = $('#file-upload')[0].files[0].name;
							$(this).prev('label').text(file);
						});
						var slider = document.getElementById("myRange");
						var output = document.getElementById("demo");
						output.innerHTML = slider.value / 100; // Initialize the value to 0.25

						slider.oninput = function() {
							output.innerHTML = (this.value / 100).toFixed(2); // Adjust the value range to 0.00 to 1.00
						}
					</script>
					<div class="or-text">
				<hr>
				<h4>or</h4>
			</div>

			<div class="upload-live-form">
				<h2>Record Live Video</h2>
				    <div class="slidecontainer">
						<p><span class="bold">Confidence Level Value:</span> <span id="demo1"></span> <span class="material-symbols-outlined" data-tooltip="Confidence level is a measure of percentage on how sure the model is about its predictions. A high confidence score means the model is very confident, while a low confidence score means it's less certain." data-tooltip-location="top">
							help
							</span></p>
						<input type="range" min="0" max="100" value="25" class="slider" id="myRange1" name="slider1">
					</div>

					<label for="live" class="custom-upload-button choose-file">
						<div class="custom-file-button">
							<span class="material-symbols-outlined">
								radio_button_checked
							</span>
							Record
						</div>
					</label>
					<input id="live" type="submit" name="action" value="Live" style="display:none;">
					<script>
						var slider1 = document.getElementById("myRange1");
						var output1 = document.getElementById("demo1");
						output1.innerHTML = slider1.value / 100; // Initialize the value to 0.25

						slider1.oninput = function() {
							output1.innerHTML = (this.value / 100).toFixed(2); // Adjust the value range to 0.00 to 1.00
						}
					</script>
			</div>
				</form>
			</div>


			<hr>

			<div class="uploaded-video">
				{% if video_path %}
				<p>Uploaded Video:</p>
				<video class="small-video" controls>
					<source src="{{ video_path }}" type="video/mp4">
					Your browser does not support the video tag.
				</video>
			    {% endif %}
			</div>

	    </div>
		<a href="" class="nav-bottom">
			<span class="material-symbols-outlined icon">
				info
			</span>
			<h4 class="icon-text">How it works</h4>
		</a>
	</div>
	<div class="main" role="main">
		{% if processed_video_path %}
			<div class="video-type">
				Showing: <span class="color-bold">Processed Video</span>
			</div>
			<video class="main-video" controls>
				<source src="{{ processed_video_path }}" type="video/mp4">
				Your browser does not support the video tag.
			</video>
		{% endif %}
	</div>
</body>
</html>

'''
track_history = defaultdict(list)


def runVid(pather, confidence):
    # Setup Model
    vid_frame_count = 0

    # Setup Model
    model = YOLO('yolov8n.pt')
    model.to("cpu")
    slider_value = confidence
    print(confidence)
    if slider_value is None:
        # Handle case where slider value is missing
        conf = 0.25
        print("error")
    else:
        try:
            # Parse slider value as float and convert to range 0.00 to 1.00
            conf = float(slider_value) / 100
        except ValueError:
            # Handle case where slider value is invalid
            conf = 0.25
    print(conf)

    # Extract classes names
    names = model.model.names

    # Video setup
    videocapture = cv2.VideoCapture(pather)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    save_dir = increment_path(Path("detect"))
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{'Finalresults'}.mp4"), fourcc, fps, (frame_width, frame_height))
    counting_regions = [

        {
            "name": "YOLOv8 Rectangle Region",
            "polygon": Polygon([(0, 0), (frame_width, 0), (frame_width, frame_height), (0, frame_height)]),
            # Polygon points
            "counts": 0,
            "dragging": False,
            "region_color": (37, 255, 225),  # BGR Value
            "text_color": (0, 0, 0),  # Region Text Color
        },
    ]

    # Iterate over video frames
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # Extract the results
        results = model.track(frame, persist=True, classes=0, conf=conf)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=2, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

                track = track_history[track_id]  # Tracking Lines plot
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=2)

                # Check if detection inside region
                for region in counting_regions:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1

        # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            cv2.putText(
                frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, 2
            )
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=4)

        cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)

        video_writer.write(frame)

        for region in counting_regions:  # Reinitialize count for each region
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


def resize(path, size, final_path):
    video = VideoFileClip(path)
    # getting width and height of video 1
    width_of_video1 = video.w
    height_of_video1 = video.h
    print("Width and Height of original video : ", end=" ")
    print(str(width_of_video1) + " x ", str(height_of_video1))
    print("#################################")
    # compressing
    video_resized = video.resize(size)
    # getting width and height of video 2 which is resized
    width_of_video2 = video_resized.w
    height_of_video2 = video_resized.h
    print("Width and Height of resized video : ", end=" ")
    print(str(width_of_video2) + " x ", str(height_of_video2))
    print("###################################")
    # displaying final clip
    video_resized.write_videofile(final_path)


def heatmaper(pather, confidence):
    slider_value = confidence
    print(confidence)
    if slider_value is None:
        # Handle case where slider value is missing
        conf = 0.25
        print("error")
    else:
        try:
            # Parse slider value as float and convert to range 0.00 to 1.00
            conf = float(slider_value) / 100
        except ValueError:
            # Handle case where slider value is invalid
            conf = 0.25
    print(conf)

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(pather)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Video writer
    video_writer = cv2.VideoWriter("detect/Finalresults.mp4",
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps,
                                   (w, h))

    # Init heatmap
    heatmap_obj = heatmap.Heatmap()
    heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                         imw=w,
                         imh=h,
                         view_img=True,
                         shape="circle")

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        tracks = model.track(im0, persist=True, show=False, conf=conf, classes=0)

        im0 = heatmap_obj.generate_heatmap(im0, tracks)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


@app.route("/")
def home():
    return render_template_string(TPL)


@app.route("/process_video", methods=["POST"])
def process_video():
    global x, y
    action = request.form.get('action')

    if action == 'Upload':
        # Handle processing video action
        # Check if the POST request has the file part
        if 'video_file' not in request.files:
            return "No file part"

        file = request.files['video_file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return "No selected file"

        # Check if the file is allowed
        allowed_extensions = {'mp4', 'mov', 'avi', 'asf', 'm4v'}
        if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return "Invalid file type"

        # Save the file to the 'uploads' folder
        if os.path.exists('uploads'):
            shutil.rmtree('uploads')
            os.makedirs('uploads')
            print('it worked')
        else:
            os.makedirs('uploads')

        if os.path.exists('detect'):
            shutil.rmtree('detect')

        video_path = 'uploads/' + 'uploaded.mp4'
        file.save(video_path)
        print(request.form.get('switchType'))
        if request.form.get('switchType') == 'Heatmap':
            os.makedirs('detect')
            heatmaper(video_path, request.form.get('slider'))
        elif request.form.get('switchType') == 'Object':
            runVid(video_path, request.form.get('slider'))

        resize('detect/Finalresults.mp4', 0.5, 'detect/CFinalresults.mp4')

        if x == 0:
            processed_video_path = 'detect/CFinalresults.mp4'
            video_url = '/uploads/uploaded.mp4'
            x = 1
        elif x == 1:
            processed_video_path = 'detect/1CFinalresults.mp4'
            video_url = '/uploads/1uploaded.mp4'
            x = 0
        return render_template_string(TPL, video_path=video_url, processed_video_path=processed_video_path)
    elif action == 'Live':
        # Handle live video action
        if os.path.exists('detect'):
            shutil.rmtree('detect')
        print(request.form.get('switchType'))
        if request.form.get('switchType') == 'Heatmap':
            print(request.form.get('switchType'))
            os.makedirs('detect')
            heatmaper(0, request.form.get('slider1'))
        elif request.form.get('switchType') == 'Object':
            print(request.form.get('switchType'))
            runVid(0, request.form.get('slider1'))
        resize('detect/Finalresults.mp4', 0.5, 'detect/CFinalresults.mp4')

        if y == 0:
            processed_video_path = 'detect/CFinalresults.mp4'
            y = 1
        elif y == 1:
            processed_video_path = 'detect/1CFinalresults.mp4'
            y = 0

        return render_template_string(TPL, processed_video_path=processed_video_path)
    else:
        # Handle invalid action
        return "Invalid action specified"


@app.route('/uploads/uploaded.mp4')
def uploaded_file():
    return send_from_directory('uploads', 'uploaded.mp4')


@app.route('/uploads/1uploaded.mp4')
def uploaded_file1():
    return send_from_directory('uploads', 'uploaded.mp4')


@app.route('/detect/CFinalresults.mp4')
def serve_processed_video():
    filename = 'CFinalresults.mp4'  # Specify the name of the processed video file
    path = 'detect'
    return send_from_directory(path, filename)


@app.route('/detect/1CFinalresults.mp4')
def serve_processed_video1():
    filename = 'CFinalresults.mp4'  # Specify the name of the processed video file
    path = 'detect'
    return send_from_directory(path, filename)


# Run the app on the local development server
if __name__ == "__main__":
    app.run()
