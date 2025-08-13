import cv2
from ultralytics import YOLO
import cvzone
import math
from sort import *
# Load the YOLOv8 model
model = YOLO("yolov8m.pt")

# Class names
Class1 = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Video file path
video_path = r"R:\cv_counter_project\files\cars.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Try to set the frame dimensions. Note: For video files, these might be ignored,
# as the video has its own inherent resolution. It's more reliable to read the
# actual frame size.
cap_width = 1080
cap_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

# --- IMPORTANT: Get actual frame dimensions after cap.set() (or first read) ---
# Read one frame to get its actual dimensions
success, initial_frame = cap.read()
if not success:
    print("Error: Could not read initial frame to determine dimensions.")
    exit()

frame_height, frame_width = initial_frame.shape[0], initial_frame.shape[1]
print(f"Actual video frame dimensions: {frame_width}x{frame_height}")

# Load the mask image. Assuming it's a binary mask (black/white) for ROI,
# load it as grayscale to ensure it's 1-channel.
mask_path = r"R:\cv_counter_project\files\mask.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Load as grayscale (1 channel)
if mask is None:
    print(f"Error: Could not load mask image from {mask_path}")
    exit()

# Resize the mask to match the video frame dimensions
mask = cv2.resize(mask, (frame_width, frame_height))
print(f"Resized mask dimensions: {mask.shape[1]}x{mask.shape[0]} (Channels: {mask.ndim})")

#Tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)  # Initialize the SORT tracker

# Reset video capture to the beginning if you read an initial frame for dimensions
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

CONFIDENCE_THRESHOLD=0.5  # Confidence threshold for filtering detections

limits1=[423,297,673,297]
limits=[150,400,720,400]
total_count = []  # Initialize total count of detected vehicles
while True:
    success, img = cap.read()  # Read a frame from the camera
    if not success:
        print("End of video stream or failed to read frame.")
        break # Exit loop if no more frames or error

    # Ensure the frame is the expected size, in case cap.set() didn't work perfectly
    if img.shape[0] != frame_height or img.shape[1] != frame_width:
        img = cv2.resize(img, (frame_width, frame_height))

    # Apply the mask. The 'mask' argument of bitwise_and expects a single-channel mask.
    # The output 'imRegion' will have the same dimensions and channels as 'img'.
    imRegion = cv2.bitwise_and(img, img, mask=mask)

    # Perform inference on the original frame (img) if you want to detect objects
    # everywhere, then display a masked version. If you only want to detect
    # within the masked region, use `imRegion` here.
    results = model(imRegion, stream=True) #stream=True for faster processing
    detections = np.empty((0, 5)) # Initialize an empty array for detections

    for r in results:  # Iterate through the results
        boxes = r.boxes  # Get the bounding boxes from the results
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the coordinates of the bounding box
            w, h = x2 - x1, y2 - y1

            # confidence score
            conf = math.ceil((box.conf[0] * 100)) / 100

            # class name
            cls = int(box.cls[0])

            # --- Filter Detections ---
            # 1. Apply confidence threshold
            if conf < CONFIDENCE_THRESHOLD:
                continue # Skip drawing if confidence is too low

            currentClass = Class1[cls]  # Get the class name from the class index

            # Check if the detected class is in your defined list and draw
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass== "motorcycle" or currentClass=="bicycle" and conf > 0.4:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                   scale=1, thickness=2, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=5, rt=2)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
 
    resultsTracker = tracker.update(detections) 

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 4)  # Draw the line

    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        print(result)
        w,h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=5, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)),
                                   scale=2, thickness=3, offset=10)
        cx,cy=x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 4, (0, 255, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1]-20< cy < limits[3]+20:
            if id not in total_count:
                total_count.append(id)  # Add the ID to the total count list
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255), 4)  # Draw the line


    cvzone.putTextRect(img, f'Count: {len(total_count)}', (50, 50), scale=2, thickness=3, offset=10)   
    cv2.imshow("Original Frame with Detections", img)  # Display the frame with detections
    # cv2.imshow("Masked Region", imRegion)  # Display the masked region

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
