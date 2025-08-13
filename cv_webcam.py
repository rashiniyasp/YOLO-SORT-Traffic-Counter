import cv2
from ultralytics import YOLO
import cvzone
import math


Class1= [
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

model = YOLO("yolov8l.pt")  # Load the YOLOv8 model
cap= cv2.VideoCapture('cars.mp4')  # Use 0 for the default camera, for video file use 'video.mp4'

# cap.set(10, 150)  # Set the brightness of the camera
# cap.set(2, 150)   # Set the contrast of the camera
cap.set(3, 1080)  # Set the width of the video frame
cap.set(4, 720)   # Set the height of the video frame

while True:
    success,img=cap.read()  # Read a frame from the camera
    
    cv2.waitKey(1)  # Wait for a key press for 1 ms
    results = model(img, stream=True)# Perform inference on the frame
    for r in results:  # Iterate through the results
        boxes = r.boxes  # Get the bounding boxes from the results
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates of the bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
            # cv2.rectangle(img, (x1, y1), (w, h), (255, 0, 0), 3)  # Draw the bounding box on the frame      
            w,h= x2-x1, y2-y1     
            cvzone.cornerRect(img, (x1, y1, w, h), l=15, t=3, colorR=(255, 0, 0), colorC=(0, 255, 0))  # Draw a rectangle with rounded corners
            # confidence score
            conf=math.ceil((box.conf[0] * 100))/100  # Get the confidence score and round it
            # cvzone.putTextRect(img, f'Conf: {conf}', (max(0,x1),max(35,y1)))
            # class name
            cls = int(box.cls[0])
            # Check if the detected class is in your defined list
            if cls < len(Class1):
                cvzone.cornerRect(img, (x1, y1, w, h), l=15, t=3, colorR=(255, 0, 0), colorC=(0, 255, 0)) 
    
                conf = math.ceil((box.conf[0] * 100)) / 100
    
                    # Now, this line is safe because the index `cls` is guaranteed to be valid
                cvzone.putTextRect(img, f'{Class1[cls]} {conf}', (max(0, x1), max(20, y1)), scale=0.7, thickness=1)
    
    
    
    cv2.imshow("Image", img)  # Display the frame
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the camera and close all OpenCV windowsq
cap.release()
cv2.destroyAllWindows()
