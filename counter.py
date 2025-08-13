import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the vertical line position (e.g., in the middle of the frame)
line_position_x = frame_width // 2
LINE_START = (line_position_x, 0)
LINE_END = (line_position_x, frame_height)

# List of COCO class IDs for animals you want to detect
# cat, dog, horse, sheep, cow
desired_classes = [15, 16, 18, 19, 20] 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- SIMPLIFIED LOGIC ---
    # 1. Reset the count and line color for every new frame
    current_count = 0
    line_color = (0, 255, 0) # Green (default)

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, classes=desired_classes, verbose=False)

    # Plot the results on the frame
    annotated_frame = results[0].plot()

    # Check if any objects were detected
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        
        # Iterate over all detected boxes in the current frame
        for box in boxes:
            x, y, w, h = box
            
            # Calculate the left and right edges of the bounding box
            left_edge = x - w / 2
            right_edge = x + w / 2
            
            # 2. Check if the bounding box intersects with the vertical line
            if left_edge <= line_position_x <= right_edge:
                # If it intersects, set count to 1 and change line color
                current_count = 1
                line_color = (0, 0, 255) # Red (indicates contact)
                # Since we only care if *any* object is touching, we can stop checking
                break # Exit the loop for this frame

    # 3. Draw the line and display the count for the current frame
    cv2.line(annotated_frame, LINE_START, LINE_END, line_color, 3)
    cv2.putText(annotated_frame, f"Count: {current_count}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    cv2.imshow("YOLOv8 Simple Counter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()