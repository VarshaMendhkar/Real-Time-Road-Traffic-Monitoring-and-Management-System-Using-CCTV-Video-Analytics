import cv2  # OpenCV for video processing
import torch  # PyTorch for loading the YOLO model
import os  # For creating folder and saving images
import time  # To generate unique filenames
import yagmail  # Email sending library
import numpy as np  # For numerical operations
import warnings
warnings.filterwarnings("ignore")
# Email details

sender_email="mendhkarvb2003@gmail.com"
password="rmve xvgd ngcm nzkw"
receiver_email = "varshamendhkar@gmail.com"

# Load the YOLOv5 accident detection model
accident_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp5/weights/best.pt')

# Load the YOLOv5 vehicle detection model
signal_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load the video - make sure the path is correct
#video_path = 'test3.mp4'
video_path = 'testing2.mp4'
# video_path='test3.mp4'

cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Create a directory to store screenshots of accidents
    output_folder = 'accident_detected'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize a buffer for accident detection
    accident_buffer = []  # Buffer to hold the last 5 frames' accident detection results

    while cap.isOpened():
        ret, frame = cap.read()  # Read the video frame-by-frame
        if not ret:
            print("Finished processing the video.")
            break  # Break if no more frames

        # Run inference on the current frame for accident detection
        accident_results = accident_model(frame)

        # Initialize a flag to check for accidents in the current frame
        accident_detected = False

        # Check accident detection results
        for *box, conf, cls in accident_results.xyxy[0]:  # Iterate through detections
            x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
            label = f'Class: {int(cls)}, Conf: {conf:.2f}'  # Create label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Draw label

            # Check if the detected class is "accident" (assuming class ID 0)
            if int(cls) == 0:
                accident_detected = True
                break  # Exit the loop if an accident is detected

        # Update the accident buffer
        accident_buffer.append(accident_detected)
        if len(accident_buffer) > 5:
            accident_buffer.pop(0)  # Keep the buffer size at 5

        # If there are 5 consecutive frames with accidents detected
        if accident_buffer.count(True) >= 5:
            # Save screenshot
            timestamp = time.strftime("%d%m%Y-%H%M%S")
            filename = os.path.join(output_folder, f"accident_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Accident detected! Screenshot saved as {filename}")

            # Modify the email subject and body
            subject = "Accident Detected!"
            location_link = "https://maps.app.goo.gl/vjfDbKHRTjXf9dwq9"
            body = f"An accident was detected near to this location {location_link}."

            try:
                # Initialize yagmail and send the email with the screenshot as an attachment
                yag = yagmail.SMTP(user=sender_email, password=password)
                yag.send(
                    to=receiver_email,
                    subject=subject,
                    contents=[body, filename]  # Attach the screenshot
                )
                print("Email sent successfully with the screenshot!")

            except Exception as e:
                print(f"Failed to send email: {str(e)}")

            # Print signal management based on accident detection (no further processing)
            print("Accident detected, managing signal to stop traffic.")
            break  # Exit the loop after detecting an accident

        # If no accident detected, proceed with traffic signal management based on vehicle count
        else:
            # Initialize variables for signal timing and vehicle count
            base_green_time = 10  # Base duration of green signal in seconds
            base_yellow_time = 3  # Base duration of yellow signal in seconds
            base_red_time = 10    # Base duration of red signal in seconds
            signal_state = 'Green'
            signal_start_time = time.time()
            vehicle_count = 0
            vehicle_buffer = []
            buffer_size = 10  # Number of frames to track vehicles
            vehicle_count_threshold = 5  # Threshold to avoid overestimating vehicle count

            # Run vehicle detection for signal management
            results = signal_model(frame)
            detections = results.xyxy[0]  # Get detections as a tensor
            current_vehicle_count = 0  # Reset the count for this frame

            for *box, conf, cls in detections:
                x1, y1, x2, y2 = map(int, box)
                label = f'{signal_model.names[int(cls)]} {conf:.2f}'
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Count only vehicles (assuming vehicle classes are 2 and 3 for car and truck)
                if int(cls) in [2, 3]:  # Adjust class IDs based on your model's classes
                    current_vehicle_count += 1  # Count vehicles in this frame

            # Buffer mechanism: Keep track of vehicle counts over multiple frames
            vehicle_buffer.append(current_vehicle_count)
            if len(vehicle_buffer) > buffer_size:
                vehicle_buffer.pop(0)  # Keep the buffer size constant

            # Average the vehicle count over the last `buffer_size` frames
            avg_vehicle_count = np.mean(vehicle_buffer)

            # Apply the vehicle count threshold
            if avg_vehicle_count > vehicle_count_threshold:
                vehicle_count = int(avg_vehicle_count)  # Update vehicle count if above the threshold

            # Calculate new signal times based on vehicle count
            if vehicle_count > 0:
                green_time = base_green_time + (vehicle_count * 0.5)  # Example: 0.5 seconds for each vehicle
            else:
                green_time = base_green_time

            elapsed_time = time.time() - signal_start_time
            if signal_state == 'Green' and elapsed_time > green_time:
                signal_state = 'Yellow'
                signal_start_time = time.time()
            elif signal_state == 'Yellow' and elapsed_time > base_yellow_time:
                signal_state = 'Red'
                signal_start_time = time.time()
            elif signal_state == 'Red' and elapsed_time > base_red_time:
                signal_state = 'Green'
                signal_start_time = time.time()

            # Display the signal state on the video frame
            cv2.putText(frame, f'Signal: {signal_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Traffic Signal Detection', frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # After the loop, release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Print the final vehicle count and estimated signal times
    print(f'Total vehicles detected: {vehicle_count}')
    estimated_time_to_clear = (vehicle_count / 10) + base_red_time + base_yellow_time  # Example calculation

    # Calculate the signal times based on the estimated time to clear
    if estimated_time_to_clear > 0:
        total_time = estimated_time_to_clear * 60  # Convert minutes to seconds
        
        # Adjust the signal times realistically
        if estimated_time_to_clear > 10:  # High traffic, prioritize green
            green_time = total_time * 0.65
            yellow_time = total_time * 0.10
            red_time = total_time * 0.25
        else:
            green_time = total_time * 0.55
            yellow_time = total_time * 0.15
            red_time = total_time * 0.30

        # Convert the times back to seconds for display
        green_time_seconds = green_time
        yellow_time_seconds = yellow_time
        red_time_seconds = red_time

        print(f'Estimated time to clear traffic after processing: {estimated_time_to_clear:.2f} minutes')
        print(f'Signal Times: Green: {green_time_seconds:.2f} seconds, Yellow: {yellow_time_seconds:.2f} seconds, Red: {red_time_seconds:.2f} seconds')

    else:
        print('No vehicles detected; no signal times calculated.')

    # Print final message
    if accident_buffer.count(True) >= 5:
        print("Accident detected and managed!")
    else:
        print("No accidents detected, signal managed based on vehicle count.")
