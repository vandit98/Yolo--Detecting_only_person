# import cv2
# from ultralytics import YOLO
# import matplotlib.pyplot as plt

# # Load the YOLOv8 model
# # load th trained model

# model = YOLO('best.pt')

# # # Open the video file
# video_path = "/home/vandit/Downloads/learning/Applied_Ml/yolo/sample.mp4"
# cap = cv2.VideoCapture(video_path)


# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model(frame)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Inference", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
# load the trained model
model = YOLO('best.pt')




for i in range(1,5):
# Open the video file
    video_path = f"sample_video/sample{i}.mp4"
    cap = cv2.VideoCapture(video_path)

    # Get video details
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object for MP4
    output_path = f"sample_output/output{i}.mp4"  # Change this to your desired output path with .mp4 extension
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Save the annotated frame to the output video
            out.write(annotated_frame)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Close all windows
    cv2.destroyAllWindows()


 