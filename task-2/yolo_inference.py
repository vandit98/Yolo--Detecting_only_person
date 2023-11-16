
import cv2
from ultralytics import YOLO
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model('emotion_model/')

def emotion_predictor(crop_img):
    return "happy"
    # Resize the image to the input size expected by the emotion model
    test_image = cv2.resize(crop_img, (227, 227))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = loaded_model.predict(test_image)
    ans = ['neutral', 'sad', 'happy']
    max_prob_index = np.argmax(result)
    result=result[0]
# Check if there are multiple emotions with the same probability
    if np.sum(result == result[max_prob_index]) > 1:
        # In case of a tie, prioritize 'happy' over 'sad'
        happy_index = 2
        sad_index = 1
        nuetral_index = 0

        # Check if both 'happy' and 'sad' have the same probability
        if result[happy_index] == result[sad_index]:
            # Set the predicted emotion to 'neutral'
            predicted_emotion = 'happy'
            return predicted_emotion
        elif result[happy_index]== result[nuetral_index]:
            predicted_emotion = 'happy'
            return predicted_emotion
        else:
            # Choose the one with the higher probability
            predicted_emotion = ans[happy_index] if result[happy_index] >= result[sad_index] else ans[sad_index]
            return predicted_emotion
    else:
        # If no tie, choose the emotion with the highest probability
        predicted_emotion = ans[max_prob_index]

    print("Predicted Emotion:", predicted_emotion)
    return predicted_emotion

# Load the YOLOv8 model
model = YOLO('/home/vandit/Downloads/learning/Applied_Ml/yolo/Yolo- Detecting_only_person/task-2/yolov8n-face.pt')

# Open the video file
video_path = "/home/vandit/Downloads/learning/Applied_Ml/yolo/Yolo- Detecting_only_person/task-1/input_video/sample3.mp4"
cap = cv2.VideoCapture(video_path)

# Get video details
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object for MP4
output_path = "/home/vandit/Downloads/learning/Applied_Ml/yolo/Yolo- Detecting_only_person/task-2/output_videos/output3.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        # Iterate over the boxes and get emotion predictions
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() 
            confidences = result.boxes.conf.cpu().numpy()  # Access confidence scores
            class_labels = result.boxes.cls.cpu().numpy()
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                # print("*"*100)
                
                # print("class label is ",class_labels[i])
                crop_img = frame[int(y1):int(y2), int(x1):int(x2)]
                
        #         # Predict emotion for the cropped face
                emotion = emotion_predictor(crop_img)
                confidence = confidences[i]
                class_label = emotion
                # print("class label is ",class_label)
        #         # Annotate the frame with the predicted emotion
            
              
                cv2.putText(frame, emotion, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                # cv2.imshow("cropped", crop_img)
        # annotated_frame = results[0].plot(    ) #  We dont write the name of the class , we only show the bounding box

        # Save the annotated frame to the output video
        out.write(frame)

        # Display the annotated frame
        # cv2.imshow("YOLOv8 Inference", frame)

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

