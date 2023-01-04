import face_recognition
import os, sys
import cv2
import numpy as np
import math
import time


# Helper
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return round(linear_val * 100, 2)
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return round(value, 2)


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    counter = 0
    timer = 0
    counting_enabled = bool
    lap_time = 0
    total_start_time = time.time()
    current_time = 0
    total_time = 0

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)

    def run_recognition(self):
        self.counting_enabled = False
        name = ""
        confidence = "0"
        count_start_time = time.time()

        video_capture = cv2.VideoCapture(0)
        image = cv2.imread('/Users/SamsonIves/GIT/FacePace/UI_images/logo.png')
        image_height, image_width = image.shape[:2]

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()

            # Crop the top right quadrant of the frame
            left = frame.shape[1] // 2
            top = 0
            right = frame.shape[1]
            bottom = frame.shape[0] // 2
            frame_cropped = frame[top:bottom, left:right]

            if float(confidence) > 92.50 and confidence != "0":
                # Check the elapsed time since the last increment
                elapsed_time = time.time() - count_start_time
                print(elapsed_time)
                # time.sleep(10)
                if elapsed_time > 15:
                    print(self.timer)
                    print(confidence)
                    self.lap_time = elapsed_time
                    count_start_time = round(time.time(), 2)
                    self.counting_enabled = True
                else:
                    # Set the flag to False to prevent the counter from being incremented
                    self.counting_enabled = False

            # Check the flag to see if the counter should be incremented
            if self.counting_enabled:
                self.counter += 1

            # Only process every other frame of video to save time
            if self.process_current_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame_cropped, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = "0"

                    # Calculate the shortest distance to face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Create the frame with the name
                cv2.rectangle(frame_cropped, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame_cropped, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame_cropped, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255),
                            1)
            #   always show the current counter on the frame
            cv2.putText(frame, f'Counter: {self.counter}', (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Lap Time: {round(self.lap_time, 2)}', (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 1)
            self.total_time = time.time() - self.total_start_time
            cv2.putText(frame, f'Total Seconds: {round(self.total_time, 2)}', (10, 150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 1)


            frame[frame.shape[0] - image_height:frame.shape[0], 0:image_width] = image

            # Display the resulting image
            cv2.imshow('Face Recognition', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
