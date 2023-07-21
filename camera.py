from model import HandGestureModel
import numpy as np
import cv2
import mediapipe as mp


model = HandGestureModel("model_json", "model_weights.h5")

class VideoCamera(object):

    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(static_image_mode=False,
                              max_num_hands=2,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)
        mpDraw = mp.solutions.drawing_utils
        # Initialize the webcam
        while True:
            ret, img = self.video.read()
            if ret is True:
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(imgRGB)

                rx = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)/2
                ry = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)/2
                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                        for id, lm in enumerate(handLms.landmark):
                            # print(id,lm)
                            h, w, c = img.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            # if id ==0:
                            cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
                            if id == 9:
                                rx = int(lm.x * w)
                                ry = int(lm.y * h)
                                cv2.rectangle(img, (rx - 300, ry - 300), (rx + 300, ry + 300), (255, 0, 0), 3)
                                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                image = img[(ry-300):(ry + 300), (rx - 300):(rx + 300)]
                try:
                    image = cv2.resize(image, (48, 48))
                    label = model.predict_gesture(image[np.newaxis, :, :])
                    cv2.putText(img, str(label), (rx - 300, ry - 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except:
                    print('error')

                # Show the frame with the bounding box and prediction label in a window
                _, jpeg = cv2.imencode('.jpg', img)
                return jpeg.tobytes()
