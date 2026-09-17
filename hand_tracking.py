import cv2
import mediapipe as mp


class HandTracker:

    def __init__(
        self,
        max_num_hands=1,
        detection_confidence=0.7,
        tracking_confidence=0.7
    ):
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return self.hands.process(rgb)

    def draw_landmarks(self, frame, hand_landmarks):

        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )

    def get_fingertip(self, frame, hand_landmarks):

        h, w, _ = frame.shape

        # Landmark 8 = index fingertip
        tip = hand_landmarks.landmark[8]

        x = int(tip.x * w)
        y = int(tip.y * h)

        # Keep coordinates inside frame
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        return x, y

    def close(self):
        self.hands.close()