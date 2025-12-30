import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import math

from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions import face_mesh_connections as mp_connections

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7)

img_boohoo = cv2.imread('Assets/boohoo.webp')
img_smile = cv2.imread('Assets/smile.webp')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    display_img = frame.copy()

    mouth_x, mouth_y = 0.5, 0.5

    # --- FACE LOGIC & SKELETON ---
    if face_results.multi_face_landmarks:
        for face_lms in face_results.multi_face_landmarks:
            mp_draw.draw_landmarks(display_img, face_lms, mp_connections.FACEMESH_TESSELATION,
                                   None, mp_draw.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1))

            # Landmark 13 is the center of the lips
            mouth_x = face_lms.landmark[13].x
            mouth_y = face_lms.landmark[13].y

            # Smile Ratio Logic
            m_left, m_right = face_lms.landmark[61], face_lms.landmark[291]
            e_left, e_right = face_lms.landmark[33], face_lms.landmark[263]
            smile_ratio = abs(m_left.x - m_right.x) / abs(e_left.x - e_right.x)

            if smile_ratio > 0.75:
                if img_smile is not None:
                    display_img = cv2.resize(img_smile, (w, h))

    # --- HAND LOGIC ---
    if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) == 2:
        fists_in_position = 0

        for hand_lms in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(display_img, hand_lms, mp_hands.HAND_CONNECTIONS)

            knuckle = hand_lms.landmark[5]

            is_fist = hand_lms.landmark[8].y > hand_lms.landmark[5].y

            dist_to_mouth = math.sqrt((knuckle.x - mouth_x) ** 2 + (knuckle.y - mouth_y) ** 2)

            if is_fist and dist_to_mouth < 0.15:
                fists_in_position += 1

        if fists_in_position == 2 and img_boohoo is not None:
            display_img = cv2.resize(img_boohoo, (w, h))

    cv2.imshow("Clash Emotes AI", display_img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()