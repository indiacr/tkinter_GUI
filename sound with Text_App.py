
import tkinter as tk
from tkinter import Label, Button
import webbrowser
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import pickle
from gtts import gTTS
import pygame
import tempfile


# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Mediapipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)
labels_dict = {0: 'L', 1: 'A', 2: 'B', 3: 'C', 4: 'V', 5: 'W', 6: 'Y', 7: 'He', 8: 'Hello', 9: 'Yes', 10: 'me',
               11: 'Sorry', 12: 'Know', 13: 'Eat', 14: 'You'}

# Initialize Pygame mixer
pygame.mixer.init()

# Track the current and previous predictions
previous_prediction = None
audio_playing = False


class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition")

        # Full screen mode
        self.root.attributes("-fullscreen", True)

        # Add buttons for option selection
        self.sign_to_text_button = Button(root, text="Sign to Text and Voice", command=self.run_sign_to_text,
                                          font=("Helvetica", 20))
        self.sign_to_text_button.pack(pady=10)

        self.voice_to_sign_button = Button(root, text="Voice or Text to Sign", command=self.open_voice_to_sign_url,
                                           font=("Helvetica", 20))
        self.voice_to_sign_button.pack(pady=10)

        self.stop_voice_to_sign_button = Button(root, text="Stop Voice to Sign", command=self.stop_voice_to_sign,
                                                font=("Helvetica", 20))
        self.stop_voice_to_sign_button.pack(pady=10)

        self.stop_sign_to_text_button = Button(root, text="Stop Sign to Text", command=self.stop_sign_to_text,
                                               font=("Helvetica", 20))
        self.stop_sign_to_text_button.pack(pady=10)

        self.exit_fullscreen_button = Button(root, text="Exit Full Screen", command=self.exit_fullscreen,
                                             font=("Helvetica", 20))
        self.exit_fullscreen_button.pack(pady=10)

        self.exit_button = Button(root, text="Exit", command=self.on_closing, font=("Helvetica", 20))
        self.exit_button.pack(pady=10)

        self.label = Label(root)
        self.label.pack()

        self.cap = None
        self.browser_window = None

    def run_sign_to_text(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def open_voice_to_sign_url(self):
        # Open the URL in a new browser window
        webbrowser.open("https://sign.mt/")

    def stop_voice_to_sign(self):
        # Display a message instructing the user to close the browser tab manually
        message_window = tk.Toplevel(self.root)
        message_window.title("Close Browser Tab")
        message_label = Label(message_window,
                              text="Please close the browser tab manually to stop the voice-to-sign feature.",
                              font=("Helvetica", 16))
        message_label.pack(padx=20, pady=20)
        Button(message_window, text="OK", command=message_window.destroy, font=("Helvetica", 16)).pack(pady=10)

    def stop_sign_to_text(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.label.configure(image='')

    def update_frame(self):
        global previous_prediction, audio_playing

        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data_aux = []

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    data_aux.extend([x, y])

        # Ensure the number of features matches the model's expectations
        expected_features = 42  # Update this to match the model's expected feature count
        if len(data_aux) == expected_features:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_value = int(prediction[0])
            if predicted_value in labels_dict:
                predicted_character = labels_dict[predicted_value]

                if predicted_character != previous_prediction:
                    previous_prediction = predicted_character

                    text = predicted_character
                    tts = gTTS(text=text, lang='en')

                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                        temp_filename = fp.name

                    tts.save(temp_filename)
                    pygame.mixer.music.load(temp_filename)
                    pygame.mixer.music.play()
                    audio_playing = True

                cv2.putText(frame, f'Sign: {predicted_character}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0),
                            3, cv2.LINE_AA)
        else:
            print(f"Feature count mismatch: expected {expected_features}, but got {len(data_aux)}")

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        if self.cap is not None and self.cap.isOpened():
            self.root.after(10, self.update_frame)

    def exit_fullscreen(self):
        self.root.attributes("-fullscreen", False)

    def on_closing(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()
        pygame.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()