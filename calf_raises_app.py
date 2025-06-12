import tkinter as tk
import customtkinter as ck
import pandas as pd
import numpy as np
import pickle
import mediapipe as mp
import cv2
from PIL import Image, ImageTk
from landmarks import landmarks

window = tk.Tk()
window.geometry("480x700")
window.title("Detector de Elevación de Pantorrillas")
ck.set_appearance_mode("dark")

# Labels de información
classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
classLabel.place(x=10, y=1)
classLabel.configure(text='ETAPA')
counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
counterLabel.place(x=160, y=1)
counterLabel.configure(text='REPS')
probLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
probLabel.place(x=300, y=1)
probLabel.configure(text='PROB')

# Cajas de información
classBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
classBox.place(x=10, y=41)
classBox.configure(text='0')
counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
counterBox.place(x=160, y=41)
counterBox.configure(text='0')
probBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
probBox.place(x=300, y=41)
probBox.configure(text='0')

# Label adicional para mostrar diferencia de altura talón-tobillo (útil para debugging)
angleLabel = ck.CTkLabel(window, height=30, width=200, font=("Arial", 14), text_color="white", fg_color="gray")
angleLabel.place(x=10, y=550)
angleLabel.configure(text='Δ Altura: --')

def reset_counter():
    global counter
    counter = 0

button = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
button.place(x=10, y=600)

frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=90)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)

# Configuración de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Intentar cargar el modelo de elevación de pantorrillas (si existe)
try:
    with open('calf_raises.pkl', 'rb') as f:
        model = pickle.load(f)
    use_model = True
    print("Modelo de elevación de pantorrillas cargado exitosamente")
except FileNotFoundError:
    print("Modelo 'calf_raises.pkl' no encontrado. Usando detección por altura de talones.")
    use_model = False

cap = cv2.VideoCapture(0)
current_stage = ''
counter = 0
bodylang_prob = np.array([0, 0])
bodylang_class = ''

def detect():
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob

    ret, frame = cap.read()
    if not ret:
        return

    # Convertir BGR a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Dibujar landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(106,13,173), thickness=4, circle_radius=5),
            mp_drawing.DrawingSpec(color=(255,102,0), thickness=5, circle_radius=10)
        )

        try:
            landmarks_data = results.pose_landmarks.landmark

            if use_model:
                # Usar modelo entrenado si está disponible
                row = np.array([[res.x, res.y, res.z, res.visibility] for res in landmarks_data]).flatten().tolist()
                X = pd.DataFrame([row], columns=landmarks)
                bodylang_prob = model.predict_proba(X)[0]
                bodylang_class = model.predict(X)[0]

                if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
                    current_stage = "abajo"
                elif current_stage == "abajo" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
                    current_stage = "arriba"
                    counter += 1

            else:
                # Detección basada en la diferencia de altura talón-tobillo
                # Coords normalizadas [0..1], y crece hacia abajo en la imagen
                left_heel = landmarks_data[mp_pose.PoseLandmark.LEFT_HEEL.value]
                left_ankle = landmarks_data[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                right_heel = landmarks_data[mp_pose.PoseLandmark.RIGHT_HEEL.value]
                right_ankle = landmarks_data[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                # Altura (coordenada y) del talón y tobillo para cada pierna
                y_heel_left = left_heel.y
                y_ankle_left = left_ankle.y
                y_heel_right = right_heel.y
                y_ankle_right = right_ankle.y

                # Diferencia de altura tobillo - talón (positivo si el talón está más arriba que el tobillo)
                diff_left = y_ankle_left - y_heel_left
                diff_right = y_ankle_right - y_heel_right
                avg_diff = (diff_left + diff_right) / 2.0

                # Mostrar diferencia en label para debugging
                angleLabel.configure(text=f'Δ Izq: {diff_left:.3f} Der: {diff_right:.3f}')

                # Umbral para considerar que el talón está levantado
                heel_up_threshold = 0.05

                # Detectar posición "abajo" (talones en el suelo)
                if avg_diff < heel_up_threshold:
                    if current_stage == "arriba":
                        current_stage = "abajo"
                        # Asignar probabilidad aproximada para "down"
                        bodylang_prob = np.array([0.85, 0.15])
                        bodylang_class = "down"
                # Detectar transición a "arriba" (talones elevados)
                elif avg_diff >= heel_up_threshold:
                    if current_stage != "arriba":
                        current_stage = "arriba"
                        counter += 1
                        # Asignar probabilidad aproximada para "up"
                        bodylang_prob = np.array([0.15, 0.85])
                        bodylang_class = "up"

        except Exception as e:
            print(f"Error en detección: {e}")

    # Mostrar imagen en GUI
    img = image[:, :460, :]
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)

    # Actualizar interfaz
    counterBox.configure(text=counter)
    if len(bodylang_prob) > 0:
        probBox.configure(text=f'{bodylang_prob[bodylang_prob.argmax()]:.2f}')
    classBox.configure(text=current_stage)

detect()
window.mainloop()