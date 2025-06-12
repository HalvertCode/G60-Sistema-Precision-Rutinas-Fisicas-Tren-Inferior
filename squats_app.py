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
window.title("Detector de Squats")
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

# Label adicional para mostrar ángulos de rodilla (útil para debugging)
angleLabel = ck.CTkLabel(window, height=30, width=200, font=("Arial", 14), text_color="white", fg_color="gray")
angleLabel.place(x=10, y=550)
angleLabel.configure(text='Ángulos: --')

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

# Función para calcular ángulos
def calculate_angle(a, b, c):
    """
    Calcula el ángulo entre tres puntos
    a, b, c son arrays numpy con coordenadas [x, y]
    """
    a = np.array(a)
    b = np.array(b)  # vértice
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Intentar cargar el modelo de squats (si existe)
try:
    with open('squats.pkl', 'rb') as f:
        model = pickle.load(f)
    use_model = True
    print("Modelo de squats cargado exitosamente")
except FileNotFoundError:
    print("Modelo 'squats.pkl' no encontrado. Usando detección por ángulos.")
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
                # Detección basada en ángulos de rodilla para squats
                # Puntos clave para calcular ángulo de rodilla en ambas piernas
                hip_left = [landmarks_data[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks_data[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee_left = [landmarks_data[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks_data[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle_left = [landmarks_data[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks_data[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                hip_right = [landmarks_data[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks_data[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee_right = [landmarks_data[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks_data[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle_right = [landmarks_data[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks_data[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                # Calcular ángulos de las rodillas
                angle_left = calculate_angle(hip_left, knee_left, ankle_left)
                angle_right = calculate_angle(hip_right, knee_right, ankle_right)
                avg_angle = (angle_left + angle_right) / 2.0

                # Mostrar ángulos en label para debugging
                angleLabel.configure(text=f'Izq: {angle_left:.1f}° Der: {angle_right:.1f}°')

                # Umbrales de ángulo para detectar squats
                # Posición abajo: rodillas < 90°
                # Posición arriba: rodillas > 160°
                if avg_angle < 90:
                    if current_stage != "down":
                        current_stage = "down"
                        bodylang_prob = np.array([0.85, 0.15])  # Alta confianza en posición abajo
                        bodylang_class = "down"
                elif avg_angle > 160:
                    if current_stage == "down":
                        current_stage = "up"
                        counter += 1
                        bodylang_prob = np.array([0.15, 0.85])  # Alta confianza en posición arriba
                        bodylang_class = "up"
                    elif current_stage == "":
                        current_stage = "up"
                        bodylang_prob = np.array([0.1, 0.9])
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