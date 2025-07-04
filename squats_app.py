import tkinter as tk
import customtkinter as ck
import pandas as pd
import numpy as np
import pickle
import mediapipe as mp
import cv2
from PIL import Image, ImageTk
from landmarks import landmarks
from squats_form_checker import SquatsFormChecker

window = tk.Tk()
window.geometry("480x750")  # Ajustado para espacio adicional
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

# Indicador de estado de forma
formLabel = ck.CTkLabel(window, height=30, width=200, font=("Arial", 16), text_color="black", padx=10)
formLabel.place(x=10, y=620)
formLabel.configure(text='ANÁLISIS DE FORMA')

formStatusBox = ck.CTkLabel(window, height=40, width=200, font=("Arial", 14), text_color="white", fg_color="green")
formStatusBox.place(x=10, y=650)
formStatusBox.configure(text='✓ FORMA CORRECTA')

# Área de errores
errorLabel = ck.CTkLabel(window, height=30, width=200, font=("Arial", 16), text_color="black", padx=10)
errorLabel.place(x=250, y=620)
errorLabel.configure(text='ERRORES DETECTADOS')

errorBox = ck.CTkTextbox(window, height=60, width=220, font=("Arial", 12))
errorBox.place(x=250, y=650)
errorBox.insert("0.0", "Sin errores detectados")

def reset_counter():
    global counter
    counter = 0

def reset_errors():
    global squats_form_checker
    squats_form_checker.reset_error_history()
    errorBox.delete("0.0", "end")
    errorBox.insert("0.0", "Errores reseteados")

# Botones
button = ck.CTkButton(window, text='RESET REPS', command=reset_counter, height=40, width=120, font=("Arial", 16), text_color="white", fg_color="blue")
button.place(x=10, y=580)

error_button = ck.CTkButton(window, text='RESET ERRORES', command=reset_errors, height=40, width=140, font=("Arial", 16), text_color="white", fg_color="red")
error_button.place(x=140, y=580)

# Frame de video
frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=90)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)

# MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Calcular ángulo
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Modelo
try:
    with open('squats.pkl', 'rb') as f:
        model = pickle.load(f)
    use_model = True
except FileNotFoundError:
    use_model = False

cap = cv2.VideoCapture(0)
current_stage = ''
counter = 0
bodylang_prob = np.array([0, 0])
bodylang_class = ''
squats_form_checker = SquatsFormChecker()

def update_error_display(form_errors):
    if not form_errors:
        formStatusBox.configure(text="✓ FORMA CORRECTA", fg_color="green")
        errorBox.delete("0.0", "end")
        errorBox.insert("0.0", "Sin errores detectados")
    else:
        critical_errors = [e for e in form_errors if e['severity'] == 'critical']
        formStatusBox.configure(
            text="⚠ FORMA CRÍTICA" if critical_errors else "⚠ FORMA MEJORABLE",
            fg_color="red" if critical_errors else "orange"
        )
        errorBox.delete("0.0", "end")
        error_text = ""
        for i, error in enumerate(form_errors[:3]):
            symbol = "🔴" if error['severity'] == 'critical' else "🟡"
            error_text += f"{symbol} {error['message']}\n"
        errorBox.insert("0.0", error_text.strip())

def detect():
    global current_stage, counter, bodylang_class, bodylang_prob, squats_form_checker
    ret, frame = cap.read()
    if not ret:
        return

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    try:
        if results.pose_landmarks:
            landmarks_data = results.pose_landmarks.landmark

            # Estructura para análisis de forma
            landmarks_dict = {
                i: {
                    'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility
                } for i, lm in enumerate(landmarks_data)
            }

            form_errors = squats_form_checker.check_squat_form(landmarks_dict, bodylang_class)
            update_error_display(form_errors)
            critical_errors = [e for e in form_errors if e['severity'] == 'critical']

            if use_model:
                row = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks_data]).flatten().tolist()
                X = pd.DataFrame([row], columns=landmarks)
                bodylang_prob = model.predict_proba(X)[0]
                bodylang_class = model.predict(X)[0]
                if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
                    current_stage = "abajo"
                elif current_stage == "abajo" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
                    if len(critical_errors) == 0:
                        current_stage = "arriba"
                        counter += 1
                    else:
                        current_stage = "arriba (sin contar)"
            else:
                # Lógica con ángulos
                hip = [landmarks_data[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks_data[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks_data[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks_data[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks_data[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks_data[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                angle = calculate_angle(hip, knee, ankle)
                if angle < 90 and current_stage != "down":
                    current_stage = "down"
                elif angle > 160 and current_stage == "down":
                    if len(critical_errors) == 0:
                        current_stage = "up"
                        counter += 1
                    else:
                        current_stage = "up (sin contar)"

    except Exception as e:
        print(f"Error en detección: {e}")

    img = image[:, :460, :]
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)

    counterBox.configure(text=counter)
    probBox.configure(text=f"{bodylang_prob[bodylang_prob.argmax()]:.2f}")
    classBox.configure(text=current_stage)

detect()
window.mainloop()
cap.release()
cv2.destroyAllWindows()