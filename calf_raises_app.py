import tkinter as tk
import customtkinter as ck
import pandas as pd
import numpy as np
import pickle
import mediapipe as mp
import cv2
from PIL import Image, ImageTk
from landmarks import landmarks
from calf_raises_form_checker import CalfRaisesFormChecker

window = tk.Tk()
window.geometry("480x750")  # Ajustado para espacio adicional
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

# Label de debugging específico
angleLabel = ck.CTkLabel(window, height=30, width=200, font=("Arial", 14), text_color="white", fg_color="gray")
angleLabel.place(x=10, y=550)
angleLabel.configure(text='Δ Altura: --')

def reset_counter():
    global counter
    counter = 0

def reset_errors():
    global calf_raises_form_checker
    calf_raises_form_checker.reset_error_history()
    errorBox.delete("0.0", "end")
    errorBox.insert("0.0", "Errores reseteados")

# Botones
resetRepsBtn = ck.CTkButton(window, text='RESET REPS', command=reset_counter,
                            height=40, width=120, font=("Arial", 16),
                            text_color="white", fg_color="blue")
resetRepsBtn.place(x=10, y=580)

resetErrsBtn = ck.CTkButton(window, text='RESET ERRORES', command=reset_errors,
                            height=40, width=140, font=("Arial", 16),
                            text_color="white", fg_color="red")
resetErrsBtn.place(x=140, y=580)

# Frame de video
frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=90)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)

# MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Intentar cargar modelo
try:
    with open('calf_raises.pkl', 'rb') as f:
        model = pickle.load(f)
    use_model = True
except FileNotFoundError:
    use_model = False

cap = cv2.VideoCapture(0)
current_stage = ''
counter = 0
bodylang_prob = np.array([0, 0])
bodylang_class = ''
calf_raises_form_checker = CalfRaisesFormChecker()

def update_error_display(form_errors):
    if not form_errors:
        formStatusBox.configure(text="✓ FORMA CORRECTA", fg_color="green")
        errorBox.delete("0.0", "end")
        errorBox.insert("0.0", "Sin errores detectados")
    else:
        crit = [e for e in form_errors if e['severity']=='critical']
        formStatusBox.configure(
            text="⚠ FORMA CRÍTICA" if crit else "⚠ FORMA MEJORABLE",
            fg_color="red" if crit else "orange"
        )
        errorBox.delete("0.0", "end")
        txt = ""
        for e in form_errors[:3]:
            sym = "🔴" if e['severity']=='critical' else "🟡"
            txt += f"{sym} {e['message']}\n"
        errorBox.insert("0.0", txt.strip())

def detect():
    global current_stage, counter, bodylang_class, bodylang_prob, calf_raises_form_checker
    ret, frame = cap.read()
    if not ret:
        return

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:
            lm = results.pose_landmarks.landmark

            # Detección de altura de talones
            left_h, left_a = lm[mp_pose.PoseLandmark.LEFT_HEEL.value], lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_h, right_a = lm[mp_pose.PoseLandmark.RIGHT_HEEL.value], lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            diff_l = left_a.y - left_h.y
            diff_r = right_a.y - right_h.y
            avg_diff = (diff_l + diff_r)/2
            angleLabel.configure(text=f'Δ Izq: {diff_l:.3f} Der: {diff_r:.3f}')

            if use_model:
                row = np.array([[v.x, v.y, v.z, v.visibility] for v in lm]).flatten().tolist()
                X = pd.DataFrame([row], columns=landmarks)
                bodylang_prob = model.predict_proba(X)[0]
                bodylang_class = model.predict(X)[0]

                if bodylang_class=="down" and bodylang_prob.max()>0.7:
                    current_stage="abajo"
                elif current_stage=="abajo" and bodylang_class=="up" and bodylang_prob.max()>0.7:
                    current_stage="arriba"
                    counter += 1

            else:
                thresh = 0.05
                if avg_diff < thresh and current_stage=="arriba":
                    current_stage="abajo"
                    bodylang_prob = np.array([0.85,0.15])
                    bodylang_class="down"
                elif avg_diff>=thresh and current_stage!="arriba":
                    current_stage="arriba"
                    counter += 1
                    bodylang_prob = np.array([0.15,0.85])
                    bodylang_class="up"

            # Preparar landmarks para análisis
            landmarks_dict = {i: {'x': v.x, 'y': v.y, 'z': v.z, 'visibility': v.visibility}
                              for i, v in enumerate(lm)}

            # Análisis de forma con calf_raises_form_checker
            errors = calf_raises_form_checker.check_calf_raise_form(landmarks_dict, current_stage)
            update_error_display(errors)

        except Exception as e:
            print(f"Error en detección: {e}")

    # Render video en GUI
    img = image[:, :460, :]
    imgtk = ImageTk.PhotoImage(Image.fromarray(img))
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)

    # Actualizar contadores y etiquetas
    counterBox.configure(text=counter)
    probBox.configure(text=f"{bodylang_prob.max():.2f}")
    classBox.configure(text=current_stage)

detect()
window.mainloop()
cap.release()
cv2.destroyAllWindows()