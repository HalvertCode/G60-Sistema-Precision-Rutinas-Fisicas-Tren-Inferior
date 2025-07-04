import tkinter as tk
import customtkinter as ck
import pandas as pd
import numpy as np
import pickle
import mediapipe as mp
import cv2
from PIL import Image, ImageTk
from landmarks import landmarks
from deadlift_form_checker import DeadliftFormChecker

window = tk.Tk()
window.geometry("480x750")  # Aumenté la altura para más elementos
window.title("Deadlift Trainer")
ck.set_appearance_mode("dark")

# Labels principales
classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
classLabel.place(x=10, y=1)
classLabel.configure(text='ETAPA')
counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
counterLabel.place(x=160, y=1)
counterLabel.configure(text='REPS')
probLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
probLabel.place(x=300, y=1)
probLabel.configure(text='PROB')

# Boxes principales
classBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
classBox.place(x=10, y=41)
classBox.configure(text='0')
counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
counterBox.place(x=160, y=41)
counterBox.configure(text='0')
probBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
probBox.place(x=300, y=41)
probBox.configure(text='0')

# Nuevos elementos para detección de errores
formLabel = ck.CTkLabel(window, height=30, width=200, font=("Arial", 16), text_color="black", padx=10)
formLabel.place(x=10, y=620)
formLabel.configure(text='ANÁLISIS DE FORMA')

# Indicador de estado de forma
formStatusBox = ck.CTkLabel(window, height=40, width=200, font=("Arial", 14), text_color="white", fg_color="green")
formStatusBox.place(x=10, y=650)
formStatusBox.configure(text='✓ FORMA CORRECTA')

# Área de alertas de errores
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
    global deadlift_form_checker
    deadlift_form_checker.reset_error_history()
    errorBox.delete("0.0", "end")
    errorBox.insert("0.0", "Errores reseteados")

# Botones
button = ck.CTkButton(window, text='RESET REPS', command=reset_counter, height=40, width=120, font=("Arial", 16), text_color="white", fg_color="blue")
button.place(x=10, y=580)

error_button = ck.CTkButton(window, text='RESET ERRORES', command=reset_errors, height=40, width=140, font=("Arial", 16), text_color="white", fg_color="red")
error_button.place(x=140, y=580)

# Frame del video
frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=90)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)

# Inicialización de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Cargar modelo
with open('deadlift.pkl', 'rb') as f:
    model = pickle.load(f)

# Inicializar variables
cap = cv2.VideoCapture(0)
current_stage = ''
counter = 0
bodylang_prob = np.array([0,0])
bodylang_class = ''

# Inicializar el verificador de forma
deadlift_form_checker = DeadliftFormChecker()

def detect():
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob
    global deadlift_form_checker

    ret, frame = cap.read()
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # Dibujar landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(106,13,173), thickness=4, circle_radius = 5),
        mp_drawing.DrawingSpec(color=(255,102,0), thickness=5, circle_radius = 10))

    try:
        if results.pose_landmarks:
            # Extraer coordenadas para el modelo
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns = landmarks)
            bodylang_prob = model.predict_proba(X)[0]
            bodylang_class = model.predict(X)[0]

            # Análisis de forma
            landmarks_dict = {}
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks_dict[i] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
            
            # Verificar forma del ejercicio
            form_errors = deadlift_form_checker.check_deadlift_form(landmarks_dict, bodylang_class)
            
            # Actualizar interfaz de errores
            update_error_display(form_errors)

            # Lógica de conteo (solo contar si la forma es buena)
            critical_errors = [error for error in form_errors if error['severity'] == 'critical']
            
            if bodylang_class =="down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
                current_stage = "abajo"
            elif current_stage == "abajo" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
                if len(critical_errors) == 0:  # Solo contar si no hay errores críticos
                    current_stage = "arriba"
                    counter += 1
                else:
                    current_stage = "arriba (sin contar)"

    except Exception as e:
        print(f"Error en detección: {e}")

    # Actualizar imagen
    img = image[:, :460, :]
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)

    # Actualizar displays
    counterBox.configure(text=counter)
    probBox.configure(text=f"{bodylang_prob[bodylang_prob.argmax()]:.2f}")
    classBox.configure(text=current_stage)

def update_error_display(form_errors):
    """Actualiza la visualización de errores en la interfaz"""
    if not form_errors:
        formStatusBox.configure(text="✓ FORMA CORRECTA", fg_color="green")
        errorBox.delete("0.0", "end")
        errorBox.insert("0.0", "Sin errores detectados")
    else:
        # Determinar color según severidad
        critical_errors = [e for e in form_errors if e['severity'] == 'critical']
        if critical_errors:
            formStatusBox.configure(text="⚠ FORMA CRÍTICA", fg_color="red")
        else:
            formStatusBox.configure(text="⚠ FORMA MEJORABLE", fg_color="orange")
        
        # Mostrar errores
        errorBox.delete("0.0", "end")
        error_text = ""
        for i, error in enumerate(form_errors[:3]):  # Mostrar máximo 3 errores
            severity_symbol = "🔴" if error['severity'] == 'critical' else "🟡"
            error_text += f"{severity_symbol} {error['message']}\n"
        
        errorBox.insert("0.0", error_text.strip())

# Iniciar detección
detect()
window.mainloop()

# Limpiar recursos
cap.release()
cv2.destroyAllWindows()