import numpy as np
import math
from collections import deque

class CalfRaisesFormChecker:
    def __init__(self, history_size=10):
        """
        Inicializa el verificador de forma para calf raises (elevaciones de pantorrillas)
        Args:
            history_size: Número de frames a mantener en historial
        """
        self.history_size = history_size
        self.error_history = deque(maxlen=history_size)

        # Umbrales específicos para calf raises
        self.thresholds = {
            'heel_lift_min': 0.05,        # Elevación mínima normalizada del talón
            'symmetry_diff_max': 0.02,    # Diferencia máxima entre talones
            'knee_angle_min': 160,        # Rodilla casi extendida
            'back_straightness': 20       # Inclinación máxima permitida en la espalda (grados)
        }

    def calculate_angle(self, p1, p2, p3):
        """
        Calcula el ángulo entre tres puntos p1-p2-p3
        Cada punto es un dict con 'x','y'.
        """
        try:
            v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
            v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            return angle
        except:
            return 0

    def check_knee_straightness(self, landmarks):
        """
        Verifica que las rodillas estén casi extendidas (cerca de 180°)
        """
        errors = []
        left_knee = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        right_knee = self.calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        if left_knee < self.thresholds['knee_angle_min']:
            errors.append({'message': 'Rodilla izq no extendida', 'severity': 'critical'})
        if right_knee < self.thresholds['knee_angle_min']:
            errors.append({'message': 'Rodilla der no extendida', 'severity': 'critical'})
        return errors

    def check_heel_lift(self, landmarks, current_stage):
        """
        Verifica elevación de talones y simetría
        """
        errors = []
        # Leer puntos
        left_h = landmarks[mp_pose_idx('LEFT_HEEL')]
        right_h = landmarks[mp_pose_idx('RIGHT_HEEL')]
        left_a = landmarks[mp_pose_idx('LEFT_ANKLE')]
        right_a = landmarks[mp_pose_idx('RIGHT_ANKLE')]

        diff_l = left_a['y'] - left_h['y']
        diff_r = right_a['y'] - right_h['y']
        avg_diff = (diff_l + diff_r) / 2

        # Fase abajo: talones no levantados
        if current_stage == 'down' and avg_diff >= self.thresholds['heel_lift_min']:
            errors.append({'message': 'Inició elevación sin bajar completamente', 'severity': 'warning'})
        # Fase arriba: talones deben estar levantados
        if current_stage == 'up' and avg_diff < self.thresholds['heel_lift_min']:
            errors.append({'message': 'Talones no suficientemente elevados', 'severity': 'critical'})

        # Simetría
        if abs(diff_l - diff_r) > self.thresholds['symmetry_diff_max']:
            errors.append({'message': 'Diferencia entre talones alta', 'severity': 'warning'})

        return errors

    def check_back_straightness(self, landmarks):
        """
        Verifica que la espalda se mantenga recta durante el movimiento
        """
        errors = []
        left_sh = landmarks[11]
        right_sh = landmarks[12]
        left_hp = landmarks[23]
        right_hp = landmarks[24]
        shoulder_mid = {'x': (left_sh['x']+right_sh['x'])/2, 'y': (left_sh['y']+right_sh['y'])/2}
        hip_mid = {'x': (left_hp['x']+right_hp['x'])/2, 'y': (left_hp['y']+right_hp['y'])/2}

        spine = np.array([shoulder_mid['x']-hip_mid['x'], shoulder_mid['y']-hip_mid['y']])
        vert = np.array([0,1])
        try:
            cos_a = np.dot(spine, vert) / np.linalg.norm(spine)
            angle = math.degrees(math.acos(abs(cos_a)))
            if angle > self.thresholds['back_straightness']:
                errors.append({'message': 'Espalda muy inclinada', 'severity': 'critical'})
        except:
            pass
        return errors

    def check_calf_raise_form(self, landmarks, current_stage):
        """
        Función principal para verificar forma de calf raises
        """
        all_err = []
        if landmarks and len(landmarks) >= 33:
            all_err.extend(self.check_knee_straightness(landmarks))
            all_err.extend(self.check_heel_lift(landmarks, current_stage))
            all_err.extend(self.check_back_straightness(landmarks))

            self.error_history.append(all_err)
            return self._get_consistent_errors()
        return []

    def _get_consistent_errors(self):
        if len(self.error_history) < 3:
            return list(self.error_history[-1]) if self.error_history else []
        counts = {}
        recent = list(self.error_history)[-5:]
        for frame in recent:
            for err in frame:
                key = err['message']
                if key not in counts:
                    counts[key] = {'count':0, 'error':err}
                counts[key]['count'] += 1
        consistent = [data['error'] for k,data in counts.items() if data['count']>=3]
        return consistent

    def reset_error_history(self):
        self.error_history.clear()