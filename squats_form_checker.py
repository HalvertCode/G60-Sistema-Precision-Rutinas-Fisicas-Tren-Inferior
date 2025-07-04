import numpy as np
import math
from collections import deque

class SquatsFormChecker:
    def __init__(self, history_size=10):
        """
        Inicializa el verificador de forma de sentadillas
        
        Args:
            history_size: Número de frames a mantener en historial
        """
        self.history_size = history_size
        self.error_history = deque(maxlen=history_size)

        self.thresholds = {
            'knee_angle_min': 70,       # Rodilla suficientemente flexionada en la bajada
            'knee_angle_max': 160,      # Rodilla extendida en la subida
            'hip_angle_min': 60,        # Cadera suficientemente flexionada
            'hip_angle_max': 170,       # Cadera extendida
            'back_straightness': 30,    # Inclinación máxima permitida en la espalda
            'symmetry_diff_max': 20     # Diferencia de ángulos entre lados izquierdo y derecho
        }

    def calculate_angle(self, p1, p2, p3):
        try:
            v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
            v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            return math.degrees(angle)
        except:
            return 0

    def check_knees(self, landmarks):
        errors = []

        left_knee_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        right_knee_angle = self.calculate_angle(landmarks[24], landmarks[26], landmarks[28])

        if left_knee_angle < self.thresholds['knee_angle_min']:
            errors.append({'message': 'Rodilla izq muy flexionada', 'severity': 'critical'})
        if right_knee_angle < self.thresholds['knee_angle_min']:
            errors.append({'message': 'Rodilla der muy flexionada', 'severity': 'critical'})

        angle_diff = abs(left_knee_angle - right_knee_angle)
        if angle_diff > self.thresholds['symmetry_diff_max']:
            errors.append({'message': 'Rodillas desalineadas', 'severity': 'warning'})

        return errors

    def check_hips(self, landmarks):
        errors = []

        left_hip_angle = self.calculate_angle(landmarks[11], landmarks[23], landmarks[25])
        right_hip_angle = self.calculate_angle(landmarks[12], landmarks[24], landmarks[26])
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2

        if avg_hip_angle < self.thresholds['hip_angle_min']:
            errors.append({'message': 'Cadera no suficientemente baja', 'severity': 'warning'})
        elif avg_hip_angle > self.thresholds['hip_angle_max']:
            errors.append({'message': 'Cadera demasiado extendida', 'severity': 'warning'})

        return errors

    def check_back_straightness(self, landmarks):
        errors = []

        shoulder_mid = {
            'x': (landmarks[11]['x'] + landmarks[12]['x']) / 2,
            'y': (landmarks[11]['y'] + landmarks[12]['y']) / 2
        }
        hip_mid = {
            'x': (landmarks[23]['x'] + landmarks[24]['x']) / 2,
            'y': (landmarks[23]['y'] + landmarks[24]['y']) / 2
        }

        spine_vector = np.array([shoulder_mid['x'] - hip_mid['x'], 
                                 shoulder_mid['y'] - hip_mid['y']])
        vertical = np.array([0, 1])

        try:
            cos_angle = np.dot(spine_vector, vertical) / np.linalg.norm(spine_vector)
            spine_angle = math.degrees(math.acos(abs(cos_angle)))
            if spine_angle > self.thresholds['back_straightness']:
                errors.append({'message': 'Espalda muy inclinada', 'severity': 'critical'})
        except:
            pass

        return errors

    def check_squat_form(self, landmarks, current_stage):
        """
        Verifica la forma del squat.
        
        Args:
            landmarks: Lista de landmarks de MediaPipe
            current_stage: Etapa del ejercicio ('down', 'up', etc.)

        Returns:
            Lista de errores detectados
        """
        all_errors = []

        if landmarks and len(landmarks) >= 33:
            all_errors.extend(self.check_knees(landmarks))
            all_errors.extend(self.check_hips(landmarks))
            all_errors.extend(self.check_back_straightness(landmarks))

            self.error_history.append(all_errors)
            consistent = self._get_consistent_errors()
            return consistent

        return []

    def _get_consistent_errors(self):
        if len(self.error_history) < 3:
            return list(self.error_history[-1]) if self.error_history else []

        error_counts = {}
        recent_errors = list(self.error_history)[-5:]

        for frame_errors in recent_errors:
            for error in frame_errors:
                key = error['message']
                if key not in error_counts:
                    error_counts[key] = {'count': 0, 'error': error}
                error_counts[key]['count'] += 1

        consistent_errors = []
        for key, data in error_counts.items():
            if data['count'] >= 3:
                consistent_errors.append(data['error'])

        return consistent_errors

    def reset_error_history(self):
        self.error_history.clear()
