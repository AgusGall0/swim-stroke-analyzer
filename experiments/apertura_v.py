import cv2 as cv
import mediapipe as mp
import math

# --- 1. EL MOTOR TRIGONOMÉTRICO ---
def calcular_angulo(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    angulo = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angulo = abs(angulo)
    if angulo > 180.0:
        angulo = 360.0 - angulo
    return angulo

# Configuración de la API moderna (Tasks API)
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- IMPORTAMOS EL MAPA DIRECTAMENTE DE LA API ---
# Así evitamos escribir la lista manualmente
CONEXIONES = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_heavy.task"),
    running_mode=VisionRunningMode.VIDEO
)

with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv.VideoCapture('ejemplo.mp4')
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(frame_count * 1000 / fps)
        
        resultados = landmarker.detect_for_video(mp_image, timestamp_ms)

        if resultados.pose_landmarks:
            alto, ancho, _ = frame.shape
            
            for pose_landmarks in resultados.pose_landmarks:
                
                # --- A. Dibujar conexiones de forma dinámica ---
                for conexion in CONEXIONES:
                    # Extraemos el inicio y fin usando los atributos de la nueva API
                    idx_inicio = conexion.start if hasattr(conexion, 'start') else conexion[0]
                    idx_fin = conexion.end if hasattr(conexion, 'end') else conexion[1]
                    
                    p_inicio = pose_landmarks[idx_inicio]
                    p_fin = pose_landmarks[idx_fin]
                    
                    if p_inicio.visibility > 0.5 and p_fin.visibility > 0.5:
                        x1, y1 = int(p_inicio.x * ancho), int(p_inicio.y * alto)
                        x2, y2 = int(p_fin.x * ancho), int(p_fin.y * alto)
                        cv.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                
                # --- B. Dibujar articulaciones ---
                for landmark in pose_landmarks:
                    if landmark.visibility > 0.5:
                        x, y = int(landmark.x * ancho), int(landmark.y * alto)
                        cv.circle(frame, (x, y), 4, (0, 0, 255), -1)
                
                # --- C. Extraer y calcular ---
                hombro_der = pose_landmarks[12]
                codo_der = pose_landmarks[14]
                muneca_der = pose_landmarks[16]
                
                if hombro_der.visibility > 0.5 and codo_der.visibility > 0.5 and muneca_der.visibility > 0.5:
                    p1 = (int(hombro_der.x * ancho), int(hombro_der.y * alto))
                    p2 = (int(codo_der.x * ancho), int(codo_der.y * alto))
                    p3 = (int(muneca_der.x * ancho), int(muneca_der.y * alto))
                    
                    angulo_codo = calcular_angulo(p1, p2, p3)
                    
                    cv.putText(frame, str(int(angulo_codo)), p2, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        cv.imshow('Analizador de Crol - Telemetria', frame)
        if cv.waitKey(24) == ord('q'):
            break
        
        frame_count += 1

    cap.release()
cv.destroyAllWindows()