import cv2
import mediapipe as mp


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_full.task"),
    running_mode=VisionRunningMode.IMAGE)
with PoseLandmarker.create_from_options(options) as landmarker:

# Intentamos acceder a la cámara principal
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    while True:
        # Leer el video frame por frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Se cortó la señal de la cámara.")
            break
        
        # Mostrar la imagen en una ventana
        cv2.imshow("Prueba de Camara Pura", frame)
        
        # Esperar a que presiones 'q' para cerrar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Limpieza final
cap.release()
cv2.destroyAllWindows()
print("3. Programa terminado correctamente.")