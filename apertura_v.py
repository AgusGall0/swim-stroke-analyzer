import cv2 as cv
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

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
            print(f"Frame {frame_count}: Detectados {len(resultados.pose_landmarks[0])} puntos de referencia.")
        else:
            print(f"Frame {frame_count}: No se detectó ninguna persona.")

        cv.imshow('Analizador de Crol', frame)
        if cv.waitKey(24) == ord('q'):
            
            break
        
        frame_count += 1

    cap.release()
cv.destroyAllWindows()