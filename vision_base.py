import cv2

print("1. Librería cargada. Intentando encender la cámara...")

# Intentamos acceder a la cámara principal
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: No se pudo abrir la cámara.")
    print("Pistas: ¿Está tapada? ¿La está usando otra app? ¿Windows le dio permisos?")
else:
    print("✅ ¡Luz verde! Cámara encendida. Abriendo ventana... (Presiona 'q' para salir)")
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