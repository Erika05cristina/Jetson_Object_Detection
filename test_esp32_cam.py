import requests
import cv2
import numpy as np

url = "http://192.168.18.154:81/stream"

# Abrimos la conexión HTTP al stream MJPEG
stream = requests.get(url, stream=True)

if stream.status_code != 200:
    print("No se pudo conectar al stream")
    exit()

bytes_data = bytes()
for chunk in stream.iter_content(chunk_size=1024):
    bytes_data += chunk
    a = bytes_data.find(b'\xff\xd8')   # start of JPEG
    b = bytes_data.find(b'\xff\xd9')   # end of JPEG

    if a != -1 and b != -1 and b > a:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]
        # Decodificamos imagen
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imshow("ESP32-CAM stream", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()
