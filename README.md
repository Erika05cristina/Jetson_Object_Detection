
# Detección de Objetos en Tiempo Real con Jetson Orin Nano y ESP32-CAM

Este proyecto implementa un sistema de detección de objetos en tiempo real utilizando una Jetson Orin Nano y un modelo YOLOv11. El video se transmite de forma inalámbrica desde una ESP32-CAM, se procesa con OpenCV y PyTorch, y las detecciones se muestran en vivo.

---

## Descripción del Sistema

- **Jetson Orin Nano**: Dispositivo de cómputo en el borde para ejecutar el modelo localmente con aceleración GPU.
- **YOLOv11**: Modelo de detección rápido, ligero y preciso.
- **ESP32-CAM**: Módulo de cámara con Wi-Fi que transmite video por red.
- **Python + OpenCV**: Usado para el procesamiento y visualización del video en tiempo real.
- **PyTorch**: Framework utilizado para cargar y ejecutar el modelo YOLO.

---

## Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/yourusername/jetson-object-detection.git
cd jetson-object-detection
```

### 2. Crear Entorno (Se recomienda Conda)

```bash
conda create -n yolo python=3.10
conda activate yolo
pip install -r requirements.txt
```

### 3. Instalar Dependencias del Sistema

Asegúrate de que tu Jetson tenga:

- CUDA y cuDNN instalados
- PyTorch con soporte para GPU
- OpenCV compilado con GStreamer y CUDA

---

## Configuración de la ESP32-CAM

- Cargar el firmware de streaming (por ejemplo, con Arduino IDE y MJPEG).
- Conectarla a la misma red Wi-Fi que la Jetson.
- Anotar la URL del stream (ejemplo: `http://192.168.0.101:81/stream`).

---

## Ejecutar la Aplicación

Actualizar la URL del stream en el script si es necesario:

```python
video_url = "http://192.168.x.x:81/stream"
```

Luego ejecutar:

```bash
python examples/video_stream.py
```

El modelo procesará los cuadros en tiempo real y mostrará los resultados en una ventana de OpenCV.

---

## Carga del Modelo

El modelo YOLOv11 se carga con:

```python
from ultralytics import YOLO
self.object_model = YOLO(model_path).to(torch.device('cuda'))
```

Se puede reemplazar por cualquier otra variante de YOLO soportada por [Ultralytics](https://docs.ultralytics.com).

---

## Rendimiento

- **FPS**: Entre 20 y 25 FPS en Jetson Orin Nano
- **Uso de GPU**: Inferencia acelerada por CUDA
- **Ancho de banda**: Bajo, gracias a MJPEG
- **Latencia**: Baja, adecuada para tiempo real

---

## Dataset y Entrenamiento (Opcional)

Es posible entrenar un modelo personalizado en PyTorch y exportarlo para usarlo en este sistema.

---

## Tecnologías Usadas

- Jetson Orin Nano
- YOLOv11
- ESP32-CAM
- Python 3.10
- OpenCV 4.x (con CUDA)
- PyTorch
- Ultralytics

---

## Demostración

<p align="center">
  <img src="demo/demo_gif.gif" alt="demo" width="600"/>
</p>

---

## Estructura del Proyecto

```
jetson-object-detection/
├── examples/
│   └── video_stream.py
├── processor/
│   └── main.py
├── models/
│   └── yolov11.pt
├── requirements.txt
└── README.md
```

---

## Autores

- Erika Villa
- Jorge Márquez


