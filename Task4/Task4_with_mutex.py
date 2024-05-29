import cv2
import threading
import queue
import time
import logging
import argparse
import os
from threading import Event, Lock
from queue import Queue

# Создаем каталог log, если он не существует
if not os.path.exists('log'):
    os.makedirs('log')

# Настроим логирование
logging.basicConfig(filename='log/application.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def log_message(message, level=logging.INFO):
    logging.log(level, message)

# Базовый класс для датчиков
class Sensor:
    def __init__(self, name, frequency):
        self.name = name
        self.frequency = frequency
        self.queue = Queue()
        self.stop_event = Event()

    def run(self):
        while not self.stop_event.is_set():
            data = self.get_data()
            self.queue.put(data)
            time.sleep(1 / self.frequency)

    def get_data(self):
        raise NotImplementedError

    def stop(self):
        self.stop_event.set()

# Класс для камеры
class SensorCam(Sensor):
    def __init__(self, camera_index, resolution):
        super().__init__("Camera", 30)  # Устанавливаем частоту для камеры на 30 Гц
        self.resolution = resolution
        try:
            log_message(f"Attempting to open camera with index {camera_index}")
            self.cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not self.cam.isOpened():
                raise ValueError("Cannot open camera")
            log_message(f"Camera opened successfully with index {camera_index}")
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        except Exception as e:
            log_message(f"Error initializing camera: {e}", level=logging.ERROR)
            raise

    def run(self):
        while not self.stop_event.is_set():
            ret, frame = self.cam.read()
            if not ret:
                log_message("Error reading frame from camera", level=logging.ERROR)
                time.sleep(1 / self.frequency)
                continue
            self.queue.put(frame)
            time.sleep(1 / self.frequency)

    def __del__(self):
        if self.cam.isOpened():
            self.cam.release()
            log_message("Camera released")

# Новый класс SensorX
class SensorX(Sensor):
    """Sensor X"""

    def __init__(self, delay: float, name: str, frequency: int):
        super().__init__(name, frequency)
        self._delay = delay
        self._data = 0

    def get_data(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data

    def sensor_x_work(self):
        while not self.stop_event.is_set():
            data = self.get_data()
            self.queue.put(data)

# Класс для отображения изображения
class WindowImage:
    def __init__(self, frequency):
        self.frequency = frequency
        self.stop_event = Event()

    def show(self, img):
        cv2.imshow('Window', img)
        if cv2.waitKey(int(1000 / self.frequency)) & 0xFF == ord('q'):
            self.stop_event.set()

    def __del__(self):
        cv2.destroyAllWindows()
        log_message("Window destroyed")

def main(camera_index, resolution, display_frequency):
    try:
        log_message("Starting up...")
        log_message("Parsing arguments")
        
        sensor_data = [
            {"name": "sensor0", "delay": 0.01, "frequency": 100},  # 100Hz
            {"name": "sensor1", "delay": 0.1, "frequency": 10},    # 10Hz
            {"name": "sensor2", "delay": 1, "frequency": 1}        # 1Hz
        ]

        sensors = [
            SensorX(data["delay"], data["name"], data["frequency"])
            for data in sensor_data
        ]

        sensor_texts_lock = Lock()
        sensor_texts = [""] * len(sensors)

        for i, sensor in enumerate(sensors, start=1):
            log_message(f"Starting Sensor X {i} thread...", level=logging.INFO)
            threading.Thread(target=sensor.sensor_x_work).start()

        log_message("Starting Sensor Cam thread...", level=logging.INFO)
        camera = SensorCam(camera_index, resolution)
        camera_thread = threading.Thread(target=camera.run)
        camera_thread.start()

        log_message("Starting Window Image...", level=logging.INFO)
        window = WindowImage(display_frequency)

        log_message("Starting Frame Assembly...", level=logging.INFO)

        while not window.stop_event.is_set():
            frame = camera.queue.get()
            if frame is not None:
                sensor_values = [sensor.queue.queue[-1] if not sensor.queue.empty() else None for sensor in sensors]
                log_message(sensor_values)
                
                # Синхронизируем доступ к sensor_texts
                with sensor_texts_lock:
                    for i, (sensor, value) in enumerate(zip(sensors, sensor_values)):
                        if value is not None:
                            sensor_texts[i] = f"Data from {sensor.name}: {value}"
                        else:
                            sensor_texts[i] = f"No data from {sensor.name}"
                    
                    # Накладываем данные на изображение
                    for i, text in enumerate(sensor_texts):
                        y = 30 + i * 30
                        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Показываем изображение
                window.show(frame)

    except Exception as e:
        log_message(f"Error in main loop: {e}", level=logging.ERROR)

    finally:
        log_message("Cleaning up...", level=logging.INFO)
        
        for sensor in sensors:
            sensor.stop()
        camera.stop()

        for thread in threading.enumerate():
            if thread != threading.current_thread():
                thread.join()
        camera_thread.join()

        log_message("Main loop finished", level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensor and Camera Data Display")
    parser.add_argument("--camera_index", type=int, default=0, help="Index of the camera device")
    parser.add_argument("--resolution", type=str, default="1280x720", help="Resolution of the camera")
    parser.add_argument("--frequency", type=int, default=30, help="Display frequency in Hz")
    args = parser.parse_args()

    resolution = tuple(map(int, args.resolution.split('x')))
    main(args.camera_index, resolution, args.frequency)
