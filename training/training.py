import os
from ultralytics import YOLO

def train_and_test_yolov8(data_yaml_path, epochs):
    """
    Навчає попередньо натреновану модель YOLOv8 на користувацьких даних і зберігає результати тестування.

    Args:
        data_yaml_path (str): Шлях до YAML-файлу з набором даних.
        epochs (int): Кількість епох для перенавчання моделі.
        custom_save_dir (str): Папка для збереження результатів тестування.
    """
    print("=== Навчання та тестування моделі YOLOv8 ===")
    
    # Завантаження базової моеделі YOLOv8
    model = YOLO('yolov8n.pt')
    
    # Перенавчання моделі на користувацькому наборі даних
    model.train(data=data_yaml_path, epochs=epochs, imgsz=640,  device=0)  
     
    # Виконуємо тестування на тестовому наборі даних
    results = model.val(data=data_yaml_path, save=True, save_json=True)

     
    return results

if __name__ == "__main__":
    # Шлях до конфігураційного файлу YAML
    data_yaml_path = "D:/university/Vlad/training/People-Detection-8/data.yaml"
    
    # Запуск перенавчання і тестування моделі YOLOv8
    train_and_test_yolov8(data_yaml_path, epochs=50)
