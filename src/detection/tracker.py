from collections import defaultdict
import cv2
from ultralytics import YOLO

model = YOLO('D:/university/Vlad/runs/detect/train2/weights/best.pt')
video_path = 'D:/university/Vlad/data/input_video/People.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = 'D:/university/Vlad/data/output_video/output_people.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Трекер для подсчета уникальных ID
tracker = {}
person_ids = set()
next_id = 0

def process_frame_with_tracking(frame):
    global next_id
    results = model.predict(frame)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = result.names[cls]

            # Проверяем, является ли объект человеком (COCO class 'person' == 0)
            if label == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Определяем центр объекта
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Проверяем, был ли человек уже задетектирован
                detected = False
                for tid, tcenter in tracker.items():
                    dist = ((center_x - tcenter[0])**2 + (center_y - tcenter[1])**2)**0.5
                    if dist < 50:  # Если объект близко к предыдущей позиции
                        tracker[tid] = (center_x, center_y)
                        detected = True
                        break

                # Если это новый человек
                if not detected:
                    tracker[next_id] = (center_x, center_y)
                    person_ids.add(next_id)
                    next_id += 1

                # Рисуем рамку вокруг человека
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 'Person', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Добавляем текст с уникальным числом людей
    cv2.putText(frame, f'Total People Detected: {len(person_ids)}', 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame_with_tracking(frame)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f'Загальна кількість людей: {len(person_ids)}')
