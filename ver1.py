import cv2
import os
import numpy as np
from datetime import datetime
import multiprocessing
from multiprocessing import Queue
from ultralytics import solutions

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def process_single(link: str, id_process: int, results_queue: Queue) -> None:
    speed_tool = solutions.SpeedEstimator(
        model = "best.pt",
        verbose = False,
    )    
        
    time_pre = datetime.now()
    
    cap = cv2.VideoCapture(link)
    if not cap.isOpened():
        print(f"[{id_process}] Không thể mở video: {link}")
        return

    
    count_car_display = 0
    count_motor_display = 0

    list_count_car = []
    list_count_motor = []
    
    speed_car_display = 0
    speed_motor_display = 0

    list_speed_car = []
    list_speed_motor = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (400, 300))
        frame_predict = np.ascontiguousarray(frame[135:, 60:])
        
        time_now = datetime.now()
        delta_time = (time_now - time_pre).total_seconds()
        
        if delta_time >= 30:
            time_pre = time_now
            
            if list_count_car:
                count_car_display = sum(list_count_car) // len(list_count_car)
            else:
                count_car_display = 0

            if list_speed_car:
                speed_car_display = sum(list_speed_car) // len(list_speed_car)
            else:
                speed_car_display = 0

            if list_count_motor:
                count_motor_display = sum(list_count_motor) // len(list_count_motor)
            else:
                count_motor_display = 0

            if list_speed_motor:
                speed_motor_display = sum(list_speed_motor) // len(list_speed_motor)
            else:
                speed_motor_display = 0
            

            results_queue.put({
                "name": link,
                "count_car": count_car_display,
                "count_motor": count_motor_display,
                "speed_car": speed_car_display,
                "speed_motor": speed_motor_display,
                
            })

            list_count_car.clear()
            list_count_motor.clear()
            list_speed_car.clear()
            list_speed_motor.clear()
            
        
        frame_predict_cp = frame_predict.copy()
        speed_tool.process(frame_predict_cp)
        
        speeds = speed_tool.spd    
        ids = speed_tool.track_data.id.cpu().numpy().astype(int)
        boxes = speed_tool.track_data.xyxy.cpu().numpy().astype(int)
        classes = speed_tool.track_data.cls.cpu().numpy().astype(int)
        
        count_car = np.count_nonzero(classes == 0)    
        count_motor = np.count_nonzero(classes == 1)
        list_count_car.append(count_car)
        list_count_motor.append(count_motor)
        
        car_ids = ids[classes == 0]
        motor_ids = ids[classes == 1]
        list_speed_car.extend([speeds[tid] for tid in car_ids if tid in speeds])
        list_speed_motor.extend([speeds[tid] for tid in motor_ids if tid in speeds])

                
        if ids is not None:
            for i, box in enumerate(boxes):
                track_id = ids[i]
                class_id = classes[i]
                speed_id = speeds.get(track_id, 0)  # 0 là giá trị mặc định nếu không tìm thấy

                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                      
                label = f"{str(speed_id)} km/h" if class_id == 0 else f"{str(speed_id)} km/h"
                color = (0, 0, 255) if class_id == 1 else (255, 0, 0)
                
                cv2.putText(frame_predict, label, (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                cv2.circle(frame_predict, (cx, cy), 5, color, -1)

        frame[135:, 60:] = frame_predict
        
        cv2.rectangle(frame, (60, 135), (400, 300), (0, 255, 255), 2)
        cv2.putText(frame, f"Xe may: {count_motor_display} xe, Vtb = {speed_motor_display} km/h", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"O to: {count_car_display} xe, Vtb = {speed_car_display} km/h", (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow(f"{link[:-4]}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_multi(links: list, results_queue: Queue) -> None:
    processes = []
    for id, link in enumerate(links):
        p = multiprocessing.Process(
            target=process_single,
            args=(link, id, results_queue),
            daemon=True
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    video_links = ["video test/vid5.mp4", "video test/vid8.mp4", "video test/vid6.mp4", "video test/vid9.mp4", "video test/vid10.mp4"]
    results_queue = Queue()

    process = multiprocessing.Process(target=process_multi, args=(video_links, results_queue))
    process.start()

    try:
        while True:
            data = results_queue.get(timeout= 40)  # chờ kết quả từ bất kỳ tiến trình nào
            print(
                f"\033[93m\t\t\t\t [{data['name']}]:\033[0m\n"
                f"\033[94m Ô tô: {data['count_car']} xe \t\t\t\t\t\t\t\t tốc độ trung bình: {data['speed_car']} km/h\033[0m\n"
                f"\033[91m Xe máy: {data['count_motor']} xe \t\t\t\t\t\t\t\t tốc độ trung bình: {data['speed_motor']} km/h\033[0m"
            )

            print("__________________________________________________________________________________________________________")
    except:
        print("Không còn dữ liệu hoặc kết thúc xử lý.")
        
    if not process.is_alive():
        exit()