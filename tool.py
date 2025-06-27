import cv2
import os
import numpy as np
from datetime import datetime
from ultralytics import solutions

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class tool:
    def __init__(self, model_path : str, time_step: int, is_draw = True, device= 'cpu'):
        self.speed_tool = solutions.SpeedEstimator(
            model = model_path,
            verbose = False,
            device = device
        )
        self.count_car_display = 0
        self.list_count_car = []
        self.speed_car_display = 0
        self.list_speed_car = []
        
        self.count_motor_display = 0
        self.list_count_motor = []
        self.speed_motor_display = 0
        self.list_speed_motor = []
    
        self.time_pre = datetime.now()
        self.result = None
        self.frame_output = None
        self.time_step = time_step
        self.frame_predict = None
        self.is_draw = is_draw
        self.delta_time = 0
    def update_data(self, time_step: int):
        time_now = datetime.now()
        self.delta_time = (time_now - self.time_pre).total_seconds()
        
        if self.delta_time >= time_step:
            self.time_pre = time_now
            
            if self.list_count_car:
                self.count_car_display = sum(self.list_count_car) // len(self.list_count_car)
            else:
                self.count_car_display = 0

            if self.list_speed_car:
                self.speed_car_display = sum(self.list_speed_car) // len(self.list_speed_car)
            else:
                self.speed_car_display = 0

            if self.list_count_motor:
                self.count_motor_display = sum(self.list_count_motor) // len(self.list_count_motor)
            else:
                self.count_motor_display = 0

            if self.list_speed_motor:
                self.speed_motor_display = sum(self.list_speed_motor) // len(self.list_speed_motor)
            else:
                self.speed_motor_display = 0
            

            self.result = {
                "count_car": self.count_car_display,
                "count_motor": self.count_motor_display,
                "speed_car": self.speed_car_display,
                "speed_motor": self.speed_motor_display,
                
            }

            self.list_count_car.clear()
            self.list_count_motor.clear()
            self.list_speed_car.clear()
            self.list_speed_motor.clear()
            
    def process(self, frame_input) -> None:   
        frame_in = cv2.resize(frame_input.copy(), (400, 300))
        self.frame_output = frame_in.copy()
        self.frame_predict = np.ascontiguousarray(frame_in[135:, 60:])
        
        frame_predict_cp = self.frame_predict.copy()
        self.speed_tool.process(frame_predict_cp)
        
        self.speeds = self.speed_tool.spd    
        self.ids = self.speed_tool.track_data.id.cpu().numpy().astype(int)
        self.boxes = self.speed_tool.track_data.xyxy.cpu().numpy().astype(int)
        self.classes = self.speed_tool.track_data.cls.cpu().numpy().astype(int)
    
        count_car = np.count_nonzero(self.classes == 0)    
        count_motor = np.count_nonzero(self.classes == 1)
        self.list_count_car.append(count_car)
        self.list_count_motor.append(count_motor)
        
        car_ids = self.ids[self.classes == 0]
        motor_ids = self.ids[self.classes == 1]
        self.list_speed_car.extend([self.speeds[tid] for tid in car_ids if tid in self.speeds])
        self.list_speed_motor.extend([self.speeds[tid] for tid in motor_ids if tid in self.speeds])

        # Update
        self.update_data(self.time_step)
        
        if self.is_draw:
            self.draw()
    def draw(self):
        if self.ids is not None:
            for i, box in enumerate(self.boxes):
                track_id = self.ids[i]
                class_id = self.classes[i]
                speed_id = self.speeds.get(track_id, 0)  # 0 là giá trị mặc định nếu không tìm thấy

                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
                label = f"{str(speed_id)} km/h" if class_id == 0 else f"{str(speed_id)} km/h"
                color = (0, 0, 255) if class_id == 1 else (255, 0, 0)
                
                cv2.putText(self.frame_predict, label, (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 0), 1)
                cv2.circle(self.frame_predict, (cx, cy), 2, color, -1)

        self.frame_output[135:, 60:] = self.frame_predict
        
        cv2.rectangle(self.frame_output, (60, 135), (400, 300), (0, 255, 255), 2)
        cv2.putText(self.frame_output, f"Xe may: {self.count_motor_display} xe, Vtb = {self.speed_motor_display} km/h", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        cv2.putText(self.frame_output, f"O to: {self.count_car_display} xe, Vtb = {self.speed_car_display} km/h", (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2)

cam = cv2.VideoCapture("vid9.mp4")

t = tool(model_path= "best.pt", time_step= 30, is_draw= True)

while True:
    r, cap = cam.read()
    
    t.process(cap)
    
    if t.result and t.delta_time >= t.time_step:
        print(t.result)
        
    cv2.imshow('out', t.frame_output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
