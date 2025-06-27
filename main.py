import cv2
from tool import tool
import multiprocessing

def processing_single(path: str) -> None:
    cam = cv2.VideoCapture(path)
    t = tool(model_path= "best.pt", time_step= 30, is_draw= True)
    
    while True:
        r, cap = cam.read()
        
        t.process(cap)
        
        if t.result and t.delta_time >= t.time_step:
            print(t.result)
        
        frame = cv2.resize(t.frame_output, (600, 400))
        cv2.imshow('out', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':        
    video_links = ["video test/vid9.mp4", "video test/vid5.mp4"]

    processes = []
    for link in video_links:
        p = multiprocessing.Process(
            target= processing_single,
            args= (link, ),
            daemon= True
        )
        processes.append(p)
        p.start()
        

    for p in processes:
        p.join()