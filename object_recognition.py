import cv2
import os
from ultralytics import YOLO
from tqdm import tqdm
from multiprocessing import Process, Manager

# ---------------- CONFIG ----------------
NUM_WORKERS = 4
FRAME_SKIP = 5
openvino_model_folder = "yolo11n_openvino_model/"
highlight_objects = []
model_path_default = "yolo11n.pt"

# ---------------- Utilities ----------------
def detect_objects_in_frame(frame, model, objects_of_interest):
    objs = []
    try:
        results = model(frame, verbose=False, imgsz=640)
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    if conf > 0.5 and cls_name in objects_of_interest:
                        objs.append(cls_name)
    except Exception as e:
        print(f"⚠️ Error: {e}")
    return objs

def get_video_segments(video_path, num_segments):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    frames_per_segment = total_frames // num_segments
    segments = []
    for i in range(num_segments):
        start = i * frames_per_segment
        end = (i + 1) * frames_per_segment if i < num_segments - 1 else total_frames
        segments.append((start, end))
    return segments, fps

# ---------------- Worker ----------------
def worker_process(video_path, start_frame, end_frame, objects_of_interest, return_dict, worker_id):
    model = YOLO(model_path_default)  # each worker loads its own model
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    total_frames = end_frame - start_frame

    sec_objects = {}
    pbar = tqdm(total=total_frames, desc=f"Worker {worker_id}", position=worker_id)
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SKIP == 0:
            objs = detect_objects_in_frame(frame, model, objects_of_interest)
            if objs:
                sec = int(frame_idx / 30)  # assume fps ~30, adjust if needed
                if sec not in sec_objects:
                    sec_objects[sec] = []
                sec_objects[sec].extend(objs)
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return_dict[worker_id] = sec_objects

# ---------------- Main ----------------
if __name__ == "__main__":
    test_video = "test_video.mp4"
    if not os.path.exists(test_video):
        print("⚠️ Test video not found")
        exit()

    segments, fps = get_video_segments(test_video, NUM_WORKERS)
    manager = Manager()
    return_dict = manager.dict()
    processes = []

    for i, seg in enumerate(segments):
        p = Process(target=worker_process, args=(test_video, seg[0], seg[1], highlight_objects, return_dict, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge results
    final_objects = {}
    for sec_objs in return_dict.values():
        for sec, objs in sec_objs.items():
            if sec not in final_objects:
                final_objects[sec] = []
            final_objects[sec].extend(objs)

    # Summary
    total_detections = sum(len(v) for v in final_objects.values())
    print(f"✅ Total seconds with objects: {len(final_objects)}, total detections: {total_detections}")
