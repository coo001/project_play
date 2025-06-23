import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# 1. 모델 & 트래커 초기화
model = YOLO("yolov8s.pt")  # 스몰 모델 사용 (속도/정밀도 균형)
byte_tracker = sv.ByteTrack()  # ByteTrack 트래커
annotator = sv.BoxAnnotator(thickness=2)
colors = [
    (255, 0, 0),     # 빨강
    (0, 255, 0),     # 초록
    (0, 0, 255),     # 파랑
    (255, 255, 0),   # 노랑
    (255, 0, 255),   # 자홍
    (0, 255, 255),   # 청록
    (128, 0, 128),   # 보라
    (255, 165, 0),   # 주황
    (0, 128, 128),   # 청록-진함
    (128, 128, 0),   # 올리브
]

# 2. 동선 저장용 변수
track_histories: dict[int, list[tuple[int,int]]] = {}

# 3. 프레임 처리 콜백
def process_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    results = model(frame, conf=0.25)[0]
    detections = sv.Detections.from_ultralytics(results)
    tracked = byte_tracker.update_with_detections(detections)

    for item in tracked:
        print("item type:", type(item))
        print("item content:", item)
        break

    # 트랙킹된 객체 ID별 중심점 저장
    for bbox, _, _, tracker_id, class_id, _ in tracked:
        x1, y1, x2, y2 = bbox
        cx = int((float(x1) + float(x2)) / 2)
        cy = int((float(y1) + float(y2)) / 2)
        track_histories.setdefault(tracker_id, []).append((cx, cy))

    # 궤적 시각화
    overlay = frame.copy()
    for tid, pts in track_histories.items():
        if len(pts) > 1:
            color = colors[tid % len(colors)] 
            cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], False,
                          color=color, thickness=2)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 바운딩박스와 라벨 출력
    labels = [f"ID:{tid}" for tid in tracked.tracker_id]
    frame = annotator.annotate(scene=frame, detections=tracked)
    return frame

# 4. 비디오 로드 및 처리
source = "C:/Users/sukpo/OneDrive/바탕 화면/play_project/synthetic_people.mp4"      # 입력 영상 파일
target = "C:/Users/sukpo/OneDrive/바탕 화면/play_project/output_track.mp4"  # 결과 저장 파일

sv.process_video(source_path=source, target_path=target, callback=process_frame)
print(f"✅ 완료: {target} 에 저장되었습니다.")
