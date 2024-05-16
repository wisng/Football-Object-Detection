import sys
import cv2 as cv
sys.path.append('../')
from utils import measure_distance, get_foot_position

class SpeedAndDistance():
    def __init__(self) -> None:
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object == 'players':
                number_of_frames = len(object_tracks)
                for frame_num in range(0, number_of_frames, self.frame_window):
                    last_frame = min(frame_num + self.frame_window, number_of_frames-1)

                    for track_id, _ in object_tracks[frame_num].items():
                        if track_id not in object_tracks[last_frame]:
                            continue
                        start_position = object_tracks[frame_num][track_id]['position_transformed']
                        end_position = object_tracks[last_frame][track_id]['position_transformed']

                        if start_position is None or end_position is None:
                            continue

                        distance = measure_distance(start_position, end_position)
                        time = (last_frame - frame_num) / self.frame_rate
                        speed = distance / time
                        speed_km = speed * 3.6
                        
                        if object not in total_distance:
                            total_distance[object] = {}

                        if track_id not in total_distance[object]:
                            total_distance[object][track_id] = 0

                        total_distance[object][track_id] += distance

                        for frame_num_batch in range(frame_num, last_frame):
                            if track_id not in tracks[object][frame_num_batch]:
                                continue
                            tracks[object][frame_num_batch][track_id]['speed'] = speed_km
                            tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            for object, object_tracks in tracks.items():
                if object == 'players':
                    for _, track_info in object_tracks[frame_num].items():
                        if 'speed' in track_info:
                            speed = track_info.get('speed', None)
                            distance = track_info.get('distance', None)
                            if speed is None or distance is None:
                                continue

                            bbox = track_info.get('bbox', None)
                            position = get_foot_position(bbox)
                            position = list(position)
                            position[1] += 40
                            position = tuple(map(int, position))
                            cv.putText(frame, f'Speed: {speed:.2f} km/h', position, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            cv.putText(frame, f'Distance: {distance:.2f} m', (position[0], position[1] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_video_frames.append(frame)

        return output_video_frames
