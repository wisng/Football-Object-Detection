from utils import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner
import cv2 as cv
from player_ball_assignment import PlayBallAssignment
from camera_movement import CameraMovement
import numpy as np
from view_transformer import ViewTransformer
from speed_and_distance import SpeedAndDistance
def main():
    video_frames = read_video('input_videos/08fd33_4.mp4')

    tracker = Tracker('runs/detect/train3/weights/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    tracker.add_position_to_tracks(tracks)

    camera_movement = CameraMovement(video_frames[0])
    camera_movement_per_frame = camera_movement.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stubs.pkl')

    camera_movement.adjust_position_to_tracks(tracks, camera_movement_per_frame)

    view_transormer = ViewTransformer()
    view_transormer.add_transform_position_to_tracks(tracks)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame], track['bbox'], player_id)
            tracks['players'][frame][player_id]['team'] = team
            tracks['players'][frame][player_id]['team_color'] = team_assigner.team_colors[team]

    tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])

    speed_and_distance = SpeedAndDistance()
    speed_and_distance.add_speed_and_distance(tracks)

    player_ball_assignment = PlayBallAssignment()
    team_ball_control = []

    for frame, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame][1]['bbox']
        assigned_player = player_ball_assignment.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame][assigned_player]['team'])
        else:
            if len(team_ball_control) == 0:
                team_ball_control.append(-1)
            else:
                team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)

    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     cv.imwrite(f'output_videos/{track_id}.jpg', cropped_image)
    #     break

    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    output_video_frames = camera_movement.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    
    output_video_frames = speed_and_distance.draw_speed_and_distance(output_video_frames, tracks)

    save_video(output_video_frames, 'output_videos/output.avi')

if __name__ == '__main__':
    main()