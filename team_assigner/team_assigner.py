import cv2 as cv
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player in player_detections.items():
            bbox = player['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=0)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

        return player_colors

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2), :]
        kmeans = self.get_clustering_model(top_half_image)

        labels = kmeans.labels_
        labels = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        corner_clusters = [labels[0,0], labels[0,-1], labels[-1,0], labels[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1-non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def get_clustering_model(self, top_half_image):
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=0)
        image_2d = top_half_image.reshape(-1, 3)
        kmeans.fit(image_2d)
        return kmeans
    
    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id