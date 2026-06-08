from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ClipMatcher:
    def __init__(self):
        self.clip_db = {}

    def add_clip(self, path: str, vector):
        self.clip_db[path] = np.array(vector)

    def match(self, song_vector, gender_preference=None, top_k=20):
        """Bidirectional Tinder Match: Song needs ↔ Clip offers"""
        # TODO: Implement cosine similarity + gender vector boosting
        return []
