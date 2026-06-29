import numpy as np

class Normalizer:
    def __init__(self, X, min_max=None):

        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)
        if min_max is not None:
            self.mins = min_max[0]
            self.maxs = min_max[1]

    def __repr__(self):
        return f"Normalizer(mins={self.mins}, maxs={self.maxs})"

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, x):
        raise NotImplementedError()

    def unnormalize(self, x):
        raise NotImplementedError()

class LimitsNormalizer(Normalizer):
    def normalize(self, x):
        x = (x - self.mins) / (self.maxs - self.mins)
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=0):
        if x.max() > 1 + eps or x.min() < -1 - eps:
            x = np.clip(x, -1, 1)

        x = (x + 1) / 2.
        return x * (self.maxs - self.mins) + self.mins

class WallLocLimitsNormalizer(LimitsNormalizer):
    def __init__(self, X, maze_size):
        assert len(X.shape) == 2, "X should be [num_walls, 2]"
        self.X = X.astype(np.float32)

        self.mins = np.array([0.5, 0.5], dtype=np.float32)
        self.maxs = np.array(maze_size, dtype=np.float32) - 0.5

class TrajectoryLimitsNormalizer(LimitsNormalizer):
    def __init__(self, X, maze_size):
        assert len(X.shape) == 2, "X should be [horizon, 2]"
        self.X = X.astype(np.float32)

        self.mins = np.array([0.0, 0.0], dtype=np.float32)
        self.maxs = np.array(maze_size, dtype=np.float32)

class GoalLimitsNormalizer(LimitsNormalizer):
    def __init__(self, X, maze_size):
        assert len(X.shape) == 1, "X should be [2]"
        self.X = X.astype(np.float32)

        self.mins = np.array([0.0, 0.0], dtype=np.float32)
        self.maxs = np.array(maze_size, dtype=np.float32)

if __name__ == "__main__":

    wall_data = np.random.rand(6, 2)
    wall_normalizer = WallLocLimitsNormalizer(wall_data, maze_size=(5,5))
    normalized_walls = wall_normalizer.normalize(wall_data)

    traj_data = np.random.rand(48, 2)
    traj_normalizer = TrajectoryLimitsNormalizer(traj_data, maze_size=(5,5))
    normalized_traj = traj_normalizer.normalize(traj_data)
