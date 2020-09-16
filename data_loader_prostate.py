import numpy as np
import random
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(
        self,
        data,
        batch_size,
        dims=3,
        shuffle=True,
        rotate=45,
        displace=1,
        deform=0,
        shuffle_points=False,
        kept_points=2048,
        slices=False,
        sweeps=0,
    ):
        self.data = data
        self.batch_size = batch_size
        self.dims = dims

        self.shuffle = shuffle
        self.rotate = rotate
        self.displace = displace
        self.deform = deform
        self.shuffle_points = shuffle_points
        self.kept_points = kept_points
        self.slices = slices
        self.sweeps = sweeps

        self.nb_sample = self.data.shape[1]
        self.indexes = np.arange(self.nb_sample)

    def __len__(self):
        return int(np.floor(self.nb_sample / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        data_temp = [[self.data[0][k], self.data[1][k]] for k in indexes]

        # Generate augmented batch data
        X, y = self.__generator(data_temp)

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __generator(self, data):
        X1 = []
        X2 = []
        X3 = []
        Y = []
        eps = 1e-7
        for fixed, moving in data:
            
            to_reg = moving
            ground_truth = fixed

            # Deform.
            if self.deform:
                sigma = np.random.normal()
                c = np.random.normal(loc=0, scale=self.deform, size=(10, 3))
                x = np.random.uniform(low=-1, high=1, size=(10, 3))
                k = compute_RBF(x, moving, sigma)
                moving = np.matmul(k.T, c) + moving
                moving_mean = np.mean(moving, axis=0)
                moving = moving - moving_mean

            # Rotate, translate.
            y, p, r = ypr_rand(-self.rotate, self.rotate)
            R = e2r(y, p, r)
            d = d_rand(-self.displace, self.displace)
            T = get_T(R, d)
            moving_with_ones = np.ones((moving.shape[0], 4))
            moving_with_ones[:, :-1] = moving
            moving = np.dot(T, moving_with_ones.T).T[:, :-1]

            if self.dims == 4:
                moving_with_ones = np.ones((moving.shape[0], 4))
                moving_with_ones[:, :-1] = moving
                moving = moving_with_ones
                to_reg_with_ones = np.ones((to_reg.shape[0], 4))
                to_reg_with_ones[:, :-1] = to_reg
                to_reg = to_reg_with_ones
                fixed_with_ones = np.ones((fixed.shape[0], 4))
                fixed_with_ones[:, :-1] = fixed
                fixed = fixed_with_ones
                gt_with_ones = np.ones((ground_truth.shape[0], 4))
                gt_with_ones[:, :-1] = ground_truth
                ground_truth = gt_with_ones

            if self.shuffle_points:
                fixed = fixed[np.random.permutation(fixed.shape[0]), :]
                moving = moving[np.random.permutation(moving.shape[0]), :]
                to_reg = to_reg[np.random.permutation(to_reg.shape[0]), :]
                ground_truth = ground_truth[np.random.permutation(ground_truth.shape[0]), :]

            if self.slices:
                fixed_X = fixed[fixed[:, 0] <= 0.02, :]
                fixed_X = fixed_X[fixed_X[:, 0] >= -0.02, :]
                fixed_Y = fixed[fixed[:, 1] <= 0.02, :]
                fixed_Y = fixed_Y[fixed_Y[:, 1] >= -0.02, :]
                fixed = np.concatenate((fixed_X, fixed_Y), axis=0)

            if self.sweeps:
                thresh = 0.02
                swept_prostate = np.array([])

                TRUS_tip = np.array([0, 1, -1])
                TRUS_end = np.array([0, -1, -1])

                if self.sweeps == 1:
                    P = np.array([0, 0, 1])
                    a, b, c = np.cross(TRUS_tip - TRUS_end, TRUS_tip - P)
                    d = -(a * P[0] + b * P[1] + c * P[2])
                    for point in fixed:
                        if point_plane_distance(point, a, b, c, d) < thresh:
                            swept_prostate = np.append(swept_prostate, point)

                elif self.sweeps == 2:
                    P = np.array([-0.5, 0, 1])
                    a, b, c = np.cross(TRUS_tip - TRUS_end, TRUS_tip - P)
                    d = -(a * P[0] + b * P[1] + c * P[2])
                    for point in fixed:
                        if point_plane_distance(point, a, b, c, d) < thresh:
                            swept_prostate = np.append(swept_prostate, point)
                    P = np.array([0.5, 0, 1])
                    a, b, c = np.cross(TRUS_tip - TRUS_end, TRUS_tip - P)
                    d = -(a * P[0] + b * P[1] + c * P[2])
                    for point in fixed:
                        if point_plane_distance(point, a, b, c, d) < thresh:
                            swept_prostate = np.append(swept_prostate, point)

                else:
                    end_points = np.linspace(-1, 1, self.sweeps)
                    for end_point in end_points:
                        P = np.array([end_point, 0, 1])
                        a, b, c = np.cross(TRUS_tip - TRUS_end, TRUS_tip - P)
                        d = -(a * P[0] + b * P[1] + c * P[2])
                        for point in fixed:
                            if point_plane_distance(point, a, b, c, d) < thresh:
                                swept_prostate = np.append(swept_prostate, point)

                swept_prostate = np.reshape(swept_prostate, (-1, self.dims))
                fixed = np.resize(swept_prostate, (self.kept_points, self.dims))

            fixed = fixed[np.random.choice(fixed.shape[0], self.kept_points, replace=False), :] if self.kept_points < fixed.shape[0] else fixed
            moving = moving[np.random.choice(moving.shape[0], self.kept_points, replace=False), :] if self.kept_points < moving.shape[0] else moving
            to_reg = to_reg[np.random.choice(to_reg.shape[0], self.kept_points, replace=False), :] if self.kept_points < to_reg.shape[0] else to_reg
            ground_truth = ground_truth[np.random.choice(ground_truth.shape[0], self.kept_points, replace=False), :] if self.kept_points < ground_truth.shape[0] else ground_truth
            
            X1.append(fixed)
            X2.append(moving)
            X3.append(to_reg)
            Y.append(ground_truth)

        return [np.array(X1), np.array(X2), np.array(X3)], np.array(Y)


def ypr_rand(min_deg, max_deg):
    y = [random.uniform(min_deg, max_deg)]
    p = [random.uniform(min_deg, max_deg)]
    r = [random.uniform(min_deg, max_deg)]
    return y, p, r


def d_rand(min_d, max_d):
    d = [
        random.uniform(min_d, max_d),
        random.uniform(min_d, max_d),
        random.uniform(min_d, max_d),
    ]
    return d


def e2r(yaw, pitch, roll):

    yaw_rad = np.deg2rad(yaw) / 2
    pitch_rad = np.deg2rad(pitch) / 2
    roll_rad = np.deg2rad(roll) / 2

    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)],
        ]
    )

    R_y = np.array(
        [
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)],
        ]
    )

    R_z = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def get_T(R, d):
    T = np.empty((4, 4))
    T[:3, :3] = R
    T[:3, 3] = d
    T[3, :] = [0, 0, 0, 1]
    return T


def compute_RBF(x, y, sigma):

    n, d = x.shape
    m, _ = y.shape
    K = np.zeros([n, m])
    for i in range(d):
        K += (x[:, [i]] * np.ones([1, m]) - np.ones([n, 1]) * y[:, i].T) ** 2
    K = np.exp(np.divide(K, (-2 * (sigma ** 2))))

    return K


def compute_RBF_defm(y):

    sigma = np.random.normal()
    c = np.random.normal(loc=0, scale=0.1, size=(10, 3))
    x = np.random.uniform(low=-1, high=1, size=(10, 3))
    k = compute_RBF(x, y, sigma)

    return np.matmul(k.T, c) + y


def point_plane_distance(P, a, b, c, d):

    return np.abs(P[0] * a + P[1] * b + P[2] * c + d) / np.sqrt(a*a + b*b + c*c)