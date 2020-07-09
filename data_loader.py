import numpy as np
from scipy.spatial import distance
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
        part=0,
        part_nn=0,
        noise=0,
        shuffle_points=False,
    ):
        self.data = data
        self.batch_size = batch_size
        self.dims = dims

        self.shuffle = shuffle
        self.rotate = rotate
        self.displace = displace
        self.deform = deform
        self.part = part
        self.part_nn = part_nn
        self.noise = noise
        self.shuffle_points=shuffle_points

        self.nb_sample = self.data["data"].shape[0]
        self.indexes = np.arange(self.nb_sample)

    def __len__(self):
        return int(np.floor(self.nb_sample / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        data_temp = [self.data["data"][k] for k in indexes]

        # Generate augmented batch data
        X, y = self.__generator_PART(data_temp)

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
        for d in data:
            # Extract data.
            fixed = d
            dims = fixed.shape

            # Normalize between [-1, 1] and mean center.
            fixed = 2 * (fixed - np.min(fixed)) / (np.ptp(fixed) + eps) - 1
            fixed = fixed - np.mean(fixed, axis=0)
            moving = fixed
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
            moving_with_ones = np.ones((dims[0], dims[1] + 1))
            moving_with_ones[:, :-1] = moving
            moving = np.dot(T, moving_with_ones.T).T[:, :-1]

            # Add some gaussian noise.
            if self.noise > 0:
                moving = moving + np.random.normal(0, self.noise, moving.shape)

            # Create the point cloud we actually register.
            to_reg = moving

            # Take part(s) from point set(s).
            test = True
            if self.part > 0:  # Register a part to whole
                index = np.random.randint(0, dims[0])
                part = np.random.randint(int(0.6 * 2048), int(0.9 * 2048))
                if self.part_nn:
                    part = int(
                        np.random.normal(self.part_nn, (2048 - self.part_nn) / 2)
                    )
                    part = max(min(part, 2048), 1024)
                if test:
                    part = self.part_nn
                D = distance.cdist([moving[index, :]], moving)
                closest = np.argsort(D[0])
                closest = closest[:part]
                moving = moving[closest, :]
                if self.part > 1:  # Register a part to a part
                    index = np.random.randint(0, dims[0])
                    part = np.random.randint(int(0.6 * 2048), int(0.9 * 2048))
                    if self.part_nn:
                        part = int(
                            np.random.normal(self.part_nn, (2048 - self.part_nn) / 2)
                        )
                        part = max(min(part, 2048), 1024)
                    if test:
                        part = self.part_nn
                    D = distance.cdist([fixed[index, :]], fixed)
                    closest = np.argsort(D[0])
                    closest = closest[:part]
                    fixed = fixed[closest, :]
            flip = False
            if flip:
                temp = moving
                moving = fixed
                fixed = temp

            # Resize after adding noise to prevent duplicate point-pairs from having different noise added.
            if self.part:
                moving = np.resize(moving, dims)
                fixed = np.resize(fixed, dims)
                ground_truth = np.resize(ground_truth, dims)

            if self.dims == 4:
                moving_with_ones = np.ones((dims[0], dims[1] + 1))
                moving_with_ones[:, :-1] = moving
                moving = moving_with_ones
                to_reg_with_ones = np.ones((dims[0], dims[1] + 1))
                to_reg_with_ones[:, :-1] = to_reg
                to_reg = to_reg_with_ones
                fixed_with_ones = np.ones((dims[0], dims[1] + 1))
                fixed_with_ones[:, :-1] = fixed
                fixed = fixed_with_ones
                gt_with_ones = np.ones((dims[0], dims[1] + 1))
                gt_with_ones[:, :-1] = ground_truth
                ground_truth = gt_with_ones

            if self.shuffle_points:
                X1.append(fixed[np.random.permutation(fixed.shape[0]), :])
                X2.append(moving[np.random.permutation(moving.shape[0]), :])
                X3.append(to_reg[np.random.permutation(to_reg.shape[0]), :])
                Y.append(ground_truth[np.random.permutation(ground_truth.shape[0]), :])
            else:
                X1.append(fixed)
                X2.append(moving)
                X3.append(to_reg)
                Y.append(ground_truth)
        return [np.array(X1), np.array(X2), np.array(X3)], np.array(Y)

    def __generator_PART(self, data):
        X1 = []
        X2 = []
        X3 = []
        Y = []
        eps = 1e-7
        for d in data:
            # Extract data.
            fixed = d
            dims = fixed.shape

            # Normalize between [-1, 1] and mean center.
            fixed = 2 * (fixed - np.min(fixed)) / (np.ptp(fixed) + eps) - 1
            fixed = fixed - np.mean(fixed, axis=0)
            moving = fixed

            # Take part(s) from point set(s).
            test = True
            if self.part > 0:  # Register a part to whole
                index = np.random.randint(0, dims[0])
                part = np.random.randint(int(0.6 * 2048), int(0.9 * 2048))
                if self.part_nn:
                    part = int(
                        np.random.normal(self.part_nn, (2048 - self.part_nn) / 2)
                    )
                    part = max(min(part, 2048), 1024)
                if test:
                    part = self.part_nn
                D = distance.cdist([moving[index, :]], moving)
                closest = np.argsort(D[0])
                closest = closest[:part]
                moving = moving[closest, :]
                if self.part > 1:  # Register a part to a part
                    index = np.random.randint(0, dims[0])
                    part = np.random.randint(int(0.6 * 2048), int(0.9 * 2048))
                    if self.part_nn:
                        part = int(
                            np.random.normal(self.part_nn, (2048 - self.part_nn) / 2)
                        )
                        part = max(min(part, 2048), 1024)
                    if test:
                        part = self.part_nn
                    D = distance.cdist([fixed[index, :]], fixed)
                    closest = np.argsort(D[0])
                    closest = closest[:part]
                    fixed = fixed[closest, :]
            flip = False
            if flip:
                temp = moving
                moving = fixed
                fixed = temp

            ground_truth = moving  # Make the ground truth the unmoved-moving part.

            # Resize after adding noise to prevent duplicate point-pairs from having different noise added.
            if self.part:
                moving = np.resize(moving, dims)
                fixed = np.resize(fixed, dims)
                ground_truth = np.resize(ground_truth, dims)

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
            moving_with_ones = np.ones((dims[0], dims[1] + 1))
            moving_with_ones[:, :-1] = moving
            moving = np.dot(T, moving_with_ones.T).T[:, :-1]

            # Add some gaussian noise.
            if self.noise > 0:
                moving = moving + np.random.normal(0, self.noise, moving.shape)

            # Create the point cloud we actually register.
            to_reg = moving

            if self.dims == 4:
                moving_with_ones = np.ones((dims[0], dims[1] + 1))
                moving_with_ones[:, :-1] = moving
                moving = moving_with_ones
                to_reg_with_ones = np.ones((dims[0], dims[1] + 1))
                to_reg_with_ones[:, :-1] = to_reg
                to_reg = to_reg_with_ones
                fixed_with_ones = np.ones((dims[0], dims[1] + 1))
                fixed_with_ones[:, :-1] = fixed
                fixed = fixed_with_ones
                gt_with_ones = np.ones((dims[0], dims[1] + 1))
                gt_with_ones[:, :-1] = ground_truth
                ground_truth = gt_with_ones

            if self.shuffle_points:
                X1.append(fixed[np.random.permutation(fixed.shape[0]), :])
                X2.append(moving[np.random.permutation(moving.shape[0]), :])
                X3.append(to_reg[np.random.permutation(to_reg.shape[0]), :])
                Y.append(ground_truth[np.random.permutation(ground_truth.shape[0]), :])
            else:
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
