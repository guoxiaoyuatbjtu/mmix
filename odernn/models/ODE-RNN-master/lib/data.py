import abc

import numpy as np
import stheno
import torch

from lib.utils import device

__all__ = ['GPGenerator', 'SawtoothGenerator']


def _rand(val_range, *shape):
    lower, upper = val_range
    return lower + np.random.rand(*shape) * (upper - lower)


def _uprank(a):
    if len(a.shape) == 1:
        return a[:, None, None]
    elif len(a.shape) == 2:
        return a[:, :, None]
    elif len(a.shape) == 3:
        return a
    else:
        return ValueError(f'Incorrect rank {len(a.shape)}.')


class LambdaIterator:
    """Iterator that repeatedly generates elements from a lambda.

    Args:
        generator (function): Function that generates an element.
        num_elements (int): Number of elements to generate.
    """

    def __init__(self, generator, num_elements):
        self.generator = generator
        self.num_elements = num_elements
        self.index = 0

    def __next__(self):
        self.index += 1
        if self.index <= self.num_elements:
            return self.generator()
        else:
            raise StopIteration()

    def __iter__(self):
        return self


class DataGenerator(metaclass=abc.ABCMeta):
    """Data generator for GP samples.

    Args:
        batch_size (int, optional): Batch size. Defaults to 16.
        num_tasks (int, optional): Number of tasks to generate per epoch.
            Defaults to 256.
        x_range (tuple[float], optional): Range of the inputs. Defaults to
            [-2, 2].
        max_train_points (int, optional): Number of training points. Must be at
            least 3. Defaults to 50.
        max_test_points (int, optional): Number of testing points. Must be at
            least 3. Defaults to 50.
    """

    def __init__(self,
                 batch_size=16,
                 num_tasks=256,
                 x_range=(-2, 2),
                 max_train_points=50,
                 max_test_points=50,
                 num_points_per_batch=256):

        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.x_range = x_range
        self.max_train_points = max(max_train_points, 3)
        self.max_test_points = max(max_test_points, 3)
        self.num_points_per_batch = num_points_per_batch

    @abc.abstractmethod
    def sample(self, x):
        """Sample at inputs `x`.

        Args:
            x (vector): Inputs to sample at.

        Returns:
            vector: Sample at inputs `x`.
        """

    def generate_task(self):
        """Generate a task.

        Returns:
            dict: A task, which is a dictionary with keys `x`, `y`, `x_context`,
                `y_context`, `x_target`, and `y_target.
        """

        # self.x_range(-2, 2)
        x = np.sort(_rand(self.x_range, self.num_points_per_batch))

        task = {}

        task['mask_obs']   = np.zeros((self.batch_size, len(x), 1))
        task['mask_y']     = np.zeros((self.batch_size, len(x), 1))
        task['mask_first'] = np.zeros((self.batch_size, len(x), 1))
        task['y']          = np.zeros((self.batch_size, len(x), 1))

        for i in range(self.batch_size):
            num_train_points = np.random.randint(3, self.max_train_points + 1)
            num_test_points  = np.random.randint(3, self.max_test_points + 1)
            num_points = num_train_points + num_test_points
            start_point_index = np.random.randint(0, self.num_points_per_batch - num_points + 1)

            points_indicies = np.random.choice(np.arange(start_point_index + 1, self.num_points_per_batch), size=num_points - 1, replace=False)
            points_indicies = np.concatenate([[start_point_index], points_indicies, np.arange(self.num_points_per_batch, len(x))])

            task['y'][i, points_indicies, 0] = self.sample(x[points_indicies])
            task['mask_first'][i, start_point_index:] = 1
            task['mask_obs'][i, points_indicies[:num_train_points]] = 1
            task['mask_y'][i, points_indicies] = 1

        # Delete unused times.
        used_time_steps = task['mask_y'].sum(axis=(0, -1)) > 0

        # Stack batch and convert to PyTorch.
        task = {k: torch.tensor(v[:, used_time_steps], dtype=torch.float32).to(device) for k, v in task.items()}
        task['x'] = torch.tensor(x[used_time_steps], dtype=torch.float32).to(device)

        return task


    def __iter__(self):
        return LambdaIterator(lambda: self.generate_task(), self.num_tasks)


class GPGenerator(DataGenerator):
    """Generate samples from a GP with a given kernel.

    Further takes in keyword arguments for :class:`.data.DataGenerator`.

    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
            Defaults to an EQ kernel.
    """

    def __init__(self, kernel=stheno.EQ(), **kw_args):
        self.gp = stheno.GP(kernel)
        DataGenerator.__init__(self, **kw_args)

    def sample(self, x):
        return np.squeeze(self.gp(x).sample())


class SawtoothGenerator(DataGenerator):
    """Generate samples from a random sawtooth.

    Further takes in keyword arguments for :class:`.data.DataGenerator`. The
    default numbers for `max_train_points` and `max_test_points` are 100.

    Args:
        freq_dist (tuple[float], optional): Lower and upper bound for the
            random frequency. Defaults to [3, 5].
        shift_dist (tuple[float], optional): Lower and upper bound for the
            random shift. Defaults to [-5, 5].
        trunc_dist (tuple[float], optional): Lower and upper bound for the
            random truncation. Defaults to [10, 20].
    """

    def __init__(self,
                 freq_dist=(3, 5),
                 shift_dist=(-5, 5),
                 trunc_dist=(10, 20),
                 max_train_points=100,
                 max_test_points=100,
                 **kw_args):
        self.freq_dist = freq_dist
        self.shift_dist = shift_dist
        self.trunc_dist = trunc_dist
        DataGenerator.__init__(self,
                               max_train_points=max_train_points,
                               max_test_points=max_test_points,
                               **kw_args)

    def sample(self, x):
        # Sample parameters of sawtooth.
        amp = 1
        freq = _rand(self.freq_dist)
        shift = _rand(self.shift_dist)
        trunc = np.random.randint(self.trunc_dist[0], self.trunc_dist[1] + 1)

        # Construct expansion.
        x = x[:, None] + shift
        k = np.arange(1, trunc + 1)[None, :]
        return 0.5 * amp - amp / np.pi * \
               np.sum((-1) ** k * np.sin(2 * np.pi * k * freq * x) / k, axis=1)
