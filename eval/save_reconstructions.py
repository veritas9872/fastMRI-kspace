import torch
import numpy
import h5py


def load_model():
    pass


def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    for file_name, recons in reconstructions.items():
        with h5py.File(out_dir / file_name, 'w') as f:
            f.create_dataset('reconstruction', data=recons)


if __name__ == '__main__':
    pass
