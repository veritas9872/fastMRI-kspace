import numpy as np
import h5py


def load_model():
    pass


def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Additionally, values below 0 are automatically set to 0.
    Also, compression is performed to enhance the speed of data transfer to and from the cloud.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    gzip = dict(compression='gzip', compression_opts=1, shuffle=True, fletcher32=True)
    for file_name, recons in reconstructions.items():
        with h5py.File(out_dir / file_name, mode='x', libver='latest') as hf:
            hf.create_dataset('reconstruction', data=np.maximum(recons, 0), **gzip)


if __name__ == '__main__':
    pass
