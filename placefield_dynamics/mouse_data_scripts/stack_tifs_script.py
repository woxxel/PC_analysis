import sys, shutil, os
import numpy as np
from pathlib import Path
from placefield_dynamics.neuron_detection import make_stack_from_single_tifs
from tifffile import TiffFile

def stack_tifs_for_mouse(pathMouse,session=None):
    '''
        session is provided as slurm-array number
    '''

    session -= 1  # to start from 0
    pathMouse = Path(pathMouse)
    pathsSession = sorted([path for path in pathMouse.iterdir() if path.stem.startswith('Session')])

    pathSession = pathsSession[session]
    print(f'Processing {pathSession.stem} in {pathMouse.stem}...')

    try:
        assert (pathSession / 'images').is_dir(), "Images folder not found!"
    except:
        return

    stackPath = make_stack_from_single_tifs(pathSession / 'images',pathSession,data_type='float16',normalize=True,clean_after_stacking=False)

    assert stackPath.is_file(), "Stacking of tif-files was not successful - file not found!"

    ## check if was properly created:
    tif_stack = TiffFile(stackPath)

    assert len(tif_stack.pages) == len(list(Path(pathSession / 'images').glob('*'))), "Number of frames in the imaging data do not agree!"

    ## picking three random frames to compare to original data
    T = len(tif_stack.pages)
    for page in np.random.choice(T, 3, replace=False):
        with TiffFile(
            pathSession / "images" / (stackPath.stem + f"{(page+1):04d}.tif")
        ) as tif_single:
            assert (
                tif_stack.pages[0].shape == tif_single.pages[0].shape
            ), "Dimensions of the imaging data do not agree!"

            dTif = np.abs(
                1
                - tif_stack.pages[page].asarray()
                / (tif_single.pages[0].asarray() / (2**16 - 1))
            )
            assert (
                dTif[np.isfinite(dTif)].max() < 1e-2
            ), "Data in the imaging data do not agree!"
    print('Stacking of tif-files was successful!')

    ## then, remove the old image files
    shutil.rmtree(pathSession / "images", ignore_errors=True)
    print('Successfully removed images folder!')

if __name__ == '__main__':
    _, pathMouse = sys.argv

    session = int(os.environ['SLURM_ARRAY_TASK_ID']) -1
    stack_tifs_for_mouse(pathMouse,int(session))
