import os, time
from pathlib import Path
from .utils.connections import *


def copy_loewel_data(
    path_source="/scratch/users/caiman_share",
    path_target="/usr/users/cidbn1/neurodyn/PSD95Mice_Loewel",
    # resultFile_name="results_CaImAn",
    hpc="sofja",
):
    """
    Run neuron detection on the GWDG cluster

    Both, path_source and path_target should be paths to folders on the
    GWDG cluster

    The source contains data stored in one folder per mouse, with subfolders
    for each day, containing possibly multiple sessions (up to 3?). They are
    named by their date of recording (YYYYMMDD) and the number of session per
    day, thus alphabetical sorting provides proper ordering.

    The target folder will contain one folder per recording session, named
    'SessionXX' with XX being the number of the (overall) session.

    """

    client, path_code, batch_params = set_hpc_params(hpc)
    # sftp_client = client.open_sftp()

    _, stdout, stderr = client.exec_command(
        f"ls -d {os.path.join(path_source,'ID*/ | xargs -n 1 basename')}"
    )
    mice = str(stdout.read(), encoding="utf-8").splitlines()
    print(mice)

    for m, mouse in enumerate(mice):

        print(mouse)
