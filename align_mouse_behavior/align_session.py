import sys
## change import path (was in subfolder before)
from alignment import *

_, datapath_in, datapath_out, dataset, mouse, session = sys.argv

align_data_on_hpc(datapath_in,datapath_out,dataset,mouse,session)
