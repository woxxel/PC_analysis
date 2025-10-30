import os, time, warnings, itertools, pickle, copy
from pathlib import Path
import numpy as np
import scipy as sp
from tqdm import *
import itertools
from scipy.optimize import curve_fit

from multiprocessing import get_context

from .utils import (
    pickleData,
    fdr_control,
    periodic_distr_distance,
    get_reliability,
    get_status_arr,
    get_average,
)
from .utils import cluster_parameters
from .utils import timing

from .placefield_detection import prepare_behavior_from_file, get_firingrate
from .mouse_data_scripts.get_session_specifics import *
from .neuron_matching import matching, matching_params, load_data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
warnings.filterwarnings("ignore")


class cluster_analysis:
    """
        class of functions and objects to accumulate previously obtained
        results processing sessions separately and calculating cross session statistics

        requires
                                                                                                                                                                                                                                                                        - processed footprint data (footprint matching)
        'pathAssignments' (single path to matching results file)
        from module "neuron_matching" @ https://github.com/woxxel/neuron_matching.git
                                                                                                                                                                                                                                                                                                        - processed calcium activity (tuning curve detection)
        'paths' (list of processed temporal activity files)
        from module "PC_detection" @ https://github.com/woxxel/PC_detection.git
                                                                                                                                                                                                                                                                                                        - processed behavior files (from... where? extra module?)

    TODO:
        [ ] where did calculation of 'rotation' (normal and anchor) go?
        [ ] fix calculation of oof_/if_firingrate and implement as being calculated on startup
        [ ] properly document and check cluster.status['activity'] - something's fishy there!
        [ ] (re?)add possibility to specify session range for analysis
    """

    def __init__(
        self,
        path_mouse,
        # path_matching,
        paths_behavior=None,
        paths_place_field_detection=None,
        paths_neuron_detection=None,
        mouse=None,
        # session_order=None,
        s_corr_min=0.2,
        matlab=False,
        matching_only=False,
        suffix="redetected",
    ):

        self.matlab = matlab
        self.path_mouse = path_mouse

        paramsObj = cluster_parameters(mouse, s_corr_min, matlab=matlab)
        # paramsObj.set_paths(paths_session,paths_results,self.path_mouse)
        # self.paths = paramsObj.paths

        self.data = paramsObj.data
        self.params = paramsObj.params

        self.paths = {
            # "matching": path_matching,
            "behavior": paths_behavior,
            "place_field_detection": paths_place_field_detection,
            "neuron_detection": paths_neuron_detection,
            "figures": Path(path_mouse) / "figures",
        }
        self.suffix = suffix

        self.matching_only = matching_only

        if not matching_only:
            assert paths_behavior is not None, "No behavior data provided"
            assert (
                paths_place_field_detection is not None
            ), "No place field data provided"
            assert (
                paths_neuron_detection is not None
            ), "No neuron detection data provided"

            ld = load_data(paths_place_field_detection[0])
            self.data["nbin"] = ld["firingstats"]["map"].shape[1]

        # self.data["nSes"] = len(self.paths["behavior"])

    def get_path(self, file: str, s: int, check_exists: bool = False):
        """
        helper function to get path to file of type 'file' for session s
        and optionally check if it exists
        """

        if file in ["neuron_detection", "place_field_detection", "behavior"]:
            path = self.paths[file][s]
        # path = os.path.join(
        #     self.paths["sessions"][s], self.paths[f"fileName{file}"]
        # )
        else:
            assert False, f'entry "fileName{file}" not found'

        if check_exists:
            # print(file, s, path)
            return path, (path is None) or path.exists()
        else:
            return path

    def run_complete(self, sessions=None, n_processes=8, reprocess=False):

        ### loading data from different states of the analysis process
        # if reprocess: #| (not os.path.exists(self.paths['assignments'])) |

        self.get_matching(sessions=sessions)
        self.get_behavior()
        self.get_stats()
        self.get_PC_fields()

        self.update_status()

        self.calculate_field_firingrates()
        self.calculate_recurrence(n_shuffles=1000)

        self.compareSessions(n_processes=n_processes)

        self.stability = self.calculate_placefield_stability(
            dsMax=self.data["nSes"], n_processes=n_processes, N_bs=100
        )
        # _,_ = self.recalc_firingrate()
        self.save()

    def prepare_dicts(self, which=None, overwrite=False):

        if not which:
            which = ["status", "alignment", "matching", "stats", "compare"]

        if "status" in which and (not hasattr(self, "status") or overwrite):
            self.status = {
                "sessions": np.zeros(self.data["nSes"], "bool"),
                "clusters": np.zeros(self.data["nC"], "bool"),
                "activity": np.zeros((self.data["nC"], self.data["nSes"], 6), "bool"),
            }

        if "alignment" in which and (not hasattr(self, "alignment") or overwrite):
            self.alignment = {
                "shift": np.zeros((self.data["nSes"], 2)),
                "corr": np.full(self.data["nSes"], np.nan),
                "borders": np.full((2, 2), np.nan),
                # 'flow': np.zeros((self.data['nSes'],2)+self.params['dims']),
                "transposed": np.zeros(self.data["nSes"], "bool"),
            }

        if "matching" in which and (not hasattr(self, "matching") or overwrite):
            self.matching = {
                "IDs": None,
                "com": np.full((self.data["nC"], self.data["nSes"], 2), np.nan),
                "score": np.zeros((self.data["nC"], self.data["nSes"], 2)),
            }

        if "stats" in which and (not hasattr(self, "stats") or overwrite):
            self.stats = {
                "firingrate": np.full((self.data["nC"], self.data["nSes"]), np.nan),
                "firingmap": np.zeros(
                    (self.data["nC"], self.data["nSes"], self.data["nbin"])
                ),
                # 'trial_map':np.zeros((0,self.data['nSes'],self.params['field_count_max'],self.sessions['trial_ct'].max()),'bool'),
                "SNR_comp": np.zeros((self.data["nC"], self.data["nSes"])),
                "cnn_preds": np.zeros((self.data["nC"], self.data["nSes"])),
                "r_values": np.zeros((self.data["nC"], self.data["nSes"])),
                "MI_value": np.zeros((self.data["nC"], self.data["nSes"])),
                "MI_p_value": np.zeros((self.data["nC"], self.data["nSes"])),
                "MI_z_score": np.zeros((self.data["nC"], self.data["nSes"])),
                "Isec_value": np.zeros((self.data["nC"], self.data["nSes"])),
            }

        if "behavior" in which and (not hasattr(self, "behavior") or overwrite):

            self.behavior = {
                "trial_ct": np.zeros(self.data["nSes"], "int"),
                "trial_frames": {},
                "time_active": np.zeros(self.data["nSes"]),
                "speed": np.zeros(self.data["nSes"]),
                "performance": {},
            }

        if "fields" in which and (not hasattr(self, "fields") or overwrite):
            self.fields = {
                "nModes": np.zeros((self.data["nC"], self.data["nSes"])).astype(
                    "uint8"
                ),
                "status": np.zeros(
                    (self.data["nC"], self.data["nSes"], self.params["field_count_max"])
                ).astype("uint8"),
                "trial_act": np.zeros(
                    (
                        self.data["nC"],
                        self.data["nSes"],
                        self.params["field_count_max"],
                        self.behavior["trial_ct"].max(),
                    ),
                    "bool",
                ),
                "logz": np.full(
                    (
                        self.data["nC"],
                        self.data["nSes"],
                        self.params["field_count_max"] + 1,
                    ),
                    np.nan,
                ),
                "Bayes_factor": np.full(
                    (
                        self.data["nC"],
                        self.data["nSes"],
                        self.params["field_count_max"],
                    ),
                    np.nan,
                ),
                "reliability": np.full(
                    (
                        self.data["nC"],
                        self.data["nSes"],
                        self.params["field_count_max"],
                    ),
                    np.nan,
                ),
                "max_rate": np.full(
                    (
                        self.data["nC"],
                        self.data["nSes"],
                        self.params["field_count_max"],
                    ),
                    np.nan,
                ),
                # "posterior_mass": np.full(
                #     (
                #         self.data["nC"],
                #         self.data["nSes"],
                #         self.params["field_count_max"],
                #     ),
                #     np.nan,
                # ),
                "baseline": np.full(
                    (
                        self.data["nC"],
                        self.data["nSes"],
                        3,
                    ),
                    np.nan,
                ),
                "amplitude": np.full(
                    (
                        self.data["nC"],
                        self.data["nSes"],
                        self.params["field_count_max"],
                        3,
                    ),
                    np.nan,
                ),
                "width": np.full(
                    (
                        self.data["nC"],
                        self.data["nSes"],
                        self.params["field_count_max"],
                        3,
                    ),
                    np.nan,
                ),
                "location": np.full(
                    (
                        self.data["nC"],
                        self.data["nSes"],
                        self.params["field_count_max"],
                        3,
                    ),
                    np.nan,
                ),
                "p_x": np.full(
                    (
                        self.data["nC"],
                        self.data["nSes"],
                        self.params["field_count_max"],
                        100,
                    ),
                    np.nan,
                ),
            }

        if "compare" in which and (not hasattr(self, "compare") or overwrite):
            self.compare = {}

    # def process_sessions(self, sessions=None, n_processes=0, reprocess=False):
    #     """
    #     obtain basic information from sessions to allow processing and plotting
    #     analyses combining results from different steps of the process
    #     """

    #     if reprocess | (not os.path.exists(self.paths["svSessions"])):

    #         self.get_matching()
    #         self.get_behavior()
    #         # self.save([False,True,False,False,False])

    # #         ## load originally detected data
    # #         pathLoad = pathcat([path,'results_OnACID.mat'])
    # # if os.path.exists(pathLoad):
    # #     results_original = sio.whosmat(pathLoad)
    # #     if results_original[1][0] == 'C':
    # #         self.sessions['N_original'][s] = results_original[1][1][0]
    # #     else:
    # #         print('nope')

    def set_thresholds(self, **kwargs):
        """
        set thresholds for neuron and session classification
        """

        self.thr = {
            "SNR_lowest": kwargs.get("SNR_lowest") or matching_params["SNR_lowest"],
            "SNR_min": kwargs.get("SNR_min") or matching_params["SNR_min"],
            "rval_lowest": kwargs.get("rval_lowest") or matching_params["rval_lowest"],
            "rval_min": kwargs.get("rval_min") or matching_params["rval_min"],
            "cnn_lowest": kwargs.get("cnn_lowest") or matching_params["cnn_lowest"],
            "cnn_min": kwargs.get("cnn_min") or matching_params["cnn_min"],
            
            "p_matched": kwargs.get("pm_thr") or self.params["pm_thr"],
            "firingrate": kwargs.get("fr_thr") or self.params["fr_thr"],
            "Bayes": kwargs.get("Bayes_thr") or self.params["Bayes_thr"],
            "reliability": kwargs.get("reliability_thr")
            or self.params["reliability_thr"],
            "A_0": kwargs.get("A0_thr") or self.params["A0_thr"],
            "A": kwargs.get("A_thr") or self.params["A_thr"],
            "A_rate": kwargs.get("Arate_thr") or self.params["Arate_thr"],
            "sigma": kwargs.get("sigma_thr") or self.params["sigma_thr"],
            # "alpha": self.params["MI_alpha"] if alpha is None else alpha,
            # "MI": self.params["MI_thr"] if MI_thr is None else MI_thr,
            "min_cluster_count": kwargs.get("min_cluster_count")
            or self.params["min_cluster_count"],
        }

    def get_matching(self, sessions=None):
        """
        loads information from matching algorithm, such as session-specifics (e.g. shift and session-correlation) and neuron specifics (e.g. center of mass, matching probability, IDs)
        """

        ## load model first, to allow proper matching of footprints
        m = matching(
            mousePath=self.path_mouse,
            paths=self.paths["neuron_detection"],
            matlab=self.matlab,
            suffix=self.suffix,
        )
        m.load_model()
        m.dynamic_fit()
        m.load_registration()
        m.load_data()

        # matching_data = load_data(path_matching)
        self.data["nC"], self.data["nSes"] = m.results["assignments"].shape
        if self.matching_only:
            self.data["nbin"] = (
                10  # just to provide some basic value for initialization
            )

        self.prepare_dicts(which=["alignment", "matching", "stats"])
        self.matching["f_same"] = m.model["f_same"]
        # modelPath = os.path.join(*os.path.split(self.paths['assignments'])[:-1],f'match_model{self.paths["suffix"]}.{"mat" if self.matlab else "pkl"}')
        # ldModel = load_data(modelPath)

        # if 'f_same' in ldModel.keys():
        # 	self.matching['f_same'] = ldModel['f_same']

        # results = load_data(self.paths['assignments'])
        # m.results = results
        self.matching["IDs"] = m.results["assignments"]

        ## get matching results
        self.matching["score"] = m.results["p_matched"]
        self.matching["com"] = m.results["cm"]

        self.stats["SNR_comp"] = m.results["SNR_comp"]
        self.stats["r_values"] = m.results["r_values"]
        self.stats["cnn_preds"] = m.results["cnn_preds"]

        self.alignment["transposed"] = m.results["remap"][
            "transposed"
        ]  # if has_reference else False
        self.alignment["shift"] = m.results["remap"][
            "shift"
        ]  # if has_reference else [0,0]
        self.alignment["corr"] = m.results["remap"]["corr"]  # if has_reference else 1
        self.alignment["corr_zscored"] = m.results["remap"][
            "corr_zscored"
        ]  # if has_reference else 1

        self.alignment["flow"] = np.zeros((self.data["nSes"], 2) + self.params["dims"])
        for s in range(self.data["nSes"]):
            if not s in m.data or not "remap" in m.data[s].keys():
                continue
            self.alignment["flow"][s, ...] = m.data[s]["remap"]["flow"]

        # has_reference = False
        # for s in range(self.data['nSes']):

        # 	_, path_exists = self.get_path('neuron_detection',s,check_exists=True)
        # 	if not path_exists:
        # 		continue

        # 	# self.alignment['flow'][s,...] = ldData['data'][s]['remap']['flow'] if has_reference else np.nan

        # 	# idx_c = np.where(np.isfinite(self.matching['IDs'][:,s]))[0]

        # 	# ## match- and best non-match-score should be calculated and stored in matching algorithm
        # 	# if not has_reference:
        # 	#     self.matching['score'][idx_c,s,0] = 1
        # 	#     self.matching['score'][idx_c,s,1] = np.nan
        # 	# elif s in ldData['data'].keys():

        # 	#     p_all = ldData['data'][s]['p_same']

        # 	#     idx_c_first = idx_c[idx_c>=p_all.shape[0]]    # first occurence of a neuron is always certain match!
        # 	#     self.matching['score'][idx_c_first,s,0] = 1
        # 	#     self.matching['score'][idx_c_first,s,1] = np.nan

        # 	#     idx_c = idx_c[idx_c<p_all.shape[0]]    # remove entries of first-occurence neurons (no matching possible)
        # 	#     self.matching['score'][idx_c,s,0] = p_matched[idx_c,s]
        # 	#     scores_now = p_all.toarray()
        # 	#     self.matching['score'][idx_c,s,1] = [max(scores_now[c,np.where(scores_now[c,:]!=self.matching['score'][c,s,0])[0]]) for c in idx_c]
        # 	has_reference = True

        self.classify_sessions(sessions=sessions)
        self.classify_components()

    def classify_sessions(self, sessions=None):  # ,max_shift=None,min_corr=None):
        """
        checks all sessions to pass certain criteria to
        be included in the further analysis
        """

        self.prepare_dicts(which=["status"])

        self.status["sessions"][:] = False

        ## if 'sessions' is provided (tuple), it specifies range
        ## of sessions to be included
        if sessions is None:
            # print()
            # self.sStart = 0
            self.sStart = np.where(self.alignment["corr"] == 1.0)[0][0]
            self.sEnd = self.data["nSes"]
        else:
            self.sStart = max(0, sessions[0] - 1)
            self.sEnd = sessions[-1]

        # print(f'processing sessions {self.sStart} to {self.sEnd}')
        self.status["sessions"][self.sStart : self.sEnd] = True

        # print(f'processing sessions {self.sStart} to {self.sEnd}')
        # print(self.alignment['shift'].dtype)

        ## check for coherence with other sessions (low shift, high correlation)
        abs_shift = np.array(
            [np.sqrt(x**2 + y**2) for (x, y) in self.alignment["shift"]]
        )
        self.status["sessions"][
            abs_shift > self.params["session_max_shift"]
        ] = False  ## huge shift
        self.status["sessions"][
            self.alignment["corr_zscored"]
            < self.params["min_session_correlation_zscore"]
        ] = False  ## huge shift
        self.status["sessions"][np.isnan(self.alignment["corr_zscored"])] = False

        # ## reset first session to True if needed (doesnt pass correlation check)
        if self.sStart == 0:
            self.status["sessions"][0] = True

        ## finally, check if data can be loaded properly
        for s in np.where(self.status["sessions"])[0]:

            # print(CNMF_exists,Fields_exists,Behavior_exists)
            if self.matching_only:
                if not self.paths["neuron_detection"][s].exists():
                    self.status["sessions"][s] = False
            else:
                for key in ["neuron_detection", "place_field_detection", "behavior"]:
                    if (self.paths[key][s] is None) or not self.paths[key][s].exists():
                        self.status["sessions"][s] = False

        thr_high = self.params["dims"] + self.alignment["shift"][
            self.status["sessions"], :
        ].min(0)
        thr_low = self.alignment["shift"][self.status["sessions"], :].max(0)

        self.alignment["borders"] = np.vstack([thr_low, thr_high])

    def get_status_component_detected(self):

        return (
            (
                ## minimum requirements for each neuron
                (self.stats["SNR_comp"] > self.thr["SNR_lowest"])
                & (self.stats["r_values"] > self.thr["rval_lowest"])
                & (self.stats["cnn_preds"] > self.thr["cnn_lowest"])
            )
            & (
                ## each neuron needs to exceed at least one of the following thresholds
                (self.stats["SNR_comp"] > self.thr["SNR_min"])
                | (self.stats["r_values"] > self.thr["rval_min"])
                | (self.stats["cnn_preds"] > self.thr["cnn_min"])
            )
            & (self.matching["score"][..., 0] > self.thr["p_matched"])
        )

    def classify_components(self, border_margin=None):
        """
        checks all clusters to pass certain criteria to be considered in the analysis

        Each cluster is required to:
            * pass all "lowest" thresholds (SNR, r-value, CNN-prediction)
            * at least one of the "min" thresholds (SNR, r-value, CNN-prediction)
            * be present in at least 'min_cluster_count' sessions
            * have a center of mass within the borders of the imaging window, leaving some margin of 'border_margin'
        """
        self.prepare_dicts(which=["status"])

        self.params["border_margin"] = border_margin or self.params["border_margin"]

        self.status["clusters"] = np.ones(self.data["nC"]).astype("bool")

        ## check for neuron detection thresholds
        if not hasattr(self, "thr"):
            self.set_thresholds()

        self.status["component_detected"] = (
            (
                ## minimum requirements for each neuron
                (self.stats["SNR_comp"] > self.thr["SNR_lowest"])
                & (self.stats["r_values"] > self.thr["rval_lowest"])
                & (self.stats["cnn_preds"] > self.thr["cnn_lowest"])
            )
            & (
                ## each neuron needs to exceed at least one of the following thresholds
                (self.stats["SNR_comp"] > self.thr["SNR_min"])
                | (self.stats["r_values"] > self.thr["rval_min"])
                | (self.stats["cnn_preds"] > self.thr["cnn_min"])
            )
            & (self.matching["score"][..., 0] > self.thr["p_matched"])
        )

        ## remove components from sessions that are not included in the data
        self.status["component_detected"][:, ~self.status["sessions"]] = False

        ## check for presence in at least 'min_cluster_count' sessions
        self.status["clusters"][
            self.status["component_detected"][:, self.status["sessions"]].sum(1)
            < self.thr["min_cluster_count"]
        ] = False

        ## check for distance from imaging window borders
        for i in range(2):
            idx_remove_low = self.matching["com"][:, self.status["sessions"], i] < (
                self.alignment["borders"][0, i] + self.params["border_margin"]
            )
            self.status["clusters"][np.any(idx_remove_low, 1)] = False

            idx_remove_high = self.matching["com"][:, self.status["sessions"], i] > (
                self.alignment["borders"][1, i] - self.params["border_margin"]
            )
            self.status["clusters"][np.any(idx_remove_high, 1)] = False

    def get_behavior(self):
        """
        accumulates information from all sessions of this mouse
        """

        self.prepare_dicts(which=["behavior"])

        for s in range(self.data["nSes"]):
            pathBehavior = self.get_path("behavior", s)

            if not self.status["sessions"][s]:
                continue

            ldData = prepare_behavior_from_file(
                pathBehavior, nbin_coarse=20, calculate_performance=True
            )
            self.behavior["trial_ct"][s] = ldData["trials"]["ct"]
            self.behavior["trial_frames"][s] = ldData["trials"]["nFrames"]

            self.behavior["speed"][s] = np.nanmean(ldData["velocity"])
            self.behavior["time_active"][s] = ldData["active"].sum() / self.params["f"]

            if "performance" in ldData.keys():
                self.behavior["performance"][s] = ldData["performance"]

        self.session_data = get_session_specifics(self.data["mouse"], self.data["nSes"])

    def get_stats(self):

        self.prepare_dicts(which=["stats"])

        for s in tqdm(range(self.data["nSes"]), leave=False):

            if not self.status["sessions"][s]:
                continue

            idx_c = np.where(np.isfinite(self.matching["IDs"][:, s]))[0]
            n_arr = self.matching["IDs"][idx_c, s].astype("int")

            ## load results from the place field detection algorithm
            # pathPCFields, path_exists = self.get_path(
            #     "place_field_detection", s, check_exists=True
            # )
            if self.paths["place_field_detection"][s].exists():

                PCFields = load_data(self.paths["place_field_detection"][s])
                # with open(self.paths['place_field_detection'][s], "rb") as f_open:
                #     PCFields = pickle.load(f_open)

                self.stats["firingrate"][idx_c, s] = PCFields["firingstats"]["rate"][
                    n_arr
                ]
                self.stats["firingmap"][idx_c, s, :] = PCFields["firingstats"]["map"][
                    n_arr, :
                ]

                # print('loading MI')
                # self.stats["MI_value"][idx_c, s] = PCFields["status"]["MI_value"][n_arr]
                # self.stats["MI_p_value"][idx_c, s] = PCFields["status"]["MI_p_value"][
                #     n_arr
                # ]
                # self.stats["MI_z_score"][idx_c, s] = PCFields["status"]["MI_z_score"][
                #     n_arr
                # ]

                # self.stats["Isec_value"][idx_c, s] = PCFields["status"]["Isec_value"][
                #     n_arr
                # ]

                # self.save([False,False,True,False,False])
                # print('stats obtained - time taken: %5.3g'%(time.time()-t_start))

    def get_PC_fields(self):

        t_start = time.time()

        self.prepare_dicts(which=["fields"])

        for s in tqdm(range(self.data["nSes"]), leave=False):
            if not self.status["sessions"][s]:
                continue

            if not self.paths["place_field_detection"][s].exists():
                print(
                    f"Data for Session {self.paths['place_field_detection'][s]} does not exist"
                )
                continue

            # idx_c = np.where(~np.isnan(self.matching["IDs"][:, s]))[0]
            idx_c = np.isfinite(self.matching["IDs"][:, s])
            # print(f"{s}: actives: {idx_c.sum()}")

            nCells = idx_c.sum()
            # print(np.where(idx_c)[0])
            n_arr = self.matching["IDs"][idx_c, s].astype("int")
            # n_arr = self.matching["IDs"][idx_c, s].astype("int")

            ld = load_data(self.paths["place_field_detection"][s])

            # firingstats_tmp = ld["firingstats"]
            fields = ld["fields"]

            ### hand over all other values
            self.fields["baseline"][idx_c, s, ...] = fields["parameter"]["global"][
                "A0"
            ][n_arr, ...]
            self.fields["logz"][idx_c, s, ...] = fields["logz"][n_arr, ..., 0]
            self.fields["Bayes_factor"][idx_c, s, ...] = np.diff(
                fields["logz"][n_arr, :, 0], axis=1
            )

            self.fields["nModes"][idx_c, s] = fields["n_modes"][n_arr]

            idx_c = idx_c & (self.fields["nModes"][:, s] > 0)
            n_arr = self.matching["IDs"][idx_c, s].astype("int")
            # print(f"{s}: PCs: {idx_c.sum()}")

            self.fields["location"][idx_c, s, ...] = fields["parameter"]["global"][
                "theta"
            ][n_arr, ...]
            self.fields["width"][idx_c, s, ...] = fields["parameter"]["global"][
                "sigma"
            ][n_arr, ...]
            self.fields["amplitude"][idx_c, s, ...] = fields["parameter"]["global"][
                "A"
            ][n_arr, ...]

            self.fields["p_x"][idx_c, s, ...] = fields["p_x"]["global"]["theta"][
                n_arr, ...
            ]

            self.fields["reliability"][idx_c, s, ...] = fields["reliability"][
                n_arr, ...
            ]

            if fields["active_trials"].shape[-1] == self.behavior["trial_ct"][s]:
                self.fields["trial_act"][
                    idx_c, s, :, : self.behavior["trial_ct"][s]
                ] = fields["active_trials"][n_arr, ...]
            else:
                self.fields["trial_act"][
                    idx_c, s, :, : fields["active_trials"].shape[-1]
                ] = fields["active_trials"][n_arr, ...]

            # for c, n in zip(idx_c, n_arr):

            # 	if self.fields["nModes"][c, s] == 0:  ## cell is PC
            # 		continue

            ### hand over field parameters
            # self.fields["location"][c, s, ...] = fields["parameter"][
            # 	"global"
            # ]["theta"][n, ...]
            # self.fields["width"][c, s, ...] = fields["parameter"]["global"][
            # 	"sigma"
            # ][n, ...]
            # self.fields["amplitude"][c, s, ...] = fields["parameter"]["global"][
            # 	"A"
            # ][n, ...]
            # self.fields["baseline"][c, s, ...] = fields["parameter"]["global"][
            # 	"A0"
            # ][n, ...]

            # self.fields["posterior_mass"][c, s, :] = fields[
            # 	"posterior_mass"
            # ][n, : self.params["field_count_max"]]

            # for f in range(fields["n_modes"][n]):

            # # 	# self.fiel
            # 	if (
            # 		firingstats_tmp["trial_map"].shape[1]
            # 		== self.behavior["trial_ct"][s]
            # 	):
            # 	(
            # 		self.fields["reliability"][c, s, f],
            # 		self.fields["max_rate"][c, s, f],
            # 		self.fields["trial_act"][
            # 			c, s, f, : self.behavior["trial_ct"][s]
            # 		],
            # 	) = get_reliability(
            # 		firingstats_tmp["trial_map"][n, ...],
            # 		firingstats_tmp["map"][n, ...],
            # 		fields["parameter"][n, ...],
            # 		f,
            # 	)
            # else:
            # 	(
            # 		self.fields["reliability"][c, s, f],
            # 		self.fields["max_rate"][c, s, f],
            # 		trial_act,
            # 	) = get_reliability(
            # 		firingstats_tmp["trial_map"][n, 1:, ...],
            # 		firingstats_tmp["map"][n, ...],
            # 		fields["parameter"][n, ...],
            # 		f,
            # 	)
            # 	# self.fields['reliability'][c,s,f] = rel
            # 	# self.fields['max_rate'][c,s,f] = max_rate
            # 	print
            # 	self.fields["trial_act"][
            # 		c, s, f, : len(trial_act)
            # 	] = trial_act

            # self.fields['trial_act'][c,s,f,:self.behavior['trial_ct'][s]] = firingstats_tmp['trial_field'][n,f,:]
            # print(self.fields['trial_act'][c,s,:,:self.behavior['trial_ct'][s]])
            # self.fields['reliability'][c,s,:] = fields['reliability'][n,:self.params['field_count_max']]

        # self.save([False,False,False,True,False])
        t_end = time.time()
        print("Fields obtained and saved, time spend: %6.4f" % (t_end - t_start))

    ### calculate shifts within clusters
    def compareSessions(self, n_processes=0):

        self.prepare_dicts(which=["compare"])

        self.compare = {
            "pointer": sp.sparse.lil_matrix(
                (
                    self.data["nC"],
                    self.data["nSes"] ** 2 * self.params["field_count_max"] ** 2,
                )
            ),
            "shifts": [],
            "shifts_distr": [],
            "inter_active": [],
            "inter_coding": [],
        }

        t_start = time.time()

        if n_processes > 1:
            pool = get_context("spawn").Pool(n_processes)
            loc = np.copy(self.fields["location"][..., 0])
            loc[~self.status["fields"]] = np.nan
            res = pool.starmap(
                get_field_shifts,
                zip(
                    self.status["activity"],
                    self.fields["p_x"],
                    loc,
                    itertools.repeat(self.data["nbin"], self.data["nC"]),
                ),
            )  # self.data['nC']))

        else:
            loc = np.copy(self.fields["location"][..., 0])
            loc[~self.status["fields"]] = np.nan
            res = []
            for c in range(self.data["nC"]):
                res.append(
                    get_field_shifts(
                        self.status["activity"][c, ...],
                        self.fields["p_x"][c, ...],
                        loc[c, ...],
                        nbin=self.data["nbin"],
                    )
                )
            # print('please use parallel processing for this')
        i = 0
        for c, r in enumerate(res):
            for key in r["shifts_distr"]:
                i += 1
                self.compare["pointer"][c, key] = i
                # print(r['shifts'][key])
                self.compare["shifts"].append(r["shifts"][key])
                self.compare["shifts_distr"].append(r["shifts_distr"][key])
                self.compare["inter_active"].append(r["inter_active"][key])
                self.compare["inter_coding"].append(r["inter_coding"][key])

        self.compare["shifts"] = np.array(self.compare["shifts"])

        self.compare["inter_active"] = np.array(self.compare["inter_active"])
        self.compare["inter_coding"] = np.array(self.compare["inter_coding"])

        self.compare["shifts_distr"] = sp.sparse.csr_matrix(
            self.compare["shifts_distr"]
        )
        self.compare["pointer"] = self.compare["pointer"].tocoo()

        # else:
        # self.compare = pickleData([],self.params['svCompare'],'load')
        t_end = time.time()
        # print('Place field shifts calculated - time %5.3f'%(t_end-t_start))

    def update_status(
        self,
        complete=True,
        **kwargs,
    ):
        """
        TODO:
            [ ] if there is time: remove first index (detected component) from status, as it is redundant with "component_detected" and not required for further analysis
        """
        # print('further, implement method to calculate inter-coding intervals, etc, after updating statuses')
        self.set_thresholds(**kwargs)
        self.classify_components()

        # t_start = time.time()

        ### reset all statuses
        self.status["activity"] = np.zeros(
            (self.data["nC"], self.data["nSes"], 6), "bool"
        )

        self.status["activity"][..., 0] = self.status["component_detected"]

        """
            status follows hierarchical structure: each index up to 2 is a subset of the previous one
            0: is active neuron (passing thresholds of SNR, r_val and cnn and matching probability to cluster)
            1: firing rate above some level
            2: is place cell
        """

        self.status["activity"][..., 1] = (
            self.stats["firingrate"] >= self.thr["firingrate"]
        ) & self.status["component_detected"]

        self.status["activity"][~self.status["clusters"], :, :] = False

        if complete:
            print("update fields")

            # self.fields["status"] = np.zeros(
            #     (self.data["nC"], self.data["nSes"], self.params["field_count_max"]),
            #     "int",
            # )
            ### place field: amplitude, A_rate, p_mass, CI-width, width(?),

            ## characterize place cells by whether their field passes some thresholds:
            A_rate = (
                self.fields["amplitude"][..., 0]
                / self.fields["baseline"][..., 0, np.newaxis]
            )
            CI_width = np.mod(
                self.fields["location"][..., 2] - self.fields["location"][..., 1],
                self.data["nbin"],
            )

            # morphed_A0_thr = self.thr["A_0"] - self.fields["reliability"] / 2

            idx_fields = (
                (self.fields["baseline"][..., 0, np.newaxis] > self.thr["A_0"])
                & (self.fields["amplitude"][..., 0] > self.thr["A"])
                # & (A_rate > self.thr["A_rate"])
                & (self.fields["width"][..., 0] > self.thr["sigma"])
                # & (self.fields["posterior_mass"] > self.thr["p_mass"])
                & (self.fields["Bayes_factor"] > self.thr["Bayes"])
                & (self.fields["reliability"] > self.thr["reliability"])
            )

            # self.fix_missed_loc()

            self.status["fields"] = idx_fields

            ##   - Bayes factor, MI(val,p_val,z_score)
            # self.stats["MI_p_value"][self.stats["MI_p_value"] == 0.001] = 10 ** (
            #     -10
            # )  ## need this - can't get any lower than 0.001 with 1000 shuffles...
            idx_PC = np.ones((self.data["nC"], self.data["nSes"]), "bool")
            # for s in np.where(self.status["sessions"])[0]:
            #     idx_PC[:, s] = fdr_control(
            #         self.stats["MI_p_value"][:, s], self.thr["alpha"]
            #     )
            idx_PC = (
                idx_PC
                & np.any(idx_fields, -1)
                # & (self.stats["MI_value"] > self.thr["MI"])
            )

            self.status["activity"][..., 2] = idx_PC & self.status["activity"][..., 1]

            self.status["fields"] = (
                self.status["fields"] & self.status["activity"][..., 2][..., np.newaxis]
            )

            # print(self.status['activity'].sum(axis=0))

            location_dependent_PC = False
            if location_dependent_PC:

                # & (~np.isnan(self.fields['location'][...,0]))
                # idx_reward = (self.fields['location'][...,0]<=self.para['zone_idx']['reward'][-1]) & \
                # (self.fields['location'][...,0]>=self.para['zone_idx']['reward'][0])
                # self.fields['status'][idx_reward] = 4
                # self.fields['status'][~self.status[...,2],:] = False

                self.session_data = get_session_specifics(
                    self.data["mouse"], self.data["nSes"]
                )

                nbin = self.data["nbin"]
                ct_field_remove = 0

                for c in range(self.data["nC"]):
                    for s in range(self.data["nSes"]):
                        # print(self.session_data['RW_pos'])
                        # print(self.session_data['GT_pos'])
                        # rw_pos = self.session_data['RW_pos'][s,:]
                        # gt_pos = self.session_data['GT_pos'][s,:]

                        rw_pos = [50, 70]
                        gt_pos = [5, 6]

                        if self.status["activity"][c, s, 2]:

                            for f in np.where(self.status["fields"][c, s, :])[
                                0
                            ]:  # range(self.fields['nModes'][c,s]):
                                break_it = False
                                # if idx_fields[c,s,f]:
                                field_loc = self.fields["location"][c, s, f, 0]
                                field_sig = self.fields["width"][c, s, f, 0]
                                field_bin_l = int(field_loc - field_sig) % nbin
                                field_bin_r = int(field_loc + field_sig + 1) % nbin

                                field_bool = np.zeros(nbin, "bool")
                                if field_bin_l < field_bin_r:
                                    field_bool[field_bin_l:field_bin_r] = True
                                else:
                                    field_bool[field_bin_l:] = True
                                    field_bool[:field_bin_r] = True

                                ## if cell shows several place fields, check whether they are highly correlated and remove, if so
                                for ff in np.where(self.status["fields"][c, s, :])[0]:
                                    if f == ff:
                                        continue
                                    field2_loc = self.fields["location"][c, s, ff, 0]
                                    field2_sig = self.fields["width"][c, s, ff, 0]
                                    field_bin_l = int(field2_loc - field2_sig) % nbin
                                    field_bin_r = (
                                        int(field2_loc + field2_sig + 1) % nbin
                                    )

                                    field2_bool = np.zeros(nbin, "bool")
                                    if field_bin_l < field_bin_r:
                                        field2_bool[field_bin_l:field_bin_r] = True
                                    else:
                                        field2_bool[field_bin_l:] = True
                                        field2_bool[:field_bin_r] = True

                                    # corr_trial = np.corrcoef(
                                    #     self.fields["trial_act"][
                                    #         c, s, f, : self.behavior["trial_ct"][s]
                                    #     ],
                                    #     self.fields["trial_act"][
                                    #         c, s, ff, : self.behavior["trial_ct"][s]
                                    #     ],
                                    # )[0, 1]

                                    # if ((field_bool & field2_bool).sum() > 3) & (
                                    #     corr_trial > 0.3
                                    # ):
                                    #     ct_field_remove += 1
                                    #     if (
                                    #         self.fields["Bayes_factor"][c, s, f]
                                    #         > self.fields["Bayes_factor"][c, s, ff]
                                    #     ):
                                    #         self.status["fields"][c, s, ff] = False
                                    #     else:
                                    #         self.status["fields"][c, s, f] = False
                                    #         break_it = True
                                    #         break  ## continue, when this field is bad

                                if break_it:
                                    continue

                                ## set status of field according to which region of the VR the field encodes
                                if (
                                    rw_pos[0]
                                    <= self.fields["location"][c, s, f, 0]
                                    <= rw_pos[1]
                                ):
                                    self.fields["status"][c, s, f] = 4
                                elif (
                                    gt_pos[0]
                                    <= self.fields["location"][c, s, f, 0]
                                    <= gt_pos[1]
                                ):
                                    self.fields["status"][c, s, f] = 3
                                else:
                                    self.fields["status"][c, s, f] = 5
                                self.status["activity"][
                                    c, s, self.fields["status"][c, s, f]
                                ] = True

                            self.fields["nModes"][c, s] = np.count_nonzero(
                                self.fields["status"][c, s, :]
                            )

                print("fields removed: %d" % ct_field_remove)
        t_end = time.time()
        # print('PC-characterization done. Time taken: %7.5f'%(t_end-t_start))

    def calculate_recurrence(self, n_shuffles=10):

        self.recurrence = {
            "active": {
                "all": np.full((self.data["nSes"], self.data["nSes"]), np.nan),
                "continuous": np.full((self.data["nSes"], self.data["nSes"]), np.nan),
                "overrepresentation": np.full(
                    (self.data["nSes"], self.data["nSes"]), np.nan
                ),
            },
            "coding": {
                "all": np.full((self.data["nSes"], self.data["nSes"]), np.nan),
                "ofactive": np.full((self.data["nSes"], self.data["nSes"]), np.nan),
                "continuous": np.full((self.data["nSes"], self.data["nSes"]), np.nan),
                "overrepresentation": np.full(
                    (self.data["nSes"], self.data["nSes"]), np.nan
                ),
            },
        }
        N = {
            "active": self.status["activity"][:, :, 1].sum(0),
            "coding": self.status["activity"][:, :, 2].sum(0),
        }

        for s in tqdm(range(self.data["nSes"]), leave=False):

            nC = self.status["clusters"].sum()

            if N["active"][s] == 0:
                continue
            overlap_act = self.status["activity"][
                self.status["activity"][:, s, 1], :, 1
            ].sum(0)
            overlap_PC = self.status["activity"][
                self.status["activity"][:, s, 2], :, 2
            ].sum(0)

            self.recurrence["active"]["all"][s, 1 : (self.data["nSes"] - s)] = (
                overlap_act[s + 1 :] / N["active"][s + 1 :]
            )

            self.recurrence["coding"]["all"][s, 1 : (self.data["nSes"] - s)] = (
                overlap_PC[s + 1 :] / N["coding"][s + 1 :]
            )
            for i, s1 in enumerate(range(s + 1, self.data["nSes"])):
                self.recurrence["coding"]["ofactive"][s, i + 1] = (
                    overlap_PC[s1]
                    / self.status["activity"][
                        self.status["activity"][:, s, 2], s1, 1
                    ].sum()
                )

            rand_pull_act = np.zeros((self.data["nSes"] - s, n_shuffles)) * np.nan
            rand_pull_PC = np.zeros((self.data["nSes"] - s, n_shuffles)) * np.nan

            for s2 in range(s + 1, self.data["nSes"]):
                if (N["active"][s] == 0) or (N["active"][s2] == 0):
                    continue
                rand_pull_act[s2 - s, :] = (
                    np.random.choice(nC, (n_shuffles, N["active"][s])) < N["active"][s2]
                ).sum(1)

                offset = N["active"][s] - overlap_act[s2]
                randchoice_1 = np.random.choice(
                    N["active"][s], (n_shuffles, N["coding"][s])
                )
                randchoice_2 = np.random.choice(
                    np.arange(offset, offset + N["active"][s2]),
                    (n_shuffles, N["coding"][s2]),
                )
                for l in range(n_shuffles):
                    rand_pull_PC[s2 - s, l] = np.isin(
                        randchoice_1[l, :], randchoice_2[l, :]
                    ).sum()

                ### find continuously coding neurons
                self.recurrence["active"]["continuous"][s, s2 - s] = (
                    self.status["activity"][:, s : s2 + 1, 1].sum(1) == (s2 - s + 1)
                ).sum() / N["active"][s2]
                self.recurrence["coding"]["continuous"][s, s2 - s] = (
                    self.status["activity"][:, s : s2 + 1, 2].sum(1) == (s2 - s + 1)
                ).sum() / N["coding"][s2]

            self.recurrence["active"]["overrepresentation"][
                s, : self.data["nSes"] - s
            ] = (overlap_act[s:] - np.nanmean(rand_pull_act, 1)) / np.nanstd(
                rand_pull_act, 1
            )
            self.recurrence["coding"]["overrepresentation"][
                s, : self.data["nSes"] - s
            ] = (overlap_PC[s:] - np.nanmean(rand_pull_PC, 1)) / np.nanstd(
                rand_pull_PC, 1
            )

    def calculate_field_firingrates(self, sd_r=-1):

        oof_frate = (
            np.zeros((self.data["nC"], self.data["nSes"])) * np.nan
        )  # out-of-field firingrate
        if_frate = (
            np.zeros(
                (self.data["nC"], self.data["nSes"], self.params["field_count_max"])
            )
            * np.nan
        )  # in-field firingrate

        for s in tqdm(range(self.data["nSes"]), leave=False):

            if not self.status["sessions"][s]:
                continue

            dataBH = prepare_behavior_from_file(self.get_path("behavior", s))

            if self.paths["neuron_detection"][s].exists():
                ld = load_data(self.paths["neuron_detection"][s])
                S = ld["S"][:, dataBH["active"]]

                c_arr = np.where(np.isfinite(self.matching["IDs"][:, s]))[0]
                n_arr = self.matching["IDs"][c_arr, s].astype("int")

                for c, n in zip(c_arr, n_arr):
                    bool_arr = np.ones(S.shape[1], "bool")
                    if self.status["activity"][c, s, 2]:
                        for f in np.where(self.fields["status"][c, s, :])[0]:
                            field_bin = int(self.fields["location"][c, s, f, 0])
                            field_bin_l = (
                                int(
                                    self.fields["location"][c, s, f, 0]
                                    - self.fields["width"][c, s, f, 0]
                                )
                                % self.data["nbin"]
                            )
                            field_bin_r = (
                                int(
                                    self.fields["location"][c, s, f, 0]
                                    + self.fields["width"][c, s, f, 0]
                                    + 1
                                )
                                % self.data["nbin"]
                            )
                            if field_bin_l < field_bin_r:
                                bool_arr[
                                    (dataBH["binpos"] > field_bin_l)
                                    & (dataBH["binpos"] < field_bin_r)
                                ] = False
                            else:
                                bool_arr[
                                    (dataBH["binpos"] > field_bin_l)
                                    | (dataBH["binpos"] < field_bin_r)
                                ] = False
                    oof_frate[c, s], _, _ = get_firingrate(
                        S[n, bool_arr], self.params["f"], sd_r=sd_r
                    )

                    if self.status["activity"][c, s, 2]:

                        for f in np.where(self.status["fields"][c, s, :])[0]:
                            bool_arr = np.ones(S.shape[1], "bool")
                            field_bin = int(self.fields["location"][c, s, f, 0])
                            field_bin_l = (
                                int(
                                    self.fields["location"][c, s, f, 0]
                                    - self.fields["width"][c, s, f, 0]
                                )
                                % self.data["nbin"]
                            )
                            field_bin_r = (
                                int(
                                    self.fields["location"][c, s, f, 0]
                                    + self.fields["width"][c, s, f, 0]
                                    + 1
                                )
                                % self.data["nbin"]
                            )
                            # print(field_bin_l,field_bin_r)
                            if field_bin_l < field_bin_r:
                                bool_arr[
                                    (dataBH["binpos"] < field_bin_l)
                                    | (dataBH["binpos"] > field_bin_r)
                                ] = False
                            else:
                                bool_arr[
                                    (dataBH["binpos"] < field_bin_l)
                                    & (dataBH["binpos"] > field_bin_r)
                                ] = False

                            for t in range(dataBH["trials"]["ct"]):
                                if ~self.fields["trial_act"][c, s, f, t]:
                                    bool_arr[
                                        dataBH["trials"]["start"][t] : dataBH["trials"][
                                            "start"
                                        ][t + 1]
                                    ] = False

                            if_frate[c, s, f], _, _ = get_firingrate(
                                S[n, bool_arr], self.params["f"], sd_r=sd_r
                            )
        self.stats["oof_firingrate_adapt"] = oof_frate
        self.stats["if_firingrate_adapt"] = if_frate
        return

    @timing
    def calculate_placefield_stability(
        self,
        dsMax,
        celltype="all",
        n_processes=8,
        N_bs=100,
        p_keys=["all", "cont", "mix", "discont", "silent_mix", "silent"],
    ):

        nSes = self.data["nSes"]
        nbin = self.data["nbin"]

        s_bool = np.ones(nSes, "bool")
        s_bool[~self.status["sessions"]] = False

        ## setting up dictionary for stability
        p = {}
        p_stats = {
            "mean": np.full((dsMax, 4), np.nan),
            "CI": np.full((dsMax, 2, 4), np.nan),
            "std": np.full((dsMax, 4), np.nan),
        }
        for key in p_keys:
            p[key] = copy.deepcopy(p_stats)

        ## obtaining all cluster, session and field indices of shifts
        s1_shifts, s2_shifts, _, _ = np.unravel_index(
            self.compare["pointer"].col,
            (
                nSes,
                nSes,
                self.params["field_count_max"],
                self.params["field_count_max"],
            ),
        )
        c_shifts = self.compare["pointer"].row

        ## filter shifts by celltype and active sessions
        if celltype == "all":
            idx_celltype = self.status["activity"][c_shifts, s1_shifts, 2]
        if celltype == "gate":
            idx_celltype = self.status["activity"][c_shifts, s1_shifts, 3]
        if celltype == "reward":
            idx_celltype = self.status["activity"][c_shifts, s1_shifts, 4]
        idx_celltype = idx_celltype & s_bool[s1_shifts] & s_bool[s2_shifts]

        ## get shift distributions by parallel or serial processing
        if n_processes > 1:
            pool = get_context("spawn").Pool(n_processes)
            # pool = mp.Pool(n_processes)
            res = pool.starmap(
                get_shift_distribution,
                zip(
                    range(1, dsMax),
                    itertools.repeat(self.compare),
                    itertools.repeat(
                        (nSes, nbin, N_bs, idx_celltype, s1_shifts, s2_shifts, p_keys)
                    ),
                ),
            )
            pool.close()
        else:
            res = []
            for ds in range(1, dsMax):
                print((nSes, nbin, N_bs, idx_celltype, s1_shifts, s2_shifts, p_keys))
                res.append(
                    get_shift_distribution(
                        ds,
                        self.compare,
                        (nSes, nbin, N_bs, idx_celltype, s1_shifts, s2_shifts, p_keys),
                    )
                )

        ## hand over and return results
        for ds, r in enumerate(res):
            for pop in r.keys():
                for key in r[pop].keys():
                    p[pop][key][ds, ...] = r[pop][key]

        return p

    def get_transition_prob(self, which=["p_post_c", "p_post_s"]):

        status_arr = ["act", "code", "stable"]

        p_posts = ["", "_nodepend", "_RW", "_GT", "_nRnG"]

        nSes = self.data["nSes"]
        nC = self.data["nC"]

        ds_max = 20

        if "p_post_s" in which:

            ## initialize dictionaries
            for p_post in p_posts:
                self.stats[f"p_post{p_post}_s"] = {}
            self.stats["p_pre_s"] = {}

            for status_key in status_arr:

                for p_post in p_posts:
                    self.stats[f"p_post{p_post}_s"][status_key] = {}
                self.stats["p_pre_s"][status_key] = {}

                for status2_key in status_arr:

                    for p_post in p_posts:
                        self.stats[f"p_post{p_post}_s"][status_key][status2_key] = (
                            np.full((nSes, ds_max + 1, 2), np.nan)
                        )
                    self.stats["p_pre_s"][status_key][status2_key] = np.full(
                        (nSes, ds_max + 1, 2), np.nan
                    )

            status, status_dep = get_status_arr(self)
            for key in status_arr:
                status[key] = status[key][self.status["clusters"], ...]
                status_dep[key] = status_dep[key][self.status["clusters"], ...]
            # status['act'] = status['act'][cluster.stats['cluster_bool'],:]
            # status['act'] = status['act'][cluster.stats['cluster_bool'],:]

            for ds in range(ds_max):

                for s in np.where(self.status["sessions"])[0][:-ds]:
                    if self.status["sessions"][s + ds]:

                        # loc = self.fields['location'][self.status['clusters'],s,:,0]
                        # idx_gt = (loc>GT_pos[0]) & (loc<GT_pos[1])
                        # idx_rw = (loc>RW_pos[0]) & (loc<RW_pos[1])

                        for key in status_arr:
                            for key2 in status_arr:
                                # if key i

                                if s - ds >= 0 and self.status["sessions"][s - ds]:
                                    status_pos = status[key2][
                                        status[key][:, s, 1]
                                        & status_dep[key2][:, s - ds],
                                        s - ds,
                                        ds,
                                    ]
                                    self.stats["p_pre_s"][key][key2][s, ds, 0] = (
                                        np.nanmean(status_pos)
                                    )
                                    status_neg = status[key2][
                                        (~status[key][:, s, 1])
                                        & status_dep[key][:, s]
                                        & status_dep[key2][:, s - ds],
                                        s - ds,
                                        ds,
                                    ]
                                    self.stats["p_pre_s"][key][key2][s, ds, 1] = (
                                        np.nanmean(status_neg)
                                    )

                                if s + ds < nSes and self.status["sessions"][s + ds]:
                                    status_pos = status[key2][
                                        status[key][:, s, 1]
                                        & status_dep[key2][:, s + ds],
                                        s + ds,
                                        ds,
                                    ]
                                    self.stats["p_post_s"][key][key2][s, ds, 0] = (
                                        np.nanmean(status_pos)
                                    )
                                    status_neg = status[key2][
                                        (~status[key][:, s, 1])
                                        & status_dep[key][:, s]
                                        & status_dep[key2][:, s + ds],
                                        s + ds,
                                        ds,
                                    ]
                                    self.stats["p_post_s"][key][key2][s, ds, 1] = (
                                        np.nanmean(status_neg)
                                    )

                                    status_pos = status[key2][
                                        status[key][:, s, 1], s + ds, ds
                                    ]
                                    self.stats["p_post_nodepend_s"][key][key2][
                                        s, ds, 0
                                    ] = np.nanmean(status_pos)
                                    status_neg = status[key2][
                                        (~status[key][:, s, 1]) & status_dep[key][:, s],
                                        s + ds,
                                        ds,
                                    ]
                                    self.stats["p_post_nodepend_s"][key][key2][
                                        s, ds, 1
                                    ] = np.nanmean(status_neg)

                                    if key in ["code", "stable"]:
                                        idx_gt = np.any(
                                            self.fields["status"][
                                                self.status["clusters"], s, :
                                            ]
                                            == 3,
                                            1,
                                        )
                                        idx_rw = np.any(
                                            self.fields["status"][
                                                self.status["clusters"], s, :
                                            ]
                                            == 4,
                                            1,
                                        )
                                        idx_nRnG = np.any(
                                            self.fields["status"][
                                                self.status["clusters"], s, :
                                            ]
                                            == 5,
                                            1,
                                        )

                                        status_pos = status[key2][
                                            status[key][:, s, 1]
                                            & status_dep[key2][:, s + ds]
                                            & idx_rw,
                                            s + ds,
                                            ds,
                                        ]
                                        self.stats["p_post_RW_s"][key][key2][
                                            s, ds, 0
                                        ] = np.nanmean(status_pos)
                                        status_neg = status[key2][
                                            (~status[key][:, s, 1])
                                            & status_dep[key][:, s]
                                            & status_dep[key2][:, s + ds]
                                            & idx_rw,
                                            s + ds,
                                            ds,
                                        ]
                                        self.stats["p_post_RW_s"][key][key2][
                                            s, ds, 1
                                        ] = np.nanmean(status_neg)

                                        status_pos = status[key2][
                                            status[key][:, s, 1]
                                            & status_dep[key2][:, s + ds]
                                            & idx_gt,
                                            s + ds,
                                            ds,
                                        ]
                                        self.stats["p_post_GT_s"][key][key2][
                                            s, ds, 0
                                        ] = np.nanmean(status_pos)
                                        status_neg = status[key2][
                                            (~status[key][:, s, 1])
                                            & status_dep[key][:, s]
                                            & status_dep[key2][:, s + ds]
                                            & idx_gt,
                                            s + ds,
                                            ds,
                                        ]
                                        self.stats["p_post_GT_s"][key][key2][
                                            s, ds, 1
                                        ] = np.nanmean(status_neg)

                                        status_pos = status[key2][
                                            status[key][:, s, 1]
                                            & status_dep[key2][:, s + ds]
                                            & idx_nRnG,
                                            s + ds,
                                            ds,
                                        ]
                                        self.stats["p_post_nRnG_s"][key][key2][
                                            s, ds, 0
                                        ] = np.nanmean(status_pos)
                                        status_neg = status[key2][
                                            (~status[key][:, s, 1])
                                            & status_dep[key][:, s]
                                            & status_dep[key2][:, s + ds]
                                            & idx_nRnG,
                                            s + ds,
                                            ds,
                                        ]
                                        self.stats["p_post_nRnG_s"][key][key2][
                                            s, ds, 1
                                        ] = np.nanmean(status_neg)

        if "p_post_c" in which:
            status, status_dep = get_status_arr(self)
            status_above = {}
            status_above["act"] = ~status["code"]
            status_above["code"] = ~status["stable"]
            status_above["stable"] = np.ones_like(status["act"], "bool")

            self.stats["p_post_c"] = {}
            for status_key in status_arr:
                self.stats["p_post_c"][status_key] = {}
                for status2_key in status_arr:
                    self.stats["p_post_c"][status_key][status2_key] = (
                        np.zeros((nC, ds_max + 1, 2)) * np.nan
                    )

            for ds in tqdm(range(1, ds_max)):
                for c in np.where(self.status["clusters"])[0]:

                    counts = {}
                    for status_key in status_arr:
                        counts[status_key] = {}
                        for status2_key in status_arr:
                            counts[status_key][status2_key] = np.zeros((2, 2))

                    for s in np.where(self.status["sessions"])[0][:-ds]:
                        if self.status["sessions"][s + ds]:

                            for status_key in status_arr:
                                for status2_key in status_arr:
                                    if status[status_key][
                                        c, s, 1
                                    ]:  # & status_above[status_key][c,s,1]:
                                        if status_dep[status2_key][c, s + ds]:
                                            counts[status_key][status2_key][0, 0] += 1
                                            if status[status2_key][c, s + ds, ds]:
                                                counts[status_key][status2_key][
                                                    0, 1
                                                ] += 1

                                    if status_dep[status_key][c, s] & (
                                        ~status[status_key][c, s, 1]
                                    ):
                                        if status_dep[status2_key][c, s + ds]:
                                            counts[status_key][status2_key][1, 0] += 1
                                            if status[status2_key][
                                                c, s + ds, ds
                                            ]:  # & status_dep[status2_key][c,s+ds]:
                                                counts[status_key][status2_key][
                                                    1, 1
                                                ] += 1

                    for status_key in status_arr:
                        for status2_key in status_arr:
                            self.stats["p_post_c"][status_key][status2_key][
                                c, ds, 0
                            ] = (
                                counts[status_key][status2_key][0, 1]
                                / counts[status_key][status2_key][0, 0]
                                if counts[status_key][status2_key][0, 0] > 0
                                else np.nan
                            )
                            self.stats["p_post_c"][status_key][status2_key][
                                c, ds, 1
                            ] = (
                                counts[status_key][status2_key][1, 1]
                                / counts[status_key][status2_key][1, 0]
                                if counts[status_key][status2_key][1, 0] > 0
                                else np.nan
                            )

    def get_locTransition_prob(self, which=["recruit", "dismiss", "stable"]):

        ## for each location, get probability of
        ### 1. recruitment (from silent / non-coding / coding)
        ### 2. stability of place fields
        ### 3. dismissal (towards silent / non-coding / coding)
        nSes = self.data["nSes"]
        nbin = self.data["nbin"]

        SD = 1.96
        sig_theta = self.stability["all"]["mean"][0, 2]
        stab_thr = SD * sig_theta

        self.stats["transition"] = {
            "recruitment": np.zeros((nSes, nbin, 3)) * np.nan,
            "stabilization": np.zeros((nSes, nbin)),
            "dismissal": np.zeros((nSes, nbin, 3)) * np.nan,
        }

        for s in np.where(self.status["sessions"])[0]:
            if self.status["sessions"][s - 1]:

                ### recruitment
                idx_recruit_silent = (
                    ~self.status["activity"][:, s - 1, 1]
                ) & self.status["activity"][
                    :, s, 2
                ]  # neurons turning from silence to coding
                idx_recruit_active = (
                    self.status["activity"][:, s - 1, 1]
                    & (~self.status["activity"][:, s - 1, 2])
                    & self.status["activity"][:, s, 2]
                )  # neurons turning from silence to coding
                idx_recruit_coding = (
                    self.status["activity"][:, s - 1, 2]
                    & self.status["activity"][:, s, 2]
                )  # neurons turning from silence to coding

                idx_fields = np.where(
                    idx_recruit_silent[:, np.newaxis] & self.status["fields"][:, s, :]
                )
                self.stats["transition"]["recruitment"][s, :, 0] = np.nansum(
                    self.fields["p_x"][idx_fields[0], s, idx_fields[1]], 0
                )

                idx_fields = np.where(
                    idx_recruit_active[:, np.newaxis] & self.status["fields"][:, s, :]
                )
                self.stats["transition"]["recruitment"][s, :, 1] = np.nansum(
                    self.fields["p_x"][idx_fields[0], s, idx_fields[1]], 0
                )

                idx_fields = np.where(
                    idx_recruit_coding[:, np.newaxis] & self.status["fields"][:, s, :]
                )
                self.stats["transition"]["recruitment"][s, :, 2] = np.nansum(
                    self.fields["p_x"][idx_fields[0], s, idx_fields[1]], 0
                )

                ### dismissal
                idx_dismiss_silent = self.status["activity"][:, s - 1, 2] & (
                    ~self.status["activity"][:, s, 1]
                )  # neurons turning from silence to coding
                idx_dismiss_active = (
                    self.status["activity"][:, s - 1, 2]
                    & (~self.status["activity"][:, s, 2])
                    & self.status["activity"][:, s, 1]
                )  # neurons turning from silence to coding
                idx_dismiss_coding = (
                    self.status["activity"][:, s - 1, 2]
                    & self.status["activity"][:, s, 2]
                )  # neurons turning from silence to coding

                idx_fields = np.where(
                    idx_dismiss_silent[:, np.newaxis]
                    & self.status["fields"][:, s - 1, :]
                )
                self.stats["transition"]["dismissal"][s, :, 0] = np.nansum(
                    self.fields["p_x"][idx_fields[0], s - 1, idx_fields[1], :], 0
                )

                idx_fields = np.where(
                    idx_dismiss_active[:, np.newaxis]
                    & self.status["fields"][:, s - 1, :]
                )
                self.stats["transition"]["dismissal"][s, :, 1] = np.nansum(
                    self.fields["p_x"][idx_fields[0], s - 1, idx_fields[1], :], 0
                )

                idx_fields = np.where(
                    idx_dismiss_coding[:, np.newaxis]
                    & self.status["fields"][:, s - 1, :]
                )
                self.stats["transition"]["dismissal"][s, :, 2] = np.nansum(
                    self.fields["p_x"][idx_fields[0], s - 1, idx_fields[1], :], 0
                )

                ### stabilization
                idx_stabilization = (
                    self.status["activity"][:, s - 1, 2]
                    & self.status["activity"][:, s, 2]
                )  # neurons turning from silence to coding
                for c in np.where(idx_stabilization)[0]:
                    field_ref = self.fields["location"][
                        c, s - 1, self.status["fields"][c, s - 1, :], 0
                    ]
                    field_compare = self.fields["location"][
                        c, s, self.status["fields"][c, s, :], 0
                    ]

                    d = np.abs(
                        np.mod(
                            field_ref[:, np.newaxis]
                            - field_compare[np.newaxis, :]
                            + nbin / 2,
                            nbin,
                        )
                        - nbin / 2
                    )
                    if np.any(d < stab_thr):
                        # print(field_ref)
                        # print(d)
                        f_stable = np.where(d < stab_thr)[0][0]
                        # print(f_stable)
                        # print(np.round(field_ref[f_stable]))
                        self.stats["transition"]["stabilization"][
                            s, int(field_ref[f_stable])
                        ] += 1

    def fix_missed_loc(self):

        idx_c, idx_s, idx_f = np.where(
            np.isnan(self.fields["location"][..., 0])
            & (~np.isnan(self.fields["baseline"][..., 0, np.newaxis]))
        )

        for c, s, f in zip(idx_c, idx_s, idx_f):
            loc = get_average(
                np.arange(100),
                self.fields["p_x"][c, s, f, :],
                periodic=True,
                bounds=[0, 100],
            )
            self.fields["location"][c, s, f, 0] = loc
            # print('field at c=%d, s=%d:'%(c,s))
            # print(self.fields['location'][c,s,f,:])
            # print(loc)

    def save(self, svBool=np.ones(5).astype("bool")):
        pass
        # if svBool[0]:
        #     pickleData(self.matching['IDs'],self.paths['svIDs'],'save')
        # if svBool[1]:
        #     pickleData(self.sessions,self.paths['svSessions'],'save')
        # if svBool[2]:
        #     pickleData(self.stats,self.params['svStats'],'save')
        # if svBool[3]:
        #     pickleData(self.fields,self.params['svPCs'],'save')
        # if svBool[4]:
        #     pickleData(self.compare,self.params['svCompare'],'save')

    def load(self, ldBool=np.ones(5).astype("bool")):
        # self.allocate_cluster()
        if ldBool[0]:
            self.matching["IDs"] = pickleData([], self.params["svIDs"], "load")
            self.data["nC"] = self.matching["IDs"].shape[0]
        if ldBool[1]:
            self.sessions = pickleData([], self.params["svSessions"], "load")
        if ldBool[2]:
            self.stats = pickleData([], self.params["svStats"], "load")
        if ldBool[3]:
            self.fields = pickleData([], self.params["svPCs"], "load")
        if ldBool[4]:
            self.compare = pickleData([], self.params["svCompare"], "load")

    ### ------------------------ CURRENTLY UNUSED CODE (START) ----------------------------- ###

    def get_trial_fr(self, n_processes):

        self.sessions["trials_fr"] = (
            np.zeros(
                (self.data["nC"], self.data["nSes"], self.sessions["trial_ct"].max())
            )
            * np.nan
        )
        for s, path in tqdm(enumerate(self.paths["sessions"])):

            c_arr = np.where(~np.isnan(self.matching["IDs"][:, s]))[0]
            n_arr = self.matching["IDs"][c_arr, s].astype("int")
            nCells = len(n_arr)

            # pathSession = self.params['paths']['session'][s]
            # pathLoad = os.path.join(path,'results_redetect.mat')
            ld = pickleData(None, self.params["paths"]["session"][s], "load")

            pool = get_context("spawn").Pool(n_processes)
            res = pool.starmap(
                get_firingrate,
                zip(ld["S"][n_arr, :], itertools.repeat(15), itertools.repeat(2)),
            )

            fr = np.zeros((self.data["nC"], ld["S"].shape[1]))
            for j, r in enumerate(res):
                fr[c_arr[j], :] = r[2]

            for i in range(self.sessions["trial_ct"][s]):
                t_start = self.sessions["trial_frames"][s][i]
                t_end = self.sessions["trial_frames"][s][i + 1]
                self.sessions["trials_fr"][:, s, i] = fr[:, t_start:t_end].sum(1) / (
                    (t_end - t_start) / self.para["f"]
                )


### ------------------------ CURRENTLY UNUSED CODE (END) ----------------------------- ###


def get_field_shifts(status, p_x, loc, nbin):
    nSes = status.shape[0]
    # nSes = 10
    nfields = p_x.shape[1]
    # nbin = p_x.shape[-1]
    # L_track = 100
    out = {"shifts": {}, "shifts_distr": {}, "inter_active": {}, "inter_coding": {}}

    for s1 in range(nSes):
        if not np.any(status[s1, 2:]):
            continue
        for s2 in range(s1 + 1, nSes):
            if not np.any(status[s2, 2:]):
                continue

            # f1_arr = np.where(~np.isnan(loc[s1,:]))[0]
            # f2_arr = np.where(~np.isnan(loc[s2,:]))[0]

            # f_norm = len(f1_arr)*len(f2_arr)
            # i=0
            # for f1 in f1_arr:
            # for f2 in f2_arr:
            # idx = np.ravel_multi_index((s1,s2,i),(nSes,nSes,nfields**2))
            # shifts, shifts_distr = periodic_distr_distance(p_x[s1,f1,:],p_x[s2,f2,:],nbin,L_track,mode='bootstrap')
            # shifts_distr /= f_norm

            d = np.abs(
                np.mod(loc[s1, :][:, np.newaxis] - loc[s2, :] + nbin / 2, nbin)
                - nbin / 2
            )
            # print(d)
            d[np.isnan(d)] = nbin
            f1, f2 = sp.optimize.linear_sum_assignment(d)
            for f in zip(f1, f2):

                if d[f] < nbin:
                    idx = np.ravel_multi_index(
                        (s1, s2, f[0], f[1]), (nSes, nSes, nfields, nfields)
                    )
                    # print(f[0], f[1])
                    shifts, shifts_distr = periodic_distr_distance(
                        p_x[s1, f[0], :],
                        p_x[s2, f[1], :],
                        nbin,
                        mode="bootstrap",
                        # mode="wasserstein",
                    )

                    out["shifts"][idx] = shifts
                    out["shifts_distr"][idx] = list(shifts_distr)

                    inter_active = status[s1 + 1 : s2, 1].sum()
                    inter_coding = status[s1 + 1 : s2, 2].sum()
                    if s2 - s1 == 1:
                        out["inter_active"][idx] = [inter_active, 1]
                        out["inter_coding"][idx] = [inter_coding, 1]
                    else:
                        out["inter_active"][idx] = [
                            inter_active,
                            inter_active / (s2 - s1 - 1),
                        ]
                        out["inter_coding"][idx] = [
                            inter_coding,
                            inter_coding / (s2 - s1 - 1),
                        ]
                    # i+=1

    return out


def get_shift_distribution(ds, compare, para):

    nSes, nbin, N_bs, idx_celltype, s1_shifts, s2_shifts, p_keys = para
    p = {}
    for key in p_keys:
        p[key] = {}

    Ds = s2_shifts - s1_shifts
    idx_ds = np.where((Ds == ds) & idx_celltype)[0]
    N_data = len(idx_ds)
    # print(N_data)

    idx_shifts = compare["pointer"].data[idx_ds].astype("int") - 1
    # shifts = compare['shifts'][idx_shifts]
    shifts_distr = compare["shifts_distr"][idx_shifts, :].toarray()

    for pop in p.keys():
        if pop == "all":
            idxes = np.ones(N_data, "bool")
        elif pop == "cont":
            idxes = compare["inter_coding"][idx_ds, 1] == 1
        elif pop == "mix":
            idxes = (
                (compare["inter_coding"][idx_ds, 1] > 0)
                & (compare["inter_coding"][idx_ds, 1] < 1)
            ) & (compare["inter_active"][idx_ds, 1] == 1)
        elif pop == "discont":
            idxes = (compare["inter_coding"][idx_ds, 1] == 0) & (
                compare["inter_active"][idx_ds, 1] == 1
            )
        elif pop == "silent_mix":
            idxes = (compare["inter_active"][idx_ds, 1] > 0) & (
                compare["inter_active"][idx_ds, 1] < 1
            )
        elif pop == "silent":
            idxes = compare["inter_active"][idx_ds, 1] == 0

        # p[pop]['mean'], p[pop]['std'], _ = bootstrap_shifts(fit_shift_model,cdf_shifts_ds[idxes,:],N_bs,nbin)
        p[pop]["mean"], p[pop]["CI"], p[pop]["std"], _ = bootstrap_shifts(
            fit_shift_model, shifts_distr[idxes, :], N_bs, nbin
        )
    return p


def bootstrap_shifts(fun, shifts, N_bs, nbin):

    N_data = len(shifts)
    if N_data == 0:
        return (
            np.full(4, np.nan),
            np.full((2, 4), np.nan),
            np.full(4, np.nan),
            np.full((2, nbin), np.nan),
        )

    samples = np.random.randint(0, N_data, (N_bs, N_data))

    # sample_randval = np.random.rand(N_bs,N_data)
    shift_distr_bs = np.zeros((N_bs, nbin))
    par = np.full((N_bs, 4), np.nan)
    for i in range(N_bs):
        shift_distr_bs[i, :] = shifts[samples[i, :], :].sum(0)
        shift_distr_bs[i, :] /= shift_distr_bs[i, :].sum()
        par[i, :], p_cov = fun(shift_distr_bs[i, :])

    # print(shift_distr_bs)
    p = np.nanmean(par, 0)
    p_CI = np.percentile(par, [2.5, 97.5], 0)
    p_std = np.nanstd(par, 0)

    return p, p_CI, p_std, shift_distr_bs


## fitting functions and options
F_shifts = lambda x, A0, A, sig, theta: A / (np.sqrt(2 * np.pi) * sig) * np.exp(
    -((x - theta) ** 2) / (2 * sig**2)
) + A0 / len(
    x
)  ## gaussian + linear offset


def fit_shift_model(data):
    p_bounds = ([0, 0, 0, -10], [1, 1, 50, 10])
    # shift_hist = np.histogram(data,np.linspace(-50,50,101),density=True)[0]
    # shift_hist[0] = shift_hist[1]
    # shift_hist /= shift_hist.sum()
    try:
        # return curve_fit(F_shifts,np.linspace(-49.5,49.5,100),shift_hist,bounds=p_bounds)
        # return curve_fit(F_shifts, np.linspace(-49.5, 49.5, 100), data, bounds=p_bounds)
        return curve_fit(F_shifts, np.linspace(-19.5, 19.5, 40), data, bounds=p_bounds)
    except:
        return np.zeros(4) * np.nan, np.nan
