import os, cv2, time, itertools, copy

from tqdm import *
import numpy as np
from matplotlib import pyplot as plt, colors, image as mpimg
from matplotlib_scalebar.scalebar import ScaleBar
import scipy as sp
import scipy.stats as sstats
from collections import Counter
from scipy.optimize import curve_fit

from multiprocessing import get_context

from .cluster_analysis import cluster_analysis

from .placefield_detection import get_firingrate

from .utils import (
    get_ICPI,
    get_dp,
    gauss_smooth,
    add_number,
    bootstrap_data,
    com,
    periodic_distr_distance,
    get_status_arr,
)

from .neuron_matching.utils import load_data


from caiman.utils.visualization import get_contours

from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter


class cluster_analysis_plots(cluster_analysis):
    """
    TODO:
        [ ] check old "TODO"s from below and update status
        [ ] move most (costly) analysis parts to analysis class, detach from plotting if not directly needed (such as parameter-dependent stability analysis)
            [ ] recurrence calculus
            [ ] stability analysis
            [ ] what else?
        [ ] somehow cut down on complexitiy of file to make it more readable and processable...
        [ ] run surrogate data testing for large scale PC-detection testing


    TODO:
        * read measurement time from file

        * Plot 45 - missing keys self.stats [p_post_s] - get_transition_prob function has some bugs
        * Plot 22 - missing key - ‘zone_mask’
        * Plot 41 - indexing issue
        * Plot 46 - missing key - ‘if_firingrate_adapt’
        * Plot 50, 52  - missing behaviour text file

    DONE:
        * Plot 12 - missing key - alignment['rotation_normal'] -> plot has been covered in neuron_matching.plot_alignment_statistics(s_compare)
        * Plot 15 - covered by plot_pv_correlation
        * Plot 16 - just added ’t_measures’ as dummy data
        * Plot 19 - variable dloc empty -> uhm - no. it's working :) (plot_hierarchy_interaction)
        * Plot 20 - completely changed into plot_parameter_dependent_neuron_numbers
        * Plot 191 - fixed get_transition_prob function and cleaned up plot function (some more to do)



    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_plots(self, sv_ext="png"):

        nSes = self.data["nSes"]
        self.pl_dat = plot_dat(
            self.data["mouse"], self.paths["figures"], nSes, self.data, sv_ext=sv_ext
        )

    def plotting(self):

        method_list = [
            func
            for func in dir(self)
            if callable(getattr(self, func)) and func.startswith("plot_")
        ]

        print("Please choose a method to plot:")
        for i, method in enumerate(method_list):
            print("\t%d: %s" % (i, method))

        choice = int(input("Enter the number of the method to plot: "))
        if choice < len(method_list):
            getattr(self, method_list[choice])()
        else:
            print("Invalid choice. Please try again.")
            self.plot_data()

    def plot_turnover_mechanism(self, mode="act", sv=False, N_bs=10):
        print("### plot cell activity statistics ###")
        s = 10

        # session_bool = np.pad(self.status['sessions'][ds:],(0,ds),constant_values=False) & np.pad(self.status['sessions'][:],(0,0),constant_values=False)

        s_bool = self.status["sessions"]
        state_label = "alpha" if (mode == "act") else "beta"

        ## reduce arrays to good clusters and sessions
        status_act = self.status["activity"][self.status["clusters"], :, 1]
        status_act = status_act[:, self.status["sessions"]]

        status_PC = self.status["activity"][self.status["clusters"], :, 2]
        status_PC = status_PC[:, self.status["sessions"]]

        nC_good, nSes_good = status_act.shape
        nSes_max = np.where(self.status["sessions"])[0][-1]

        active_neurons = status_act.mean(0)
        silent_neurons = (~status_act).mean(0)
        print(
            "active neurons: %.3g +/- %.3g"
            % (active_neurons.mean() * 100, active_neurons.std() * 100)
        )
        print(
            "silent neurons: %.3g +/- %.3g"
            % (silent_neurons.mean() * 100, silent_neurons.std() * 100)
        )

        coding_neurons = status_PC.sum(0) / status_act.sum(0)
        ncoding_neurons = ((~status_PC) & status_act).sum(0) / status_act.sum(0)
        print(
            "coding neurons: %.3g +/- %.3g"
            % (coding_neurons.mean() * 100, coding_neurons.std() * 100)
        )
        print(
            "non-coding neurons: %.3g +/- %.3g"
            % (ncoding_neurons.mean() * 100, coding_neurons.std() * 100)
        )

        p_act = np.count_nonzero(status_act) / (nC_good * nSes_good)
        p_PC = np.count_nonzero(status_PC) / np.count_nonzero(status_act)
        # print(p_PC)
        rnd_var_act = np.random.random(status_act.shape)
        rnd_var_PC = np.random.random(status_PC.shape)
        status_act_test = np.zeros(status_act.shape, "bool")
        status_act_test_rnd = np.zeros(status_act.shape, "bool")
        status_PC_test = np.zeros(status_PC.shape, "bool")
        status_PC_test_rnd = np.zeros(status_PC.shape, "bool")
        for c in range(nC_good):

            # status_act_test[c,:] = rnd_var_act[c,:] < (np.count_nonzero(status_act[c,:])/nSes_good)
            nC_act = status_act[c, :].sum()
            status_act_test[c, np.random.choice(nSes_good, nC_act, replace=False)] = (
                True
            )
            status_act_test_rnd[c, :] = rnd_var_act[c, :] < p_act
            # status_PC_test[c,status_act_test[c,:]] = rnd_var_PC[c,status_act_test[c,:]] < (np.count_nonzero(status_PC[c,:])/np.count_nonzero(status_act[c,:]))
            # print(status_act[c,:])
            # status_PC_test[c,status_act[c,:]] = rnd_var_PC[c,status_act[c,:]] < (np.count_nonzero(status_PC[c,:])/np.count_nonzero(status_act[c,:]))
            status_PC_test[
                c,
                np.where(status_act[c, :])[0][
                    np.random.choice(nC_act, status_PC[c, :].sum(), replace=False)
                ],
            ] = True
            status_PC_test_rnd[c, status_act[c, :]] = (
                rnd_var_PC[c, status_act[c, :]] < p_PC
            )

        status = status_act if mode == "act" else status_PC
        status_test = status_act_test if mode == "act" else status_PC_test

        fig = plt.figure(figsize=(7, 5), dpi=self.pl_dat.sv_opt["dpi"])

        ax_sketch = plt.axes([0.04, 0.875, 0.25, 0.1])
        self.pl_dat.add_number(fig, ax_sketch, order=1, offset=[-40, 10])
        if mode == "act":
            pic_path = "/home/wollex/Data/Science/PhD/Thesis/pics/sketches/neural_network_active.png"
        else:
            pic_path = "/home/wollex/Data/Science/PhD/Thesis/pics/sketches/neural_network_PC.png"

        if os.path.exists(pic_path):
            ax_sketch.axis("off")
            im = mpimg.imread(pic_path)
            ax_sketch.imshow(im)
            ax_sketch.set_xlim([0, im.shape[1]])

        if sv:  ## enable, when saving
            ### plot contours of two adjacent sessions
            # load data from both sessions
            pathLoad = self.paths["neuron_detection"][s]
            ld = load_data(pathLoad)
            A1 = ld["A"]
            Cn = ld["Cn"].transpose()
            Cn -= Cn.min()
            Cn /= Cn.max()

            pathLoad = self.paths["neuron_detection"][s + 1]
            ld = load_data(pathLoad)
            A2 = ld["A"]

            # adjust to same reference frame
            x_grid, y_grid = np.meshgrid(
                np.arange(0.0, self.params["dims"][0]).astype(np.float32),
                np.arange(0.0, self.params["dims"][1]).astype(np.float32),
            )
            x_remap = (
                x_grid
                - self.alignment["shift"][s + 1, 0]
                + self.alignment["shift"][s, 0]
                + self.alignment["flow"][s + 1, 0, :, :]
                - self.alignment["flow"][s, 0, :, :]
            ).astype("float32")
            y_remap = (
                y_grid
                - self.alignment["shift"][s + 1, 1]
                + self.alignment["shift"][s, 1]
                + self.alignment["flow"][s + 1, 1, :, :]
                - self.alignment["flow"][s, 1, :, :]
            ).astype("float32")

            ax_ROI = plt.axes([0.04, 0.48, 0.25, 0.375])
            # pl_dat.add_number(fig,ax_ROI,order=1,offset=[-25,25])
            # plot background, based on first sessions
            ax_ROI.imshow(Cn, origin="lower", clim=[0, 1], cmap="viridis")

            # plot contours occuring in first and in second session, only, and...
            # plot contours occuring in both sessions (taken from first session)

            twilight = plt.get_cmap("hsv")
            cNorm = colors.Normalize(vmin=0, vmax=100)
            scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=twilight)

            idx = 1 if mode == "act" else 2
            idx_s1 = self.status["activity"][:, s, idx] & (
                ~self.status["activity"][:, s + 1, idx]
            )
            idx_s2 = self.status["activity"][:, s + 1, idx] & (
                ~self.status["activity"][:, s, idx]
            )
            idx_s12 = self.status["activity"][:, s + 1, idx] & (
                self.status["activity"][:, s, idx]
            )

            n_s1 = self.matching["IDs"][idx_s1, s].astype("int")
            n_s2 = self.matching["IDs"][idx_s2, s + 1].astype("int")
            n_s12 = self.matching["IDs"][idx_s12, s].astype("int")

            A_tmp = sp.sparse.hstack(
                [
                    sp.sparse.csc_matrix(
                        cv2.remap(
                            img.reshape(self.params["dims"]),
                            x_remap,
                            y_remap,
                            cv2.INTER_CUBIC,
                        ).reshape(-1, 1)
                    )
                    for img in A2[:, n_s2].toarray().T
                ]
            )

            if mode == "act":
                for style, footprints in zip(
                    ["dashed", "solid", "dotted"], [A1[:, n_s1], A1[:, n_s12], A_tmp]
                ):
                    [
                        ax_ROI.contour(
                            (a / a.max()).reshape(512, 512).toarray(),
                            levels=[0.3],
                            colors="w",
                            linewidths=[0.3],
                            linestyles=[style],
                        )
                        for a in footprints.T
                    ]

            elif mode == "PC":
                # print(np.where(idx_s1)[0])
                # print(n_s1)
                for c, n in zip(np.where(idx_s1)[0], n_s1):
                    a = A1[:, n]
                    f = np.where(self.status["fields"][c, s, :])[0][0]
                    colVal = scalarMap.to_rgba(self.fields["location"][c, s, f, 0])
                    ax_ROI.contour(
                        (a / a.max()).reshape(512, 512).toarray(),
                        levels=[0.3],
                        colors=[colVal],
                        linewidths=[0.5],
                        linestyles=["dashed"],
                    )

                for i, (c, n) in enumerate(zip(np.where(idx_s2)[0], n_s2)):
                    a = A_tmp[:, i]
                    f = np.where(self.status["fields"][c, s + 1, :])[0][0]
                    colVal = scalarMap.to_rgba(self.fields["location"][c, s + 1, f, 0])
                    ax_ROI.contour(
                        (a / a.max()).reshape(512, 512).toarray(),
                        levels=[0.3],
                        colors=[colVal],
                        linewidths=[0.5],
                        linestyles=["dotted"],
                    )
                for c, n in zip(np.where(idx_s12)[0], n_s12):
                    a = A1[:, n]
                    f = np.where(self.status["fields"][c, s, :])[0][0]
                    colVal = scalarMap.to_rgba(self.fields["location"][c, s, f, 0])
                    ax_ROI.contour(
                        (a / a.max()).reshape(512, 512).toarray(),
                        levels=[0.3],
                        colors=[colVal],
                        linewidths=[0.5],
                        linestyles=["solid"],
                    )

            if mode == "PC":
                # cbaxes = plt.axes([0.285,0.75,0.01,0.225])
                cbaxes = plt.axes([0.04, 0.47, 0.15, 0.0125])
                cb = fig.colorbar(scalarMap, cax=cbaxes, orientation="horizontal")
                # cb.set_label('location')
                cb.set_label(
                    "location",
                    fontsize=8,
                    rotation="horizontal",
                    ha="left",
                    va="center",
                )  # ,labelpad=0,y=-0.5)
                # print(cb.ax.__dict__.keys())
                cbaxes.xaxis.set_label_coords(1.075, 0.4)

            ax_ROI.plot(
                np.NaN,
                np.NaN,
                "k-",
                label="$\\%s_{s_1}^+ \cap \\%s_{s_2}^+$" % (state_label, state_label),
            )
            ax_ROI.plot(np.NaN, np.NaN, "k--", label="$\\%s_{s_1}^+$" % state_label)
            ax_ROI.plot(np.NaN, np.NaN, "k:", label="$\\%s_{s_2}^+$" % state_label)
            ax_ROI.legend(
                fontsize=10,
                bbox_to_anchor=[1.2, 1.1],
                loc="upper right",
                handlelength=1,
            )

            sbar = ScaleBar(530.68 / 512 * 10 ** (-6), location="lower right")
            ax_ROI.add_artist(sbar)
            ax_ROI.set_xticks([])
            ax_ROI.set_yticks([])
        # plt.show()
        # return
        ### plot distribution of active sessions/neuron
        ax = plt.axes([0.45, 0.85, 0.175, 0.1])
        self.pl_dat.add_number(fig, ax, order=2, offset=[-225, 25])
        if mode == "act":
            ax.plot([0, self.data["nSes"]], [nC_good, nC_good], "k--", linewidth=0.5)
            ax.plot(
                np.where(self.status["sessions"])[0],
                self.status["activity"][:, self.status["sessions"], 1].sum(0),
                "k.",
                markersize=1,
            )
            ax.set_ylim([0, 1000])
            ax.set_ylabel("# neurons")
        else:
            # ax.plot([0,self.data['nSes']],[nC_good,nC_good],'k--',linewidth=0.5)
            ax.plot(
                np.where(self.status["sessions"])[0],
                self.status["activity"][:, self.status["sessions"], 2].sum(0),
                "k.",
                markersize=1,
            )
            ax.set_ylim([0, 750])
            ax.set_ylabel("# PC")
        ax.set_xlabel("session")
        self.pl_dat.remove_frame(ax, ["top", "right"])

        ax = plt.axes([0.45, 0.58, 0.175, 0.15])
        self.pl_dat.add_number(fig, ax, order=3, offset=[-200, 50])

        # plt.hist(self.status['activity'][...,1:3].sum(1),pl_dat.h_edges,color=[[0.6,0.6,0.6],'k'],width=0.4,label=['# sessions active','# sessions coding']);
        ax.axhline(self.status["sessions"].sum(), color="r", linestyle="--", zorder=0)
        ax.hist(
            status.sum(1), self.pl_dat.h_edges, color="k", width=1, label="emp. data"
        )
        # ax.hist(status_test.sum(1),pl_dat.h_edges,color='tab:red',alpha=0.7,width=0.8,label='rnd. data');
        if mode == "act":
            ax.hist(
                (status_act_test_rnd).sum(1),
                self.pl_dat.h_edges,
                color=[0.5, 0.5, 0.5],
                alpha=0.5,
                width=1,
            )
            res = sstats.ks_2samp(status_act.sum(1), status_act_test_rnd.sum(1))
            # print(res)
        elif mode == "PC":
            ax.hist(
                (status_PC_test_rnd).sum(1),
                self.pl_dat.h_edges,
                color=[1, 0.5, 0.5],
                alpha=0.5,
                width=1,
            )
            res = sstats.ks_2samp(status_PC.sum(1), status_PC_test_rnd.sum(1))
            # print(res)

        ax.set_xlabel("$N_{\\%s^+}$" % state_label)
        ax.set_ylabel("# neurons")
        # ax.legend(fontsize=10,loc='upper right')
        ax.set_xlim([-0.5, nSes_good + 0.5])
        if mode == "act":
            ax.set_ylim([0, 300])
        elif mode == "PC":
            ax.set_ylim([0, 500])

        ax.spines[["top", "right"]].set_visible(False)

        ### calculating ICPI
        status_alt = np.zeros_like(self.status["activity"][..., 1], "int")

        IPI_test = np.zeros(self.data["nSes"] + 1)
        # for c in range(self.data['nC']):
        for c in np.where(self.status["clusters"])[0]:
            s0 = 0
            inAct = False
            for s in np.where(self.status["sessions"])[0]:
                if inAct:
                    if ~self.status["activity"][c, s, 1]:
                        La = self.status["sessions"][s0:s].sum()
                        status_alt[c, s0:s] = La
                        IPI_test[La] += 1
                        # print(self.status["sessions"][s:s0].sum())
                        inAct = False
                else:
                    if self.status["activity"][c, s, 1]:
                        # print('getting active')
                        s0 = s
                        inAct = True
            if inAct:
                La = self.status["sessions"][s0 : s + 1].sum()
                status_alt[c, s0 : s + 1] = La
                IPI_test[La] += 1

        status_alt[:, ~self.status["sessions"]] = 0
        # print(IPI_test)
        ### obtain inter-coding intervals
        ICI = np.zeros((nSes_good, 2))  # inter-coding-interval
        IPI = np.zeros((nSes_good, 2))  # inter-pause-interval (= coding duration)

        # print(status.shape)
        # return

        t_start = time.time()
        ICI[:, 0] = get_ICPI(status, mode="ICI")
        # print("time taken: %.2f" % (time.time() - t_start))
        t_start = time.time()
        ICI[:, 1] = get_ICPI(status_test, mode="ICI")

        IPI[:, 0] = get_ICPI(~status, mode="IPI")
        IPI[:, 1] = get_ICPI(~status_test, mode="IPI")

        # print(IPI[:,0])

        IPI_bs = np.zeros((nSes_good, N_bs))
        IPI_bs_test = np.zeros((nSes_good, N_bs))
        ICI_bs = np.zeros((nSes_good, N_bs))
        ICI_bs_test = np.zeros((nSes_good, N_bs))
        for i in range(N_bs):
            IPI_bs[:, i] = get_ICPI(
                ~status[np.random.randint(0, nC_good, nC_good), :], mode="IPI"
            )
            IPI_bs_test[:, i] = get_ICPI(
                ~status_test[np.random.randint(0, nC_good, nC_good), :], mode="IPI"
            )

            ICI_bs[:, i] = get_ICPI(
                status[np.random.randint(0, nC_good, nC_good), :], mode="ICI"
            )
            ICI_bs_test[:, i] = get_ICPI(
                status_test[np.random.randint(0, nC_good, nC_good), :], mode="ICI"
            )

        ICI_summed = ICI * np.arange(nSes_good)[:, np.newaxis]
        IPI_summed = IPI * np.arange(nSes_good)[:, np.newaxis]

        ## end calculating ICPIs

        pval_IPI = np.zeros(nSes_good) * np.NaN
        pval_ICI = np.zeros(nSes_good) * np.NaN
        for s in range(nSes_good):
            # print('ttest (s=%d)'%(s+1))
            # print(np.nanmean(IPI_bs[s,:]),np.nanstd(IPI_bs[s,:]))
            # print(np.nanmean(IPI_bs_test[s,:]),np.nanstd(IPI_bs_test[s,:]))
            # res = sstats.ttest_ind(IPI_bs[s,:],IPI_bs_test[s,:])
            # print(res)
            res = sstats.ttest_ind_from_stats(
                np.nanmean(IPI_bs[s, :]),
                np.nanstd(IPI_bs[s, :]),
                N_bs,
                np.nanmean(IPI_bs_test[s, :]),
                np.nanstd(IPI_bs_test[s, :]),
                N_bs,
                equal_var=True,
            )
            pval_IPI[s] = res.pvalue

            res = sstats.ttest_ind_from_stats(
                np.nanmean(ICI_bs[s, :]),
                np.nanstd(ICI_bs[s, :]),
                N_bs,
                np.nanmean(ICI_bs_test[s, :]),
                np.nanstd(ICI_bs_test[s, :]),
                N_bs,
                equal_var=True,
            )
            pval_ICI[s] = res.pvalue

        print("time taken: %.2f" % (time.time() - t_start))
        IPI_bs[IPI_bs == 0] = np.NaN
        # print(IPI_bs)

        ICI[ICI == 0] = np.NaN
        IPI[IPI == 0] = np.NaN

        ax = plt.axes([0.75, 0.11, 0.225, 0.25])
        self.pl_dat.add_number(fig, ax, order=7)
        # ax.loglog(IPI[:,0],'k-',label='IPI')
        self.pl_dat.plot_with_confidence(
            ax,
            np.linspace(0, nSes_good, nSes_good),
            np.nanmean(IPI_bs, 1),
            np.nanstd(IPI_bs, 1),
            col="k",
            lw=0.5,
            label="$I_{\\%s^+}$" % state_label,
        )
        self.pl_dat.plot_with_confidence(
            ax,
            np.linspace(0, nSes_good, nSes_good),
            np.nanmean(IPI_bs_test, 1),
            np.nanstd(IPI_bs_test, 1),
            col="tab:red",
            lw=0.5,
        )
        self.pl_dat.plot_with_confidence(
            ax,
            np.linspace(0, nSes_good, nSes_good),
            np.nanmean(ICI_bs, 1),
            np.nanstd(ICI_bs, 1),
            col="k",
            ls=":",
            lw=0.5,
            label="$I_{\\%s^-}$" % state_label,
        )
        self.pl_dat.plot_with_confidence(
            ax,
            np.linspace(0, nSes_good, nSes_good),
            np.nanmean(ICI_bs_test, 1),
            np.nanstd(ICI_bs_test, 1),
            col="tab:red",
            ls=":",
            lw=0.5,
        )
        # ax.loglog(np.nanmean(IPI_bs,1)-np.nanstd(IPI_bs,1),'-',color=[0.1,0.5,0.5])
        # ax.loglog(np.nanmean(IPI_bs,1)+np.nanstd(IPI_bs,1),'-',color=[0.5,0.1,0.5])
        # ax.loglog(np.nanmean(IPI_bs,1),'k-',label='IPI')
        # ax.loglog(IPI.mean(0)+IPI.std(0),'-',color=[0.5,0.5,0.5])
        # ax.loglog(IPI[:,1],'tab:red')
        # ax.loglog(ICI[:,0],'k:',label='ICI')
        # ax.loglog(ICI[:,1],color='tab:red',linestyle=':')
        ax.set_xscale("log")
        ax.set_yscale("log")

        plt.setp(
            ax,
            xlim=[0.9, np.maximum(105, nSes_max)],
            ylim=[1, 10**5],
            ylabel="# occurence",
            xlabel="$\mathcal{L}_{\\%s^+}$ / $\mathcal{L}_{\\%s^-}$ [sessions]"
            % (state_label, state_label),
        )
        ax.legend(fontsize=10, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)

        # print(np.log10(pval))
        # ax = plt.axes([0.21,0.325,0.1,0.075])
        # ax.plot(np.log10(pval_IPI),'k',linewidth=0.5)
        # ax.plot(np.log10(pval_ICI),'k:',linewidth=0.5)
        # ax.plot([0,nSes_good],[-10,-10],'k--',linewidth=0.3)
        # ax.set_xscale('log')
        # ax.set_xlim([0.9,np.maximum(105,nSes_max)])
        # ax.set_xticks(np.logspace(0,2,3))
        # # ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        # ax.set_ylim([-300,0])
        # ax.set_ylabel('$\log_{10}(p_{val})$',fontsize=7,rotation='horizontal',labelpad=-15,y=1.15)#,ha='center',va='center')
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        # ax = plt.axes([0.45,0.325,0.15,0.15])
        # pl_dat.add_number(fig,ax,order=5)
        # ax.plot(IPI_summed[:,0]/np.nansum(IPI_summed[:,0]),'k-',label='$I_{\\%s^+}$'%(state_label))
        # ax.plot(IPI_summed[:,1]/np.nansum(IPI_summed[:,1]),'-',color='tab:red')
        # ax.plot(ICI_summed[:,0]/np.nansum(ICI_summed[:,0]),'k:',label='$I_{\\%s^-}$'%(state_label))
        # ax.plot(ICI_summed[:,1]/np.nansum(ICI_summed[:,1]),':',color='tab:red')
        # ax.set_xscale('log')
        # ax.set_xticklabels([])
        # ax.set_ylabel('$p_{\in \mathcal{L}_{\\%s^+} / \mathcal{L}_{\\%s^-}}$'%(state_label,state_label))
        # # ax.legend(fontsize=10)
        # ax.set_xlim([0.8,np.maximum(105,nSes_max)])
        # ax.set_xticks(np.logspace(0,2,3))
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.plot(IPI_summed[:,1]/np.nansum(IPI_summed[:,1]),'-',color='tab:red')
        # ax.plot(ICI_summed[:,1]/np.nansum(ICI_summed[:,1]),':',color='tab:red')
        # ax.set_yscale('log')

        ax = plt.axes([0.875, 0.35, 0.1, 0.1])
        # ax.plot(IPI*np.arange(self.data['nSes'])/self.status['activity'][...,1].sum())

        ax.plot(
            range(1, nSes_good),
            np.nancumsum(IPI_summed[1:, 0] / np.nansum(IPI_summed[1:, 0])),
            "k-",
        )
        ax.plot(
            range(1, nSes_good),
            np.nancumsum(IPI_summed[1:, 1] / np.nansum(IPI_summed[1:, 1])),
            "-",
            color="tab:red",
        )
        ax.plot(
            range(1, nSes_good),
            np.nancumsum(ICI_summed[1:, 0] / np.nansum(ICI_summed[1:, 0])),
            "k:",
        )
        ax.plot(
            range(1, nSes_good),
            np.nancumsum(ICI_summed[1:, 1] / np.nansum(ICI_summed[1:, 1])),
            ":",
            color="tab:red",
        )

        # ax.legend(fontsize=10)
        ax.set_xscale("log")
        ax.set_xlim([0.8, np.maximum(105, nSes_max)])
        ax.set_xticks(np.logspace(0, 2, 3))
        # ax.xaxis.set_minor_locator(AutoMinorLocator(9))
        # print(LogLocator().tick_values(1,100))
        # ax.set_xlabel('$\mathcal{L}_{\\%s^+}$ / $\mathcal{L}_{\\%s^-}$'%(state_label,state_label))
        ax.set_ylabel(
            "$cdf_{\in \mathcal{L}_{\\%s^+} / \mathcal{L}_{\\%s^-}}$"
            % (state_label, state_label),
            fontsize=8,
            rotation="horizontal",
            labelpad=-15,
            y=1.15,
        )  # ,ha='center',va='center')
        # ax.set_xlabel('cont.coding [ses.]')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax = plt.subplot(4,4,4)
        # ax.plot(status.sum(1),IPI_stats[:,2],'k.',markersize=0.5)

        status_act = self.status["activity"][self.status["clusters"], :, 1]
        status_PC = self.status["activity"][self.status["clusters"], :, 2]
        # status_act = self.status['activity'][...,1]
        # status_PC = self.status['activity'][...,2]
        status = status_act if mode == "act" else status_PC
        status_dep = None if mode == "act" else status_act

        status[:, ~self.status["sessions"]] = False
        ds = 1
        dp_pos, p_pos = get_dp(
            status,
            status_dep=status_dep,
            status_session=self.status["sessions"],
            ds=ds,
            mode=mode,
        )
        dp_neg, p_neg = get_dp(
            ~status,
            status_dep=status_dep,
            status_session=self.status["sessions"],
            ds=ds,
            mode=mode,
        )

        status_dep = None if mode == "act" else status_act_test
        dp_pos_test, p_pos_test = get_dp(
            status_test, status_dep=status_dep, ds=ds, mode=mode
        )
        dp_neg_test, p_neg_test = get_dp(
            ~status_test, status_dep=status_dep, ds=ds, mode=mode
        )
        # dp_pos,p_pos = get_dp(status,status_act,status_dep=status_dep,status_session=self.status["sessions"],ds=ds,mode=mode)
        # dp_neg,p_neg = get_dp(~status,status_act,status_dep=status_dep,status_session=self.status["sessions"],ds=ds,mode=mode)
        #
        # status_dep = None if mode=='act' else status_act_test
        # dp_pos_test,p_pos_test = get_dp(status_test,status_act_test,status_dep=status_dep,ds=ds,mode=mode)
        # dp_neg_test,p_neg_test = get_dp(~status_test,status_act_test,status_dep=status_dep,ds=ds,mode=mode)

        ax = plt.axes([0.1, 0.11, 0.16, 0.225])
        self.pl_dat.add_number(fig, ax, order=5)
        ax.plot(
            status.sum(1) + 0.7 * np.random.rand(nC_good),
            p_pos + 0.02 * np.random.rand(nC_good),
            "k.",
            markersize=1.5,
            markeredgewidth=0,
            alpha=0.6,
            label="$\\%s^+_s$" % (state_label),
        )
        ax.plot(
            status_test.sum(1) + 0.7 * np.random.rand(nC_good),
            p_pos_test + 0.02 * np.random.rand(nC_good),
            ".",
            color="tab:red",
            markersize=1.5,
            markeredgewidth=0,
            zorder=1,
        )
        ax.set_yticks(np.linspace(0, 1, 3))
        ax.set_xlabel("$N_{\\%s^+}$" % (state_label))
        ax.set_ylabel("$p(\\%s^+_{s+1} | \\%s^+_s)$" % (state_label, state_label))
        self.pl_dat.remove_frame(ax, ["top", "right"])

        res = sstats.ks_2samp(dp_pos, dp_pos_test)
        # print("IPI")
        # print(res)
        # print(np.nanmean(dp_pos),np.nanstd(dp_pos))
        # print(np.nanpercentile(dp_pos,[2.5,97.5]))
        # print(np.nanmean(dp_pos_test),np.nanstd(dp_pos_test))

        res = sstats.kruskal(dp_pos, dp_pos_test, nan_policy="omit")
        # print(res)

        res = sstats.ks_2samp(dp_neg, dp_neg_test)
        # print("IAI")
        # print(res)
        # print(np.nanmean(dp_neg),np.nanstd(dp_neg))
        # print(np.nanmean(dp_neg_test),np.nanstd(dp_neg_test))

        width = 0.75
        ax = plt.axes([0.41, 0.3, 0.075, 0.125])
        self.pl_dat.add_number(fig, ax, order=6)
        ax.plot([-0.5, 1.5], [0, 0], "--", color=[0.6, 0.6, 0.6], linewidth=0.5)
        bp = ax.boxplot(
            dp_pos[np.isfinite(dp_pos)],
            positions=[0],
            widths=width,
            whis=[5, 95],
            notch=True,
            bootstrap=100,
            showfliers=False,
        )  # ,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        bp_test = ax.boxplot(
            dp_pos_test[np.isfinite(dp_pos_test)],
            positions=[1],
            widths=width,
            whis=[5, 95],
            notch=True,
            bootstrap=100,
            showfliers=False,
        )  # ,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        for element in ["boxes", "whiskers", "means", "medians", "caps"]:
            plt.setp(bp[element], color="k")
            plt.setp(bp_test[element], color="tab:red")
        # ax.bar(1,np.nanmean(dp_pos_test),facecolor='tab:red')
        # ax.errorbar(1,np.nanmean(dp_pos_test),np.abs(np.nanmean(dp_pos_test)-np.nanpercentile(dp_pos_test,[2.5,97.5]))[:,np.newaxis],ecolor='r')
        self.pl_dat.remove_frame(ax, ["top", "right", "bottom"])
        ax.set_xticks([])
        ax.set_ylabel("$\left\langle \Delta p_{\\%s} \\right \\rangle$" % state_label)
        ax.set_ylim([-0.25, 0.75])

        ax2 = plt.axes([0.41, 0.11, 0.075, 0.175])
        ax2.hist(
            dp_pos,
            np.linspace(-1, 1, 101),
            facecolor="k",
            alpha=0.5,
            orientation="horizontal",
            zorder=0,
        )
        ax2.hist(
            dp_pos_test,
            np.linspace(-1, 1, 101),
            facecolor="tab:red",
            alpha=0.5,
            orientation="horizontal",
            zorder=0,
        )
        # ax2.hist(dp_pos,np.linspace(0,2,101),facecolor='k',alpha=0.5,orientation='horizontal',zorder=0)
        # ax2.hist(dp_pos_test,np.linspace(0,2,101),facecolor='tab:red',alpha=0.5,orientation='horizontal',zorder=0)
        ax2.set_xticks(
            []
        )  # status_dilate = sp.ndimage.morphology.binary_dilation(status,np.ones((1,3),'bool'))
        ax2.set_xlim([0, ax2.get_xlim()[1] * 2])
        ax2.set_ylim([-0.5, 1])
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        ax = ax2.twiny()
        ax.plot(
            status.sum(1) + 0.7 * np.random.rand(nC_good),
            dp_pos + 0.02 * np.random.rand(nC_good),
            "k.",
            markersize=1.5,
            markeredgewidth=0,
            alpha=0.6,
            label="$\\%s^+_s$" % (state_label),
        )
        ax.plot(
            status_test.sum(1) + 0.7 * np.random.rand(nC_good),
            dp_pos_test + 0.02 * np.random.rand(nC_good),
            ".",
            color="tab:red",
            markersize=1.5,
            markeredgewidth=0,
            zorder=1,
        )

        ax.set_yticks(np.linspace(-1, 1, 5))
        ax.set_xlim([-20, nSes_good])
        ax.set_ylim([-0.5, 1])
        # ax.set_ylim([0,4])

        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position("bottom")
        ax.set_xlabel("$N_{\\%s^+}$" % (state_label), x=1, y=-0.1)
        ax2.set_ylabel(
            "$\Delta p (\\%s^+_{s+1} | \\%s^+_s)$" % (state_label, state_label)
        )  #'$p_{\\alpha}^{\pm1}$')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.legend(fontsize=10,loc='upper left')

        ax = plt.axes([0.5, 0.3, 0.075, 0.125])
        ax.plot([-0.5, 1.5], [0, 0], "--", color=[0.6, 0.6, 0.6], linewidth=0.5)
        bp = ax.boxplot(
            dp_neg[np.isfinite(dp_neg)],
            positions=[0],
            widths=width,
            whis=[5, 95],
            notch=True,
            bootstrap=100,
            showfliers=False,
        )  # ,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        bp_test = ax.boxplot(
            dp_neg_test[np.isfinite(dp_neg_test)],
            positions=[1],
            widths=width,
            whis=[5, 95],
            notch=True,
            bootstrap=100,
            showfliers=False,
        )  # ,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))
        for element in ["boxes", "whiskers", "means", "medians", "caps"]:
            plt.setp(bp[element], color="k")
            plt.setp(bp_test[element], color="tab:red")

        self.pl_dat.remove_frame(ax, ["top", "right", "bottom"])
        ax.set_xticks([])
        ax.set_ylim([-0.25, 0.75])
        ax.set_yticklabels([])

        ax2 = plt.axes([0.5, 0.11, 0.075, 0.175])
        ax2.invert_xaxis()
        ax2.hist(
            dp_neg,
            np.linspace(-1, 1, 101),
            facecolor="k",
            alpha=0.5,
            orientation="horizontal",
            zorder=0,
        )
        ax2.hist(
            dp_neg_test,
            np.linspace(-1, 1, 101),
            facecolor="tab:red",
            alpha=0.5,
            orientation="horizontal",
            zorder=0,
        )
        # ax2.hist(dp,np.linspace(0,2,101),facecolor='k',alpha=0.5,orientation='horizontal',zorder=0)
        # ax2.hist(dp_test,np.linspace(0,2,101),facecolor='tab:red',alpha=0.5,orientation='horizontal',zorder=0)
        ax2.set_xticks([])
        ax2.set_xlim([ax2.get_xlim()[0] * 2, 0])
        ax2.set_ylim([-0.5, 1])
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        ax = ax2.twiny()
        ax.plot(
            status.sum(1) + 0.7 * np.random.rand(nC_good),
            dp_neg + 0.02 * np.random.rand(nC_good),
            "k.",
            markersize=1.5,
            markeredgewidth=0,
            alpha=0.6,
            label="$\\beta_s$",
        )
        ax.plot(
            status_test.sum(1) + 0.7 * np.random.rand(nC_good),
            dp_neg_test + 0.02 * np.random.rand(nC_good),
            ".",
            color="tab:red",
            markersize=1.5,
            markeredgewidth=0,
            zorder=1,
        )
        # ax.set_ylim([0,1])
        ax.set_xlim([0, nSes_good + 20])
        ax.set_yticks([])
        # ax.set_yticks(np.linspace(0,1,3))
        ax.set_ylim([-0.5, 1])
        # ax.set_ylim([0,4])
        ax.xaxis.tick_bottom()
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel(
            "$\Delta p (\\%s^-_{s+1} | \\%s^-_s)$" % (state_label, state_label)
        )
        # ax.set_xlabel('\t # sessions')
        # ax.set_ylabel('$p (\\alpha_{s\pm1} | \\alpha_s)$')#'$p_{\\alpha}^{\pm1}$')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ax.legend(fontsize=10,loc='lower left')

        # ax = plt.axes([0.75,0.625,0.125,0.325])
        # status_dilate = sp.ndimage.morphology.binary_dilation(~status,np.ones((1,3),'bool'))
        # cont_score = 1-(status_dilate&(~status)).sum(1)/(2*status.sum(1))
        # ax = plt.axes([0.75,0.625,0.225,0.325])
        # ax.plot(status.sum(1)+0.7*np.random.rand(nC_good),cont_score+0.02*np.random.rand(nC_good),'k.',markersize=1.5,markeredgewidth=0)
        #
        # status_dilate_test = sp.ndimage.morphology.binary_dilation(status_test,np.ones((1,3),'bool'))
        # cont_score_test = 1-(status_dilate_test&(~status_test)).sum(1)/(2*status_test.sum(1))
        # ax.plot(status_test.sum(1)+0.7*np.random.rand(nC_good),cont_score_test+0.02*np.random.rand(nC_good),'.',color='tab:red',markersize=1.5,markeredgewidth=0)
        # ax.set_ylim([0,1])
        # ax.set_xlim([0,nSes_max])
        # ax.set_xlabel('# sessions active')
        # ax.set_ylabel('$p (\\alpha_{s\pm1} | \\alpha_s)$')#'$p_{\\alpha}^{\pm1}$')
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        # ax.plot(ICI_stats[:,0]+np.random.rand(nC)*0.5-0.25,IPI_stats[:,0]+np.random.rand(nC)*0.5-0.25,'k.',markeredgecolor='None',markersize=1)
        # ax.hist(ICI_stats[:,0],np.linspace(0,nSes,nSes+1))
        # ax.hist(IPI_stats[:,0],np.linspace(0,nSes,nSes+1))
        # ax = plt.subplot(8,4,8)
        # ax.plot(ICI_stats[:,1]+np.random.rand(nC)*0.5-0.25,IPI_stats[:,1]+np.random.rand(nC)*0.5-0.25,'k.',markeredgecolor='None',markersize=1)
        # ax.hist(ICI_stats[:,1],np.linspace(0,nSes,nSes+1))
        # ax.hist(IPI_stats[:,1],np.linspace(0,nSes,nSes+1))
        # ax = plt.subplot(8,4,12)
        # ax.plot(ICI_stats[:,2]+np.random.rand(nC)*0.5-0.25,IPI_stats[:,2]+np.random.rand(nC)*0.5-0.25,'k.',markeredgecolor='None',markersize=1)
        # ax.hist(ICI_stats[:,2],np.linspace(0,nSes,nSes+1))
        # ax.hist(IPI_stats[:,2],np.linspace(0,nSes,nSes+1))
        # ax = plt.subplot(8,4,16)
        # ax.plot(ICI_stats[:,3]+np.random.rand(nC)*0.5-0.25,IPI_stats[:,3]+np.random.rand(nC)*0.5-0.25,'k.',markeredgecolor='None',markersize=1)
        # ax.hist(ICI_stats[:,3],np.linspace(0,nSes,nSes+1))
        # ax.hist(IPI_stats[:,3],np.linspace(0,nSes,nSes+1))

        # print(ICI_stats)
        # print(IPI_stats)
        # print(IPI*np.arange(self.data['nSes']))
        # print((IPI*np.arange(self.data['nSes'])).sum())
        status_act = status_act[:, self.status["sessions"]]
        status_PC = status_PC[:, self.status["sessions"]]
        status = status[:, self.status["sessions"]]
        recurr = np.zeros((nSes_good, nSes_good)) * np.NaN
        N_active = status_act.sum(0)
        # session_bool = np.pad(self.status["sessions"][1:],(0,1),constant_values=False) & np.pad(self.status["sessions"][:],(0,0),constant_values=False)

        for s in range(nSes_good):  # np.where(self.status["sessions"])[0]:
            overlap = status[status[:, s], :].sum(0).astype("float")
            N_ref = N_active if mode == "act" else status_act[status_PC[:, s], :].sum(0)
            recurr[s, 1 : nSes_good - s] = (overlap / N_ref)[s + 1 :]

        recurr_test = np.zeros((nSes_good, nSes_good)) * np.NaN
        N_active_test = status_test.sum(0)
        tmp = []
        for s in range(nSes_good):
            # overlap_act_test = status_test[status_test[:,s],:].sum(0).astype('float')
            overlap_test = status_test[status_test[:, s], :].sum(0).astype("float")
            N_ref = (
                N_active_test
                if mode == "act"
                else status_act_test[status_PC_test[:, s], :].sum(0)
            )
            recurr_test[s, 1 : nSes_good - s] = (overlap_test / N_ref)[s + 1 :]
            if (~np.isnan(recurr_test[s, :])).sum() > 1:
                tmp.append(recurr_test[s, ~np.isnan(recurr_test[s, :])])

        # print(tmp)
        # res = sstats.f_oneway(*tmp)
        # print(res)
        # ax = plt.subplot(2,4,8)
        rec_mean = np.nanmean(np.nanmean(recurr, 0))
        rec_var = np.sqrt(np.nansum(np.nanvar(recurr, 0)) / (recurr.shape[1] - 1))

        # print(rec_mean)
        # print(rec_var)

        if mode == "act":
            ax_sketch = plt.axes([0.675, 0.875, 0.15, 0.1])
            ax_sketch2 = plt.axes([0.85, 0.875, 0.15, 0.1])
            # pl_dat.add_number(fig,ax_sketch,order=1,offset=[-40,10])
            pic_path = "/home/wollex/Data/Science/PhD/Thesis/pics/sketches/ds1.png"
            if os.path.exists(pic_path):
                ax_sketch.axis("off")
                im = mpimg.imread(pic_path)
                ax_sketch.imshow(im)
                ax_sketch.set_xlim([0, im.shape[1]])

            pic2_path = "/home/wollex/Data/Science/PhD/Thesis/pics/sketches/ds3.png"
            if os.path.exists(pic2_path):
                ax_sketch2.axis("off")
                im2 = mpimg.imread(pic2_path)
                ax_sketch2.imshow(im2)
                ax_sketch2.set_xlim([0, im2.shape[1]])

            ax = plt.axes([0.775, 0.65, 0.2, 0.155])
            self.pl_dat.add_number(fig, ax, order=4, offset=[-250, 250])
        else:
            ax = plt.axes([0.55, 0.65, 0.1, 0.05])

            nAct = self.status["activity"][..., 1].sum(1)
            nPC = self.status["activity"][..., 2].sum(1)
            rate = nPC / nAct
            mean_r = np.zeros((self.data["nSes"], 3)) * np.NaN
            tmp = []
            print("get CI from bootstrapping")
            for i in range(1, self.data["nSes"]):
                if np.any(nAct == i):
                    mean_r[i, 0] = rate[nAct == i].mean()
                    mean_r[i, 1:] = np.percentile(rate[nAct == i], [15.8, 84.1])

            count = np.zeros(self.data["nSes"] + 1)
            for item in Counter(status_alt[self.status["activity"][..., 2]]).items():
                count[item[0]] = item[1]

            La_sessions = IPI_test * np.arange(len(IPI_test))
            pb = np.nanmean(
                self.status["activity"][self.status["clusters"], :, 2].sum(0)
                / self.status["activity"][self.status["clusters"], :, 1].sum(0)
            )
            ax.plot([0, 80], [pb, pb], "k--")
            ax.plot(
                gauss_smooth(count[: len(IPI_test)] / La_sessions, 1),
                label="$p(\\beta^+| \in \mathcal{L}_{\\alpha})$",
            )
            self.pl_dat.plot_with_confidence(
                ax,
                range(self.data["nSes"]),
                mean_r[:, 0],
                mean_r[:, 1:].T,
                col="r",
                label="$p(\\beta^+| \in N_{\\alpha})$",
            )
            ax.set_xlim([0, nSes_good])
            ax.set_ylim([0, 0.8])
            ax.set_ylabel("p", fontsize=8)
            ax.set_xlabel("$N_{\\alpha} / \mathcal{L}_{\\alpha}$", fontsize=8)
            ax.xaxis.set_label_coords(0.3, -0.6)
            ax.legend(
                fontsize=6,
                loc="lower right",
                bbox_to_anchor=[1.35, 0.9],
                handlelength=1,
            )
            self.pl_dat.remove_frame(ax, ["top", "right"])

            ax = plt.axes([0.775, 0.65, 0.2, 0.275])
            self.pl_dat.add_number(fig, ax, order=4, offset=[-150, 50])

        p = status.sum() / (nSes_good * nC_good)

        ax.axvline(p, color="k", linestyle="--")
        ax.text(10, p + 0.05, "$p^{(0)}_{\\%s^+}$" % (state_label), fontsize=8)
        SD = 1
        # ax.plot([1,nSes_good],[rec_mean,rec_mean],'k--',linewidth=0.5)

        self.pl_dat.plot_with_confidence(
            ax,
            np.linspace(1, nSes_good, nSes_good),
            np.nanmean(recurr, 0),
            SD * np.nanstd(recurr, 0),
            col="k",
            ls="-",
            label="emp. data",
        )
        self.pl_dat.plot_with_confidence(
            ax,
            np.linspace(1, nSes_good, nSes_good),
            np.nanmean(recurr_test, 0),
            SD * np.nanstd(recurr_test, 0),
            col="tab:red",
            ls="-",
            label="rnd. data",
        )
        ax.set_ylim([0, 1])
        ax.set_xlabel("$\Delta$ sessions")
        ax.set_ylabel(
            "$p(\\%s^+_{s+\Delta s} | \\%s^+_s)$" % (state_label, state_label)
        )  #'p(recurr.)')
        ax.set_xlim([0, nSes_good])
        if mode == "act":
            ax.legend(
                fontsize=8,
                loc="upper right",
                bbox_to_anchor=[1.05, 1.3],
                handlelength=1,
            )
        else:
            ax.legend(
                fontsize=8,
                loc="upper right",
                bbox_to_anchor=[1.05, 0.9],
                handlelength=1,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            if mode == "act":
                self.pl_dat.save_fig("act_dynamics")
            elif mode == "PC":
                self.pl_dat.save_fig("PC_dynamics")

        steps = min(nSes_good, 40)
        dp_pos = np.zeros((steps, 2)) * np.NaN
        dp_neg = np.zeros((steps, 2)) * np.NaN
        dp_pos_test = np.zeros((steps, 2)) * np.NaN
        dp_neg_test = np.zeros((steps, 2)) * np.NaN
        # print(get_dp(status,status_act,status_dep=status_dep,ds=ds,mode=mode))
        for ds in range(1, steps):
            status_dep = None if mode == "act" else status_act_test

            dp, _ = get_dp(status, status_dep=status_dep, ds=ds, mode=mode)
            dp_pos[ds, :] = [np.nanmean(dp), np.nanstd(dp)]
            dp_test, _ = get_dp(status_test, status_dep=status_dep, ds=ds, mode=mode)
            dp_pos_test[ds, :] = [np.nanmean(dp_test), np.nanstd(dp_test)]
            # res = sstats.ttest_ind_from_stats(dp_pos[ds,0],dp_pos[ds,1],nC,dp_pos_test[ds,0],dp_pos_test[ds,1],nC,equal_var=True)

            dp = get_dp(~status, status_dep=status_dep, ds=ds, mode=mode)
            dp_neg[ds, :] = [np.nanmean(dp), np.nanstd(dp)]
            dp_test = get_dp(~status_test, status_dep=status_dep, ds=ds, mode=mode)
            dp_neg_test[ds, :] = [np.nanmean(dp_test), np.nanstd(dp_test)]

        plt.figure()
        ax = plt.subplot(211)
        self.pl_dat.plot_with_confidence(
            ax, range(steps), dp_pos[:, 0], dp_pos[:, 1], col="k", ls="-"
        )
        self.pl_dat.plot_with_confidence(
            ax, range(steps), dp_pos_test[:, 0], dp_pos_test[:, 1], col="r", ls="-"
        )
        # plt.plot(dp_pos,'k')
        # plt.plot(dp_pos_test,'r')

        ax = plt.subplot(212)
        self.pl_dat.plot_with_confidence(
            ax, range(steps), dp_neg[:, 0], dp_neg[:, 1], col="k", ls="--"
        )
        self.pl_dat.plot_with_confidence(
            ax, range(steps), dp_neg_test[:, 0], dp_neg_test[:, 1], col="r", ls="--"
        )
        # plt.plot(dp_neg,'k--')
        # plt.plot(dp_neg_test,'r--')
        plt.show(block=False)

        # plt.figure()
        # plt.subplot(121)
        # plt.plot(dp_pos,dp,'k.')
        # plt.subplot(122)
        # plt.plot(dp_pos_test,dp_test,'r.')
        # plt.show(block=False)
        # plt.figure()
        # plt.scatter(numbers[:,0]+0.5*np.random.rand(nC),numbers[:,1]+0.5*np.random.rand(nC),s=10,marker='.')
        # plt.show(block=False)

        # plt.figure()
        # plt.hist(dp_pos,np.linspace(-1,1,201),color='k',histtype='step',cumulative=True,density=True)
        # plt.hist(dp_pos_test,np.linspace(-1,1,201),color='r',histtype='step',cumulative=True,density=True)
        # plt.show(block=False)
        # plt.figure()
        # plt.hist(dp_neg,np.linspace(-1,1,201),color='k',histtype='step',cumulative=True,density=True)
        # plt.hist(dp_neg_test,np.linspace(-1,1,201),color='r',histtype='step',cumulative=True,density=True)
        # plt.show(block=False)

    def plot_matching_results(self, sv=False):

        plt.figure(figsize=(4, 2))
        ax1 = plt.subplot(111)
        # plt.figure(figsize=(4,3))
        # ax1 = plt.axes([0.15, 0.5, 0.8, 0.45])
        # ax2 = plt.axes([0.15, 0.2, 0.8, 0.25])

        # active_time = np.zeros(self.data['nSes'])
        # for s in range(self.data['nSes']):
        # if self.status['sessions'][s]:
        # pathSession = pathcat([self.params['pathMouse'],'Session%02d'%(s+1)]);

        # for file in os.listdir(pathSession):
        # if file.endswith("aligned.mat"):
        # pathBH = os.path.join(pathSession, file)

        # f = h5py.File(pathBH,'r')
        # key_array = ['longrunperiod']

        # dataBH = {}
        # for key in key_array:
        # dataBH[key] = np.squeeze(f.get('alignedData/resampled/%s'%key).value)
        # f.close()

        # active_time[s] = dataBH['longrunperiod'].sum()/len(dataBH['longrunperiod']);

        # ax2.plot(t_ses[self.status['sessions']],active_time[self.status['sessions']],color='k')
        ##ax2.plot(t_measures(1:s_end),active_time,'k')
        # ax2.set_xlim([0,t_ses[-1]])
        # ax2.set_ylim([0,1])
        # ax2.set_xlabel('t [h]',fontsize=14)
        # ax2.set_ylabel('active time',fontsize=14)

        # ax1.plot(t_ses[self.status['sessions']],np.ones(self.status['sessions'].sum())*nC,color='k',linestyle=':',label='# neurons')
        t_ses = np.arange(self.data["nSes"])
        ax1.axhline(
            self.status["clusters"].sum(), color="k", linestyle=":", linewidth=0.5
        )
        ax1.scatter(
            t_ses[self.status["sessions"]],
            self.status["activity"][:, self.status["sessions"], 1].sum(0),
            s=20,
            color="k",
            marker="o",
            facecolor="none",
            label="# active neurons",
        )
        ax1.set_ylim([0, self.data["nC"] * 1.3])
        # ax1.set_xlim([0, t_ses[-1]])

        ax1.scatter(
            t_ses[self.status["sessions"]],
            self.status["activity"][:, self.status["sessions"], 2].sum(0),
            s=20,
            color="k",
            marker="o",
            facecolors="k",
            label="# place cells",
        )

        ax2 = ax1.twinx()
        ax2.plot(
            t_ses[self.status["sessions"]],
            self.status["activity"][:, self.status["sessions"], 2].sum(0)
            / self.status["activity"][:, self.status["sessions"], 1].sum(0),
            "r",
        )
        ax2.set_ylim([0, 0.7])
        ax2.yaxis.label.set_color("red")
        ax2.tick_params(axis="y", colors="red")
        ax2.set_ylabel("fraction PCs")

        # ax1.set_xlim([-1, t_ses[-1]])
        ax1.set_xlabel("session s", fontsize=14)
        ax1.legend(loc="upper left")
        plt.tight_layout()
        plt.show(block=False)

        # print(self.status['activity'][:,self.status['sessions'],2].sum(0)/self.status['activity'][:,self.status['sessions'],1].sum(0))
        if sv:
            self.pl_dat.save_fig("neuron_numbers")

    def plot_recurrence(self, sv=False, n_processes=4):

        t_ses = np.arange(self.data["nSes"])

        f, axs = plt.subplots(2, 2, figsize=(10, 4))

        for axx in [axs[1][0], axs[1][1]]:
            axx.axhline(0, color=[0.8, 0.8, 0.8], linestyle="--")

        for s in range(self.data["nSes"]):
            for i, population in enumerate(["active", "coding"]):
                for key, col in zip(
                    ["all", "continuous"], [[0.8, 0.8, 0.8], [0.6, 1.0, 0.6]]
                ):
                    axs[0][i].scatter(
                        self.pl_dat.n_edges,
                        self.recurrence[population][key][s, :],
                        5,
                        color=col,
                        marker="o",
                        label=key if s == 0 else None,
                    )

                axs[1][i].scatter(
                    self.pl_dat.n_edges,
                    self.recurrence[population]["overrepresentation"][s, :],
                    5,
                    color=[0.8, 0.8, 0.8],
                    marker="o",
                )

        axs[0][0].plot(
            self.pl_dat.n_edges,
            np.nanmean(self.recurrence["active"]["all"], 0),
            color="k",
        )
        axs[0][0].legend(loc="lower right", fontsize=12)

        axs[1][0].plot(
            self.pl_dat.n_edges,
            np.nanmean(self.recurrence["active"]["overrepresentation"], 0),
            color="k",
        )
        axs[0][1].plot(
            self.pl_dat.n_edges,
            np.nanmean(self.recurrence["coding"]["all"], 0),
            color="k",
        )
        axs[1][1].plot(
            self.pl_dat.n_edges,
            np.nanmean(self.recurrence["coding"]["overrepresentation"], 0),
            color="k",
        )

        plt.setp(
            axs[0][0],
            xlim=[0, t_ses[-1]],
            ylim=[0, 1],
            xticks=[],
            yticks=np.linspace(0, 1, 3),
            ylabel="fraction",
            title="active cells",
        )
        plt.setp(
            axs[0][1],
            xlim=[0, t_ses[-1]],
            ylim=[0, 1],
            xticks=[],
            yticks=np.linspace(0, 1, 3),
            title="place cells",
        )
        plt.setp(
            axs[1][0],
            xlim=[0, t_ses[-1]],
            ylim=[-10, 30],
            xlabel="session diff. $\Delta$ s",
            ylabel="overrepr.",
        )
        plt.setp(
            axs[1][1],
            xlim=[0, t_ses[-1]],
            ylim=[-10, 30],
            xlabel="session diff. $\Delta$ s",
        )

        plt.tight_layout()
        plt.show(block=False)
        if sv:
            self.pl_dat.save_fig("ROI_stability")

        plt.figure(figsize=(5, 2.5))
        ax = plt.subplot(111)
        # ax.scatter(pl_dat.n_edges,recurrence['active']['all'][0,:],5,color=[0.8,0.8,0.8],marker='o',label='any')
        # ax.scatter(pl_dat.n_edges,recurrence['active']['continuous'][0,:],5,color=[0.6,1,0.6],marker='o',label='continuous')
        self.pl_dat.plot_with_confidence(
            ax,
            self.pl_dat.n_edges - 1,
            np.nanmean(self.recurrence["active"]["all"], 0),
            1.96 * np.nanstd(self.recurrence["active"]["all"], 0),
            col="k",
            ls="-",
            label="recurrence of active cells",
        )
        # for s in range(self.data['nSes']):
        # ax.scatter(pl_dat.n_edges,recurrence['active']['all'][s,:],5,color=[0.8,0.8,0.8],marker='o')
        # ax.scatter(pl_dat.n_edges,recurrence['active']['continuous'][s,:],5,color=[0.6,1,0.6],marker='o')

        # axs[0][1].scatter(pl_dat.n_edges,recurrence['coding']['all'][s,:],5,color=[0.8,0.8,0.8],marker='o')
        # axs[0][1].scatter(pl_dat.n_edges,recurrence['coding']['continuous'][s,:],5,color=[0.6,1,0.6],marker='o')
        # ax.plot(pl_dat.n_edges-1,np.nanmean(recurrence['active']['all'],0),color='k')
        ax.legend(loc="upper right", fontsize=10)
        ax.set_xlim([0, t_ses[-1]])
        ax.set_ylim([0, 1])
        ax.set_ylabel("fraction", fontsize=14)
        ax.set_xlabel("session diff. $\Delta$ s", fontsize=14)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.show(block=False)
        if sv:
            self.pl_dat.save_fig("ROI_stability_activity")

        plt.figure(figsize=(5, 2.5))
        ax = plt.subplot(111)
        # ax.scatter(pl_dat.n_edges,recurrence['coding']['all'][0,:],5,color=[0.8,0.8,0.8],marker='o',label='any')
        # ax.scatter(pl_dat.n_edges,recurrence['coding']['continuous'][0,:],5,color=[0.6,1,0.6],marker='o',label='continuous')
        self.pl_dat.plot_with_confidence(
            ax,
            self.pl_dat.n_edges - 1,
            np.nanmean(self.recurrence["coding"]["ofactive"], 0),
            1.0 * np.nanstd(self.recurrence["coding"]["ofactive"], 0),
            col="k",
            ls="-",
            label="recurrence of place cells (of active)",
        )
        ax.plot(
            self.pl_dat.n_edges - 1,
            np.nanmean(self.recurrence["coding"]["all"], 0),
            "k--",
            label="recurrence of place cells",
        )
        # for s in range(self.data['nSes']):
        # ax.scatter(pl_dat.n_edges,recurrence['coding']['all'][s,:],5,color=[0.8,0.8,0.8],marker='o')
        # ax.scatter(pl_dat.n_edges,recurrence['coding']['continuous'][s,:],5,color=[0.6,1,0.6],marker='o')

        # axs[0][1].scatter(pl_dat.n_edges,recurrence['coding']['all'][s,:],5,color=[0.8,0.8,0.8],marker='o')
        # axs[0][1].scatter(pl_dat.n_edges,recurrence['coding']['continuous'][s,:],5,color=[0.6,1,0.6],marker='o')
        # ax.plot(pl_dat.n_edges,np.nanmean(recurrence['coding']['all'],0),color='k')
        ax.legend(loc="upper right", fontsize=10)
        ax.set_xlim([0, t_ses[-1]])
        ax.set_ylim([0, 1])
        ax.set_ylabel("fraction", fontsize=14)
        ax.set_xlabel("session diff. $\Delta$ s", fontsize=14)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.show(block=False)
        if sv:
            self.pl_dat.save_fig("ROI_stability_PC")

    def plot_stats(self, sv=False):
        print("plotting general statistics of PC and nPCs")

        nSes = self.data["nSes"]
        mask_PC = ~self.status["activity"][..., 2]
        mask_active = ~(
            self.status["activity"][..., 1] & (~self.status["activity"][..., 2])
        )

        fr_key = "firingrate"  #'firingrate_adapt'#firingrate_adapt'
        ### stats of all (PC & nPC) cells
        plt.figure(figsize=(4, 3), dpi=self.pl_dat.sv_opt["dpi"])

        key_arr = ["SNR_comp", fr_key, "Isec_value", "MI_value"]

        for i, key in enumerate(key_arr):
            print(key)
            ## firingrate
            dat_nPC = np.ma.array(self.stats[key], mask=mask_active, fill_value=np.NaN)
            dat_PC = np.ma.array(self.stats[key], mask=mask_PC, fill_value=np.NaN)

            dat_PC_mean = np.zeros(nSes) * np.NaN
            dat_PC_CI = np.zeros((2, nSes)) * np.NaN
            dat_nPC_mean = np.zeros(nSes) * np.NaN
            dat_nPC_CI = np.zeros((2, nSes)) * np.NaN
            for s in np.where(self.status["sessions"])[0]:
                dat_PC_s = dat_PC[:, s].compressed()
                # print(dat_PC_s)
                if len(dat_PC_s):
                    dat_PC_mean[s] = np.mean(dat_PC_s)
                    dat_PC_CI[:, s] = np.percentile(dat_PC_s, q=[32.5, 67.5])

                dat_nPC_s = dat_nPC[:, s].compressed()
                if len(dat_PC_s):
                    dat_nPC_mean[s] = np.mean(dat_nPC_s)
                    dat_nPC_CI[:, s] = np.percentile(dat_nPC_s, q=[32.5, 67.5])

            ax = plt.subplot(2, 2, i + 1)
            self.pl_dat.plot_with_confidence(
                ax, range(nSes), dat_nPC_mean, dat_nPC_CI, col="k", ls="-", label=None
            )
            self.pl_dat.plot_with_confidence(
                ax,
                range(nSes),
                dat_PC_mean,
                dat_PC_CI,
                col="tab:blue",
                ls="-",
                label=None,
            )

            # dat_bs_nPC = bootstrap_data(lambda x : (np.mean(x,0),0),dat_nPC,N_bs)
            # dat_bs_PC = bootstrap_data(lambda x : (np.mean(x,0),0),dat_PC,N_bs)
            # dat_bs_nPC[0][~self.status['sessions']] = np.NaN
            # dat_bs_PC[0][~self.status['sessions']] = np.NaN
            #
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_bs_nPC[0],dat_bs_nPC[1],col='k',ls='-',label=None)
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_bs_PC[0],dat_bs_PC[1],col='tab:blue',ls='-',label=None)
            ax.set_ylabel(key)
        ax = plt.subplot(221)
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_xticklabels([])

        ax = plt.subplot(222)
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_xticklabels([])
        ax.set_ylabel("$\\bar{\\nu}$")

        if nSes > 20:
            s = 20
        else:
            s = 10
        ax = plt.axes([0.8, 0.65, 0.15, 0.075])
        dat_nPC = np.ma.array(self.stats[fr_key], mask=mask_active, fill_value=np.NaN)
        dat_PC = np.ma.array(self.stats[fr_key], mask=mask_PC, fill_value=np.NaN)
        # ax.hist(dat_nPC[:,s][~mask_active[:,s]],np.linspace(0,0.3,21),density=True,facecolor='k',alpha=0.5)
        # ax.hist(dat_PC[:,s][~mask_PC[:,s]],np.linspace(0,0.3,21),density=True,facecolor='tab:blue',alpha=0.5)
        ax.hist(
            dat_nPC[:, s][~mask_active[:, s]],
            np.logspace(-2.5, 0, 21),
            density=True,
            facecolor="k",
            alpha=0.5,
        )
        ax.hist(
            dat_PC[:, s][~mask_PC[:, s]],
            np.logspace(-2.5, 0, 21),
            density=True,
            facecolor="tab:blue",
            alpha=0.5,
        )
        # ax.set_ylim([0,200])
        # ax.set_xticks()
        ax.set_xscale("log")
        ax.set_xlabel("$\\nu$")
        ax.set_yticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax = plt.subplot(223)
        # ax.set_ylabel('$r_{value}$')
        ax.set_ylabel("$I/sec$")
        ax.set_xlabel("session")

        ax = plt.subplot(224)
        ax.set_ylabel("MI")
        ax.set_xlabel("session")
        ax.set_ylim([0, ax.get_ylim()[1]])

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig("neuronStats_nPCvPC")

        # return
        ### stats of PCs
        plt.figure(figsize=(4, 3), dpi=self.pl_dat.sv_opt["dpi"])

        ax = plt.subplot(2, 2, 1)
        nPC = self.status["activity"][..., 2].sum(0).astype("float")
        nPC[~self.status["sessions"]] = np.NaN
        ax.plot(nPC, "tab:blue")
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_ylabel("# PC")

        ax2 = ax.twinx()
        dat = np.ma.array(self.fields["nModes"], mask=mask_PC)
        ax2.plot(dat.mean(0), "k-")
        ax2.set_ylim([1, 1.3])

        key_arr = ["width", "amplitude", "reliability"]

        ## field width
        for i, key in enumerate(key_arr):
            print(key)
            if len(self.fields[key].shape) == 4:
                dat = np.ma.array(
                    self.fields[key][..., 0],
                    mask=~self.status["fields"],
                    fill_value=np.NaN,
                )
            else:
                dat = np.ma.array(
                    self.fields[key], mask=~self.status["fields"], fill_value=np.NaN
                )

            ax = plt.subplot(2, 2, i + 2)  # axes([0.1,0.6,0.35,0.35])
            dat_mean = np.zeros(nSes) * np.NaN
            dat_CI = np.zeros((4, nSes)) * np.NaN
            for s in np.where(self.status["sessions"])[0]:
                dat_s = dat[:, s, :].compressed()
                if len(dat_s):
                    dat_mean[s] = np.mean(dat_s)
                    dat_CI[:, s] = np.percentile(dat_s, q=[2.5, 32.5, 67.5, 97.5])
                # ax.boxplot(dat_s,positions=[s],widths=0.4,whis=[5,95],notch=True,bootstrap=100,flierprops=dict(marker='.',markeredgecolor='None',markerfacecolor=[0.5,0.5,0.5],markersize=2))

            # dat_bs = bootstrap_data(lambda x : (np.mean(x,(0,2)),0),dat,N_bs)
            # dat_bs[0][~self.status['sessions']] = np.NaN
            # dat = dat[mask_fields]#[dat.mask] = np.NaN
            # dat[mask_fields] = np.NaN

            # ax.plot(width.mean((0,2)),'k')
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_bs[0],dat_bs[1],col='k',ls='-',label=None)
            # pl_dat.plot_with_confidence(ax,range(nSes),dat.mean((0,2)),np.percentile(dat.std((0,2))),col='k',ls='-',label=None)

            self.pl_dat.plot_with_confidence(
                ax,
                range(nSes),
                dat_mean,
                dat_CI[[0, 3], :],
                col="k",
                ls="-",
                label=None,
            )
            self.pl_dat.plot_with_confidence(
                ax,
                range(nSes),
                dat_mean,
                dat_CI[[1, 2], :],
                col="k",
                ls="-",
                label=None,
            )
            ax.set_ylim([0, ax.get_ylim()[1]])
            ax.set_ylabel(key)

        ax = plt.subplot(222)
        ax.set_ylabel("$\sigma$")
        ax.set_xticklabels([])

        ax = plt.subplot(223)
        ax.set_ylabel("$A$")
        ax.set_xlabel("session")

        ax = plt.subplot(224)
        ax.set_ylabel("reliability")
        ax.set_xlabel("session")

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig("neuronStats_PC")

    def plot_PC_stats(self, sv=False, N_bs=10):

        print("### plot place cell statistics ###")

        fig = plt.figure(figsize=(7, 4), dpi=self.pl_dat.sv_opt["dpi"])

        nSes = self.data["nSes"]
        nbin = self.data["nbin"]

        if nSes > 70:
            s = 70
        else:
            s = 10
        pathLoad = self.paths["neuron_detection"][s + 1]
        # ld = loadmat(pathLoad)

        ld = load_data(pathLoad)
        A = ld["A"]
        # .toarray().reshape(self.params['dims'][0],self.params['dims'][1],-1)
        # Cn = A.sum(1).reshape(self.params['dims'])
        Cn = ld["Cn"].transpose()
        Cn -= Cn.min()
        Cn /= Cn.max()

        # adjust to same reference frame
        # x_grid, y_grid = np.meshgrid(np.arange(0., self.params['dims'][0]).astype(np.float32), np.arange(0., self.params['dims'][1]).astype(np.float32))
        # x_remap = (x_grid - \
        #             self.alignment['shift'][s+1,0] + self.alignment['shift'][s,0] + \
        #             self.alignment['flow'][s+1,:,:,0] - self.alignment['flow'][s,:,:,0]).astype('float32')
        # y_remap = (y_grid - \
        #             self.alignment['shift'][s+1,1] + self.alignment['shift'][s,1] + \
        #             self.alignment['flow'][s+1,:,:,1] - self.alignment['flow'][s,:,:,1]).astype('float32')

        ax_ROI = plt.axes([0.05, 0.45, 0.3, 0.5])
        add_number(fig, ax_ROI, order=1, offset=[-50, 25])
        # plot background, based on first sessions
        ax_ROI.imshow(Cn, origin="lower", clim=[0, 1], cmap="viridis")

        # plot contours occuring in first and in second session, only, and...
        # plot contours occuring in both sessions (taken from first session)
        # idx_act = self.status["activity"][:, s, 1] & (~self.status["activity"][:, s, 2])
        # idx_PC = self.status["activity"][:, s, 2]
        # c_arr_PC = np.where(idx_PC)[0]

        n_act = self.matching["IDs"][
            self.status["activity"][:, s, 1] & (~self.status["activity"][:, s, 2]), s
        ].astype("int")
        n_PC = self.matching["IDs"][self.status["activity"][:, s, 2], s].astype("int")
        idx_PC = np.where(self.status["activity"][:, s, 2])[0]
        # print(n_PC)

        twilight = plt.get_cmap("hsv")
        cNorm = colors.Normalize(vmin=0, vmax=100)
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=twilight)

        contours = get_contours(A[:, n_act], self.params["dims"])
        for idx, contour in enumerate(contours):
            n = n_act[idx]
            ax_ROI.plot(
                contour["coordinates"][:, 1],
                contour["coordinates"][:, 0],
                linewidth=0.3,
                linestyle=":",
                color="white",
            )

        contours = get_contours(A[:, n_PC], self.params["dims"])
        for idx, contour in enumerate(contours):
            n = idx_PC[idx]
            # print(n, s)
            # print(self.status["fields"][n, s, :])
            f = np.where(self.status["fields"][n, s, :])[0][0]
            colVal = scalarMap.to_rgba(self.fields["location"][n, s, f, 0])
            ax_ROI.plot(
                contour["coordinates"][:, 1],
                contour["coordinates"][:, 0],
                linewidth=0.5,
                color=colVal,
            )

        cbaxes = plt.axes([0.345, 0.75, 0.01, 0.2])
        cb = fig.colorbar(scalarMap, cax=cbaxes, orientation="vertical")
        cb.set_label("PF location", fontsize=8)

        ax_ROI.plot(np.NaN, np.NaN, "k-", label="PC")
        # ax_ROI.plot(np.NaN,np.NaN,'k--',label='$\\alpha_{s_1}$')
        ax_ROI.plot(np.NaN, np.NaN, "k:", label="nPC")
        # ax_ROI.legend(fontsize=10,bbox_to_anchor=[1.2,1.1],loc='upper right',handlelength=1)

        sbar = ScaleBar(530.68 / 512 * 10 ** (-6), location="lower right")
        ax_ROI.add_artist(sbar)
        ax_ROI.set_xticks([])
        ax_ROI.set_yticks([])

        ax = plt.axes([0.525, 0.1, 0.375, 0.225])
        add_number(fig, ax, order=6)
        fields = np.zeros((100, nSes))
        for i, s in enumerate(np.where(self.status["sessions"])[0]):
            idx_PC = np.where(self.status["fields"][:, s, :])
            # fields[s,:] = np.nansum(self.fields['p_x'][:,s,:,:],1).sum(0)
            fields[:, s] = np.nansum(self.fields["p_x"][idx_PC[0], s, idx_PC[1], :], 0)
        fields /= fields.sum(0)
        fields = gauss_smooth(fields, (1, 0))

        im = ax.imshow(
            fields, origin="lower", aspect="auto", cmap="hot"
        )  # ,clim=[0,1])
        ax.set_xlim([-0.5, nSes - 0.5])

        cbaxes = plt.axes([0.92, 0.15, 0.01, 0.175])
        h_cb = plt.colorbar(im, cax=cbaxes)
        h_cb.set_label("place field \ndensity", fontsize=8)
        h_cb.set_ticks([])

        ax.set_ylim([0, 100])
        ax.set_xlabel("session")
        ax.set_ylabel("position")

        # idxes = [range(0,15),range(15,40),range(40,87)]
        # # idxes = [range(0,5),range(5,10),range(10,15)]
        # for (i,idx) in enumerate(idxes):
        #     # print(idx)
        #     ax = plt.axes([0.5,0.475-0.175*i,0.475,0.15])
        #     # ax = plt.subplot(len(idxes),1,i+1)
        #     fields = np.nansum(self.fields['p_x'][:,idx,:,:],2).sum(1).sum(0)
        #     fields /= fields.sum()
        #
        #     ax.bar(pl_dat.bin_edges,pl_dat.bars['GT'],width=1,facecolor=[0.8,1,0.8],edgecolor='none')
        #     ax.bar(pl_dat.bin_edges,pl_dat.bars['RW'],width=1,facecolor=[1,0.8,0.8],edgecolor='none')
        #     ax.bar(pl_dat.bin_edges,pl_dat.bars['PC'],width=1,facecolor=[0.7,0.7,1],edgecolor='none')
        #     #ax.bar(pl_dat.bin_edges,fields)
        #     # ax.hist(self.fields['location'][:,idx,0,0].flatten(),pl_dat.bin_edges-0.5,facecolor='k',width=0.8,density=True,label='Session %d-%d'%(idx[0]+1,idx[-1]+1))
        #
        #     idx_PC = np.where(self.fields['status']>=3)
        #     idx_fields = np.where((idx_PC[1] >= idx[0]) & (idx_PC[1] <=idx[-1]))[0]
        #     cov = self.fields['p_x'][idx_PC[0][idx_fields],idx_PC[1][idx_fields],idx_PC[2][idx_fields],:].sum(0)
        #     ax.bar(pl_dat.bin_edges,cov/cov.sum(),facecolor='k',width=0.9,label='Session %d-%d'%(idx[0]+1,idx[-1]+1))
        #     ax.set_xlim([0,L_track])
        #     ax.set_ylim([0,0.04])#np.nanmax(fields)*1.2])
        #
        #     if i==1:
        #         ax.set_ylabel('% of PC')
        #     else:
        #         ax.set_yticks([])
        #     if not (i==2):
        #         ax.set_xticks([])
        #     ax.legend(fontsize=10,loc='upper right')
        # ax.set_xlabel('position [bins]')

        # print('plot fmap corr vs distance for 1. all PC, 2. all active')
        # ax = plt.axes([0.1,0.1,0.25,0.25])
        # D_ROIs_PC =  sp.spatial.distance.pdist(self.matching['com'][c_arr_PC,s,:]))
        # ax.hist(D_ROIs[mat_PC].flat,np.linspace(0,700,201))

        nsteps = 51
        d_arr = np.linspace(0, 50, nsteps)
        mean_corr = np.zeros((nsteps, nSes, 2)) * np.NaN

        for s in tqdm(np.where(self.status["sessions"])[0]):  # range(10,15)):
            D_ROIs = sp.spatial.distance.squareform(
                sp.spatial.distance.pdist(self.matching["com"][:, s, :])
            )
            np.fill_diagonal(D_ROIs, np.NaN)

            idx_PC = self.status["activity"][:, s, 2]
            # print(idx_PC.sum())
            if idx_PC.sum() > 1:
                mat_PC = idx_PC[:, np.newaxis] & idx_PC[:, np.newaxis].T
                D_PCs = D_ROIs[idx_PC, :]
                D_PCs = D_PCs[:, idx_PC]
                NN = np.nanargmin(D_PCs, 1)

            C = np.corrcoef(self.stats["firingmap"][:, s, :])
            np.fill_diagonal(C, np.NaN)

            for i in range(nsteps - 1):
                idx = (D_ROIs > d_arr[i]) & (D_ROIs <= d_arr[i + 1])
                if idx_PC.sum() > 0:
                    mean_corr[i, s, 0] = np.mean(C[idx & mat_PC])
                mean_corr[i, s, 1] = np.mean(C[idx])

        dat_bs_PC = bootstrap_data(
            lambda x: (np.nanmean(x, 0), 0), mean_corr[..., 0].T, N_bs
        )
        dat_bs = bootstrap_data(
            lambda x: (np.nanmean(x, 0), 0), mean_corr[..., 1].T, N_bs
        )

        ax = plt.axes([0.1, 0.125, 0.25, 0.2])
        add_number(fig, ax, order=2)

        ax.plot(
            D_ROIs,
            C,
            ".",
            markerfacecolor=[0.6, 0.6, 0.6],
            markersize=0.5,
            markeredgewidth=0,
        )
        # ax.plot(D_PCs[range(n_PC),NN],C[range(n_PC),NN],'g.',markersize=1,markeredgewidth=0)
        ax.plot([0, 50], [0, 0], "r:", linewidth=0.75)
        # ax.plot(d_arr,np.nanmean(mean_corr[...,0],1),'r-',linewidth=1)
        self.pl_dat.plot_with_confidence(
            ax,
            d_arr,
            dat_bs_PC[0],
            dat_bs_PC[1],
            col="tab:blue",
            ls="-",
            label="place cells",
        )
        self.pl_dat.plot_with_confidence(
            ax, d_arr, dat_bs[0], dat_bs[1], col="k", ls="--", label="others"
        )

        # ax.plot(d_arr,np.nanmean(mean_corr[...,1],1),'r--',linewidth=1)
        ax.set_xlim([0, 50])
        ax.set_ylim([-0.25, 1])
        ax.set_xlabel("d [$\mu$m]")
        ax.set_ylabel("$c_{map(\\nu)}$")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.legend(
            fontsize=8, loc="upper right", bbox_to_anchor=[1.2, 1.2], handlelength=1
        )

        mask_PC = ~self.status["activity"][..., 2]
        mask_active = ~(
            self.status["activity"][..., 1] & (~self.status["activity"][..., 2])
        )

        fr_key = "oof_firingrate_adapt"  #'firingrate_adapt'
        # if not (fr_key in self.stats.keys()):
        # fr_key = 'firingrate'
        ### stats of all (PC & nPC) cells

        key_arr = ["rate", "SNR_comp"]  # ,'r_values','MI_value']

        for i, key in enumerate(key_arr):
            ## firingrate
            if key == "SNR_comp":
                mask_PC = ~self.status["activity"][..., 2]
                # mask_active = ~(self.status['activity'][...,1]&(~self.status['activity'][...,2]))
                # dat_nPC = np.ma.array(self.stats[key], mask=mask_active, fill_value=np.NaN)
                dat_PC = np.ma.array(self.stats[key], mask=mask_PC, fill_value=np.NaN)
            else:
                mask_PC = ~self.status["fields"]
                #     # mask_active = ~(self.status['activity'][...,1][...,np.newaxis] & (~self.status["fields"]))
                # dat_nPC = np.ma.array(self.fields['baseline'][...,0], mask=mask_active, fill_value=np.NaN)
                dat_PC = np.ma.array(
                    self.fields["amplitude"][..., 0],
                    # / self.fields["baseline"][..., 0, np.newaxis],
                    mask=mask_PC,
                    fill_value=np.NaN,
                )
                # dat_PC = np.ma.array(self.fields['baseline'][...,0], mask=mask_PC, fill_value=np.NaN)

            dat_PC_mean = np.zeros(nSes) * np.NaN
            dat_PC_CI = np.zeros((2, nSes)) * np.NaN
            # dat_nPC_mean = np.zeros(nSes)*np.NaN
            # dat_nPC_CI = np.zeros((2,nSes))*np.NaN
            for s in np.where(self.status["sessions"])[0]:
                dat_PC_s = dat_PC[:, s].compressed()
                if len(dat_PC_s):
                    dat_PC_mean[s] = np.mean(dat_PC_s)
                    dat_PC_CI[:, s] = np.percentile(
                        dat_PC_s, q=[32.5, 67.5]
                    )  # ,q=[2.5,97.5])#
                # dat_nPC_s = dat_nPC[:,s].compressed()
                # dat_nPC_mean[s] = np.mean(dat_nPC_s)
                # dat_nPC_CI[:,s] = np.percentile(dat_nPC_s,q=[32.5,67.5])#,q=[2.5,97.5])#

            # ax = plt.axes([0.525+0.2*i,0.775,0.175,0.2])#subplot(2,2,i+1)
            ax = plt.axes([0.525, 0.4 + 0.2 * i, 0.375, 0.125])  # subplot(2,2,i+1)
            add_number(fig, ax, order=5 - i, offset=[-150, 25])
            self.pl_dat.plot_with_confidence(
                ax,
                range(nSes),
                dat_PC_mean,
                dat_PC_CI,
                col="tab:blue",
                ls="-",
                label="place cells",
            )
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_nPC_mean,dat_nPC_CI,col='k',ls='--',label='others')
            ax.set_xlim([-0.5, nSes - 0.5])
            self.pl_dat.remove_frame(ax, ["top", "right"])
            # dat_bs_nPC = bootstrap_data(lambda x : (np.mean(x,0),0),dat_nPC,N_bs)
            # dat_bs_PC = bootstrap_data(lambda x : (np.mean(x,0),0),dat_PC,N_bs)
            # dat_bs_nPC[0][~self.status['sessions']] = np.NaN
            # dat_bs_PC[0][~self.status['sessions']] = np.NaN
            #
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_bs_nPC[0],dat_bs_nPC[1],col='k',ls='-',label=None)
            # pl_dat.plot_with_confidence(ax,range(nSes),dat_bs_PC[0],dat_bs_PC[1],col='tab:blue',ls='-',label=None)
            # ax.set_ylabel(key)
            if i == 1:
                ax.set_ylabel("SNR")
            else:
                # ax.set_ylabel('$\\bar{\\nu}$')
                ax.set_ylabel("$A/A_0$")
            ax.set_ylim([0, ax.get_ylim()[1]])
            ax.set_xticklabels([])

        ax = plt.axes([0.525, 0.8, 0.375, 0.15])
        add_number(fig, ax, order=3, offset=[-150, 25])
        ax.plot(
            np.where(self.status["sessions"])[0],
            self.status["activity"][:, self.status["sessions"], 1].sum(0),
            "o",
            color="k",
            markersize=2,
            label="# active neurons",
        )
        ax.plot(
            np.where(self.status["sessions"])[0],
            self.status["activity"][:, self.status["sessions"], 2].sum(0),
            "o",
            color="tab:blue",
            markersize=2,
            label="# place cells",
        )
        ax.set_ylim([0, np.max(self.status["activity"].sum(axis=0)[..., 1]) * 1.2])
        ax.set_xlim([-0.5, nSes - 0.5])
        ax.set_xticklabels([])
        ax.set_ylabel("# neurons", fontsize=8)
        self.pl_dat.remove_frame(ax, ["top", "right"])
        # ax.legend(loc='upper right')
        # ax.set_xlabel('session s',fontsize=14)

        ax2 = ax.twinx()
        ax2.plot(
            np.where(self.status["sessions"])[0],
            self.status["activity"][:, self.status["sessions"], 2].sum(0)
            / self.status["activity"][:, self.status["sessions"], 1].sum(0),
            "--",
            color="tab:blue",
            linewidth=0.5,
        )
        ax2.set_ylim([0, 0.5])
        ax2.yaxis.label.set_color("tab:blue")
        ax2.tick_params(axis="y", colors="tab:blue")
        ax2.set_ylabel("PC fraction", fontsize=8)
        self.pl_dat.remove_frame(ax2, ["top", "right"])

        plt.tight_layout()
        plt.show(block=False)

        # ax = plt.subplot(221)
        # ax.set_xticklabels([])

        # ax = plt.subplot(222)
        # ax.set_ylim([0,ax.get_ylim()[1]])
        # ax.set_xticklabels([])
        # ax.set_ylabel('$\\bar{\\nu}$')

        plt.tight_layout()

        plt.show(block=False)
        if sv:
            self.pl_dat.save_fig("PC_statistics")

    #      overrepr = occupancy(:,1:para.nbin)./(sum(nROI(:,3:5),2)/para.nbin);

    def plot_firingmaps(self, sv=True):

        nSes = self.data["nSes"]
        nbin = self.data["nbin"]
        print("### plot firingmap over sessions and over time ###")
        # if nSes>65:
        # s_ref = 50
        # else:
        s_ref = 10
        n_plots = 5
        n_plots_half = (n_plots - 1) / 2
        # ordered = False

        # if ordered:
        # print('aligned order')
        idxes_tmp = np.where(self.status["fields"][:, s_ref, :])
        idxes = idxes_tmp[0]
        sort_idx = np.argsort(
            self.fields["location"][idxes_tmp[0], s_ref, idxes_tmp[1], 0]
        )

        # idxes = np.where(self.status['activity'][:,s_ref,2])[0]
        # sort_idx = np.argsort(np.nanmin(self.fields['location'][self.status['activity'][:,s_ref,2],s_ref,:,0],-1))
        sort_idx_ref = idxes[sort_idx]
        nID_ref = len(sort_idx_ref)
        # else:
        # print('non-aligned order')

        width = 0.11
        fig = plt.figure(figsize=(7, 5), dpi=self.pl_dat.sv_opt["dpi"])

        ax = plt.axes([0.75, 0.05, 0.225, 0.275])
        pic_path = "/home/wollex/Data/Science/PhD/Thesis/pics/others/status_sketch.png"
        ax.axis("off")
        if os.path.exists(pic_path):
            im = mpimg.imread(pic_path)
            ax.imshow(im)
            ax.set_xlim([0, im.shape[1]])
            self.pl_dat.add_number(fig, ax, order=4, offset=[-75, 50])

        ax = plt.axes([0.1, 0.525, width, 0.4])
        self.pl_dat.add_number(fig, ax, order=1)
        ax = plt.axes([0.1, 0.08, width, 0.4])
        self.pl_dat.add_number(fig, ax, order=2)
        for i, s in enumerate(
            range(int(s_ref - n_plots_half), int(s_ref + n_plots_half) + 1)
        ):
            ax = plt.axes([0.1 + i * width, 0.525, width, 0.4])
            # ax = plt.subplot(2,n_plots+1,i+1)
            idxes_tmp = np.where(self.status["fields"][:, s, :])
            idxes = idxes_tmp[0]
            sort_idx = np.argsort(
                self.fields["location"][idxes_tmp[0], s, idxes_tmp[1], 0]
            )
            # idxes = np.where(self.status['activity'][:,s,2])[0]
            # sort_idx = np.argsort(np.nanmin(self.fields['location'][self.status['activity'][:,s,2],s,:,0],-1))
            sort_idx = idxes[sort_idx]
            nID = len(sort_idx)

            firingmap = self.stats["firingmap"][sort_idx, s, :]
            firingmap = gauss_smooth(firingmap, [0, 1])
            firingmap = firingmap - np.nanmin(firingmap, 1)[:, np.newaxis]
            # firingmap = firingmap / np.nanmax(firingmap,1)[:,np.newaxis]
            ax.imshow(firingmap, aspect="auto", origin="upper", cmap="jet", clim=[0, 2])

            title_str = "s"
            ds = s - s_ref
            if ds < 0:
                title_str += "%d" % ds
            elif ds > 0:
                title_str += "+%d" % ds

            ax.set_title(title_str)

            # ax.plot([self.params['zone_idx']['reward'][0],self.params['zone_idx']['reward'][0]],[1,nID],color='g',linestyle=':',linewidth=3)
            # ax.plot([self.params['zone_idx']['reward'][1],self.params['zone_idx']['reward'][1]],[1,nID],color='g',linestyle=':',linewidth=3)
            if i == 0:
                # ax.plot([self.params['zone_idx']['gate'][0],self.params['zone_idx']['gate'][0]],[1,nID],color='r',linestyle=':',linewidth=3)
                # ax.plot([self.params['zone_idx']['gate'][1],self.params['zone_idx']['gate'][1]],[1,nID],color='r',linestyle=':',linewidth=3)
                # ax.set_xticks(np.linspace(0,nbin,3))
                # ax.set_xticklabels(np.linspace(0,nbin,3))
                ax.set_ylabel("Neuron ID")
            else:
                ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xlim([0, nbin])
            ax.set_ylim([nID, 0])

            ax = plt.axes([0.1 + i * width, 0.08, width, 0.4])
            # ax = plt.subplot(2,n_plots+1,i+2+n_plots)
            # if not ordered:

            firingmap = self.stats["firingmap"][sort_idx_ref, s, :]
            firingmap = gauss_smooth(firingmap, [0, 1])
            firingmap = firingmap - np.nanmin(firingmap, 1)[:, np.newaxis]
            # firingmap = firingmap / np.nanmax(firingmap,1)[:,np.newaxis]
            im = ax.imshow(
                firingmap, aspect="auto", origin="upper", cmap="jet", clim=[0, 2]
            )

            ax.set_xlim([0, nbin])
            if i == 0:
                ax.set_ylabel("Neuron ID")
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            ax.set_ylim([nID_ref, 0])

            if i == n_plots_half:
                ax.set_xlabel("Location [bin]")

        cbaxes = plt.axes([0.67, 0.725, 0.01, 0.2])
        cb = fig.colorbar(im, cax=cbaxes, orientation="vertical")
        # cb.set_ticks([0,1])
        # cb.set_ticklabels(['low','high'])
        cb.set_label("$\\nu$", fontsize=10)

        ax = plt.axes([0.825, 0.5, 0.125, 0.45])
        self.pl_dat.add_number(fig, ax, order=3, offset=[-125, 30])
        idx_strong_PC = np.where(
            (self.status["activity"][..., 2].sum(1) > 10)
            & (self.status["activity"][..., 1].sum(1) < 70)
        )[0]
        idx_PC = np.random.choice(idx_strong_PC)  ## 28,1081
        print(idx_PC)
        firingmap = self.stats["firingmap"][idx_PC, ...]
        firingmap = gauss_smooth(firingmap, [0, 1])
        firingmap = firingmap - np.nanmin(firingmap, 1)[:, np.newaxis]
        # firingmap = firingmap / np.nanmax(firingmap,1)[:,np.newaxis]
        ax.imshow(firingmap, aspect="auto", origin="upper", cmap="jet", clim=[0, 2])
        ax.barh(
            range(nSes),
            -(self.status["activity"][idx_PC, :, 2] * 10.0),
            left=-5,
            facecolor="r",
        )
        # idx_coding = np.where(self.status[idx_PC,:,2])[0]
        # ax.plot(-np.ones_like(idx_coding)*10,idx_coding,'ro')
        ax.set_xlim([-10, nbin])
        ax.set_ylim([nSes, 0])
        ax.set_ylabel("Session")
        ax.set_xlabel("Location [bins]")
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        # plt.set_cmap('jet')
        plt.tight_layout()
        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig("PC_mapDynamics")

    def plot_stability_dynamics(
        self, n_processes=8, reprocess=False, N_bs=10, sv=False
    ):

        SD = 1.96
        nSes = self.data["nSes"]
        nbin = self.data["nbin"]
        # L_track = 100
        L_track = nbin
        steps = 100

        bin_to_steps = steps / nbin

        ### ds = 0
        plt0 = True
        if plt0:
            p_shift = np.zeros(steps)
            for s in np.where(self.status["sessions"])[0]:
                idx_field = np.where(self.status["fields"][:, s, :])
                for c, f in zip(idx_field[0], idx_field[1]):
                    roll = round(
                        (-self.fields["location"][c, s, f, 0] + nbin / 2) * bin_to_steps
                    )
                    p_shift += np.roll(self.fields["p_x"][c, s, f, :], roll)
            p_shift /= p_shift.sum()

            PC_idx = np.where(self.status["activity"][..., 2])
            N_data = len(PC_idx[0])
            print("N data: %d" % N_data)

            p_ds0, p_cov = fit_shift_model(p_shift)

        ### ds > 0
        s1_shifts, s2_shifts, f1, f2 = np.unravel_index(
            self.compare["pointer"].col,
            (
                nSes,
                nSes,
                self.params["field_count_max"],
                self.params["field_count_max"],
            ),
        )
        c_shifts = self.compare["pointer"].row

        celltype = "all"
        if celltype == "all":
            idx_celltype = self.status["activity"][c_shifts, s1_shifts, 2]
        if celltype == "gate":
            idx_celltype = self.status["activity"][c_shifts, s1_shifts, 3]
        if celltype == "reward":
            idx_celltype = self.status["activity"][c_shifts, s1_shifts, 4]

        idx_celltype = (
            idx_celltype
            & self.status["sessions"][s1_shifts]
            & self.status["sessions"][s2_shifts]
        )

        if reprocess:
            self.stability = self.calculate_placefield_stability(
                dsMax=nSes,
                celltype="all",
                n_processes=n_processes,
                N_bs=100,
                p_keys=["all", "cont", "mix", "discont", "silent_mix", "silent"],
            )

        fig = plt.figure(figsize=(7, 4), dpi=self.pl_dat.sv_opt["dpi"])

        ax_distr = plt.axes([0.075, 0.11, 0.35, 0.325])
        self.pl_dat.add_number(fig, ax_distr, order=2, offset=[-100, 50])

        ## plot shift distributions for different ds
        for j, ds in tqdm(enumerate([1, 5, 10, 20, 40])):  # min(nSes,30)):

            Ds = s2_shifts - s1_shifts
            idx_ds = np.where(
                (Ds == ds)
                & idx_celltype
                & self.status["sessions"][s1_shifts]
                & self.status["sessions"][s2_shifts]
            )[0]

            idx_shifts = self.compare["pointer"].data[idx_ds].astype("int") - 1
            shifts_distr = self.compare["shifts_distr"][idx_shifts, :].toarray()

            _, _, _, shift_distr = bootstrap_shifts(
                fit_shift_model, shifts_distr, N_bs, nbin
            )

            ax_distr.plot(
                np.linspace(-L_track / 2 + 0.5, L_track / 2 - 0.5, nbin),
                shift_distr.mean(0),
                color=[0.2 * j, 0.2 * j, 0.2 * j],
                linewidth=0.5,
                label="$\Delta$ s = %d" % ds,
            )

            # CI = np.percentile(shift_distr,[5,95],0)
            # ax_distr.errorbar(np.linspace(-L_track/2+0.5,L_track/2-0.5,nbin),shift_distr.mean(0),shift_distr.mean(0)-CI[0,:],CI[1,:]-shift_distr.mean(0),fmt='none',ecolor=[1,0.,0.],elinewidth=0.5)

        self.pl_dat.remove_frame(ax_distr, ["top", "right"])

        dx_arr = np.linspace(-L_track / 2 + 0.5, L_track / 2 - 0.5, steps)
        ax_distr.plot(dx_arr, p_shift, "k--", linewidth=0.5)
        ax_distr.set_xlim([-L_track / 2, L_track / 2])
        ax_distr.set_ylim([0, 0.1])
        ax_distr.set_xlabel("field shift $\Delta \\theta$ [bin]")
        ax_distr.set_ylabel("$\\left \\langle p(\Delta \\theta) \\right \\rangle$")
        ax_distr.set_yticks([])
        ax_distr.legend(
            loc="upper left", fontsize=8, handlelength=1, bbox_to_anchor=[0.05, 1.1]
        )

        N_data = np.zeros(nSes) * np.NaN

        D_KS = np.zeros(nSes) * np.NaN
        N_stable = np.zeros(nSes) * np.NaN
        N_total = np.zeros(nSes) * np.NaN  ### number of PCs which could be stable
        # fig = plt.figure()
        p_rec_alt = np.zeros(nSes) * np.NaN

        dx_arr = np.linspace(-L_track / 2 + 0.5, L_track / 2 - 0.5, nbin)
        for ds in range(1, nSes):  # min(nSes,30)):
            Ds = s2_shifts - s1_shifts
            idx_ds = np.where(
                (Ds == ds)
                & self.status["sessions"][s1_shifts]
                & self.status["sessions"][s2_shifts]
            )[0]
            N_data[ds] = len(idx_ds)

            idx_shifts = self.compare["pointer"].data[idx_ds].astype("int") - 1
            shifts = self.compare["shifts"][idx_shifts]
            N_stable[ds] = (
                np.abs(shifts) < (SD * self.stability["all"]["mean"][0, 2])
            ).sum()
            shifts_distr = self.compare["shifts_distr"][idx_shifts, :].toarray().sum(0)
            shifts_distr /= shifts_distr.sum()

            session_bool = np.pad(
                self.status["sessions"][ds:], (0, ds), constant_values=False
            ) & np.pad(self.status["sessions"][:], (0, 0), constant_values=False)
            N_total[ds] = self.status["fields"][:, session_bool, :].sum()
            # if ds < 20:
            #     plt.subplot(5,4,ds)
            #     plt.plot(dx_arr,np.cumsum(shifts_distr),'k')
            #     plt.plot(dx_arr,np.cumsum(fun_distr),'r')
            #     plt.title('$\Delta s=%d$'%ds)
            fun_distr = F_shifts(
                dx_arr,
                self.stability["all"]["mean"][ds, 0],
                self.stability["all"]["mean"][ds, 1],
                self.stability["all"]["mean"][ds, 2],
                self.stability["all"]["mean"][ds, 3],
            )

            D_KS[ds] = np.abs(
                np.nancumsum(shifts_distr) - np.nancumsum(fun_distr)
            ).max()

            p_rec_alt[ds] = N_stable[ds] / N_data[ds]
        print(D_KS)
        # plt.show(block=False)

        # plt.figure(fig_test.number)
        ax_p1 = plt.axes([0.05, 0.825, 0.175, 0.1])
        ax_p2 = plt.axes([0.05, 0.675, 0.175, 0.1])
        ax_shift1 = plt.axes([0.275, 0.825, 0.175, 0.1])
        ax_shift2 = plt.axes([0.275, 0.675, 0.175, 0.1])
        self.pl_dat.add_number(fig, ax_p1, order=1, offset=[-50, 25])

        ### obtain shift data
        Ds = s2_shifts - s1_shifts
        ds = 1
        idx_ds = np.where(
            (Ds == ds)
            & self.status["sessions"][s1_shifts]
            & self.status["sessions"][s2_shifts]
        )[0]
        for i, (ax, ax_shift) in enumerate(zip([ax_p1, ax_p2], [ax_shift1, ax_shift2])):

            ## choose random shift entry
            idx = np.random.choice(idx_ds)

            ## obtain field location posterior for both sessions
            p1 = self.fields["p_x"][c_shifts[idx], s1_shifts[idx], f1[idx], :]
            p2 = self.fields["p_x"][c_shifts[idx], s2_shifts[idx], f2[idx], :]

            ## calculate shift between location posteriors
            _, dp = periodic_distr_distance(p1, p2, nbin, N_bs=10000, mode="bootstrap")

            ## plot location posteriors and shift
            ax.plot(p1, color="tab:orange", label="$p(\\theta_s$)")
            ax.plot(p2, color="tab:blue", label="$p(\\theta_{s+\Delta s})$")
            ax_shift.plot(
                np.linspace(-L_track / 2 + 0.5, L_track / 2 - 0.5, nbin),
                dp,
                "k",
                label="$p(\Delta \\theta)$",
            )

            ## set axis properties
            self.pl_dat.remove_frame(ax, ["top", "left", "right"])
            self.pl_dat.remove_frame(ax_shift, ["top", "left", "right"])
            ax.set_yticks([])
            ax_shift.set_yticks([])

        ## further set axis-specific properties
        ax_p1.legend(
            fontsize=8, handlelength=1, loc="upper right", bbox_to_anchor=[1.2, 1.6]
        )
        ax_p1.set_xticklabels([])
        ax_shift1.legend(
            fontsize=8, handlelength=1, loc="upper right", bbox_to_anchor=[1.2, 1.6]
        )
        ax_shift1.set_xticklabels([])

        ax_p2.set_xlabel("position")
        ax_shift2.set_xlabel("field shift $\Delta \\theta$")

        ## plot sketch of shift distribution
        ax_img = plt.axes([0.3, 0.3, 0.15, 0.15])

        # x_arr = np.linspace(-49.5, 49.5, nbin)
        x_arr = np.linspace(-nbin / 2 + 0.5, nbin / 2 - 0.5, nbin)
        r = 0.3
        sig = 5
        y_arr = F_shifts(x_arr, 1 - r, r, sig, 0)
        ax_img.fill_between(x_arr, y_arr, color="tab:blue")
        ax_img.fill_between(x_arr, (1 - r) / nbin, color="tab:red")
        plt.plot([-sig * SD, -sig * SD], [0, 4 * (1 - r) / nbin], ":", color="tab:blue")
        plt.plot([sig * SD, sig * SD], [0, 4 * (1 - r) / nbin], ":", color="tab:blue")

        self.pl_dat.remove_frame(ax_img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # x_lim = np.where(self.status['sessions'])[0][-1] - np.where(self.status['sessions'])[0][0] + 1
        x_lim = (
            np.where(self.status["sessions"])[0][-1]
            - np.where(self.status["sessions"])[0][0]
            + 1
        )
        ax_D = plt.axes([0.6, 0.8, 0.375, 0.13])
        ax_D.plot(range(1, nSes + 1), D_KS, "k")
        ax_D.set_xlim([0, x_lim])
        ax_D.set_ylabel("$D_{KS}$")
        ax_D.yaxis.set_label_coords(-0.15, 0.5)
        ax_D.set_xticklabels([])
        ax_D.set_ylim([0, 0.2])

        self.pl_dat.add_number(fig, ax_D, order=3)

        ax_mu = plt.axes([0.6, 0.635, 0.375, 0.13])
        ax_sigma = plt.axes([0.6, 0.46, 0.375, 0.13])
        ax_r = plt.axes([0.6, 0.285, 0.375, 0.13])

        ax_sigma.plot(
            [0, nSes], [p_ds0[2], p_ds0[2]], linestyle="--", color=[0.6, 0.6, 0.6]
        )
        ax_sigma.text(10, p_ds0[2] + 1, "$\sigma_0$", fontsize=8)
        ax_mu.plot([0, nSes], [0, 0], linestyle=":", color=[0.6, 0.6, 0.6])

        sig_theta = self.stability["all"]["mean"][0, 2]
        r_random = 2 * SD * self.stability["all"]["mean"][0, 2] / nbin
        ax_r.plot(
            [1, nSes], [r_random, r_random], "--", color="tab:blue", linewidth=0.5
        )
        ax_r.plot([0, nSes], [0.5, 0.5], linestyle=":", color=[0.6, 0.6, 0.6])

        # pl_dat.plot_with_confidence(ax_mu,range(1,nSes+1),p['all']['mean'][:,3],p['all']['mean'][:,3]+np.array([[-1],[1]])*p['all']['std'][:,3]*SD,'k','-')
        # pl_dat.plot_with_confidence(ax_sigma,range(1,nSes+1),p['all']['mean'][:,2],p['all']['mean'][:,2]+np.array([[-1],[1]])*p['all']['std'][:,2]*SD,'k','-')
        # pl_dat.plot_with_confidence(ax_r,range(1,nSes+1),p['all']['mean'][:,1],p['all']['mean'][:,1]+np.array([[-1],[1]])*p['all']['std'][:,1]*SD,'k','-')

        self.pl_dat.plot_with_confidence(
            ax_mu,
            range(1, nSes + 1),
            self.stability["all"]["mean"][:, 3],
            self.stability["all"]["CI"][..., 3].T,
            "k",
            "-",
        )
        self.pl_dat.plot_with_confidence(
            ax_sigma,
            range(1, nSes + 1),
            self.stability["all"]["mean"][:, 2],
            self.stability["all"]["CI"][..., 2].T,
            "k",
            "-",
        )
        self.pl_dat.plot_with_confidence(
            ax_r,
            range(1, nSes + 1),
            self.stability["all"]["mean"][:, 1],
            self.stability["all"]["CI"][..., 1].T,
            "k",
            "-",
        )

        p_corr = np.minimum(
            1,
            self.stability["all"]["mean"][:, 1]
            + (1 - self.stability["all"]["mean"][:, 1])
            * (2 * SD * self.stability["all"]["mean"][0, 2] / nbin),
        )
        p_SD = np.sqrt(
            (1 - 2 * SD * self.stability["all"]["mean"][0, 2] / nbin) ** 2
            * self.stability["all"]["std"][:, 1] ** 2
            + ((1 - self.stability["all"]["mean"][:, 1]) * 2 * SD / nbin) ** 2
            * self.stability["all"]["std"][0, 2] ** 2
        )
        # pl_dat.plot_with_confidence(ax_r,range(1,nSes+1),p_corr,p_SD,'tab:blue','-')
        ax_r.plot(range(nSes), p_rec_alt, "-", color="tab:blue")

        # ax_r.plot(range(1,nSes+1),p_corr,'k--')
        # pl_dat.plot_with_confidence(ax_r,range(1,nSes+1),p_corr,p_corr+np.array([[-1],[1]])*p['all']['std'][:,1]*SD,'k','-',label='stable place fields (of rec. place cell)')

        ax_mu.set_xlim([0, x_lim])
        ax_mu.set_ylim([-20, 20])
        ax_mu.set_xticklabels([])
        ax_mu.set_ylabel("$\mu_{\Delta \\theta}$")
        ax_mu.yaxis.set_label_coords(-0.15, 0.5)
        ax_sigma.set_xlim([0, x_lim])
        ax_sigma.set_ylim([0, 20])
        ax_sigma.set_xticklabels([])
        ax_sigma.set_ylabel("$\sigma_{\Delta \\theta}$")
        ax_sigma.yaxis.set_label_coords(-0.15, 0.5)
        ax_r.set_xlim([0, x_lim])
        ax_r.set_ylim([0.0, 1])
        ax_r.set_yticks(np.linspace(0, 1, 3))
        ax_r.set_yticklabels(np.linspace(0, 1, 3))
        ax_r.set_xticklabels([])
        # ax_r.set_ylabel('$p(\\gamma_{\Delta s})$')
        ax_r.set_ylabel("$p_{\\gamma}$")
        ax_r.yaxis.set_label_coords(-0.15, 0.5)
        self.pl_dat.remove_frame(ax_D, ["top", "right"])
        self.pl_dat.remove_frame(ax_mu, ["top", "right"])
        self.pl_dat.remove_frame(ax_sigma, ["top", "right"])
        self.pl_dat.remove_frame(ax_r, ["top", "right"])
        # axs[0][1].set_ylim([0,1])

        ax_N = plt.axes([0.6, 0.11, 0.375, 0.13])
        ax_N.plot(N_data, "k", label="total")
        ax_N.plot(N_stable, "tab:blue", label="stable")
        ax_N.set_xlabel("session difference $\Delta s$")
        ax_N.set_xlim([0, x_lim])
        ax_N.set_ylabel("$N_{shifts}$")
        ax_N.yaxis.set_label_coords(-0.15, 0.5)
        self.pl_dat.remove_frame(ax_N, ["top", "right"])
        ax_N.legend(fontsize=8, loc="upper right", bbox_to_anchor=[1.0, 1.3])
        # print(N_stable/N_total)
        plt.tight_layout()
        plt.show(block=False)

        # plt.figure()
        # plt.plot(range(1,nSes+1),N_stable/N_total,'k--',linewidth=0.5)
        # plt.yscale('log')
        # plt.show(block=False)

        def plot_shift_distr(p, p_std, p_ds0):
            nSes = p.shape[0]
            f, axs = plt.subplots(2, 2, figsize=(6, 4), sharex=True)
            axs[1][0].plot(
                [0, nSes], [p_ds0[2], p_ds0[2]], linestyle="--", color=[0.6, 0.6, 0.6]
            )
            axs[1][1].plot([0, nSes], [0, 0], linestyle="--", color=[0.6, 0.6, 0.6])
            for i in range(4):
                self.pl_dat.plot_with_confidence(
                    axs[int(np.floor(i / 2))][i % 2],
                    range(nSes),
                    p[:, i],
                    p[:, i] + np.array([[-1], [1]]) * p_std[:, i] * SD,
                    "k",
                    "--",
                )
            # axs[0][1].set_yscale('log')

            axs[0][1].yaxis.set_label_position("right")
            axs[0][1].yaxis.tick_right()

            axs[1][1].yaxis.set_label_position("right")
            axs[1][1].yaxis.tick_right()

            axs[0][0].set_xlim([0, max(20, nSes / 2)])
            axs[0][0].set_ylim([0, 1])
            axs[0][1].set_ylim([0, 1])
            axs[1][0].set_ylim([0, 10])
            axs[1][1].set_ylim([-10, 10])

            axs[1][0].set_xlabel("$\Delta$ s", fontsize=14)
            axs[1][1].set_xlabel("$\Delta$ s", fontsize=14)
            axs[0][0].set_ylabel("1-$r_{stable}$", fontsize=14)
            axs[0][1].set_ylabel("$r_{stable}$", fontsize=14)
            axs[1][0].set_ylabel("$\sigma$", fontsize=14)
            axs[1][1].set_ylabel("$\mu$", fontsize=14)
            plt.tight_layout()
            # plt.title('cont')
            plt.show(block=False)

        if sv:
            self.pl_dat.save_fig("stability_dynamics")

        plot_shift_distr(
            self.stability["all"]["mean"], self.stability["all"]["std"], p_ds0
        )
        # for key in p.keys():
        #   plot_shift_distr(p[key]['mean'],p[key]['std'],p_ds0)

        t_ses = np.arange(nSes)
        plt.figure(figsize=(4, 4))
        ax = plt.axes([0.15, 0.725, 0.8, 0.225])
        self.pl_dat.plot_with_confidence(
            ax,
            self.pl_dat.n_edges - 1,
            np.nanmean(self.recurrence["active"]["all"], 0),
            SD * np.nanstd(self.recurrence["active"]["all"], 0),
            col="k",
            ls="-",
            label="recurrence of active cells",
        )
        ax.legend(loc="lower right", fontsize=10, bbox_to_anchor=[1.05, 0.8])
        ax.set_xlim([0, t_ses[-1]])
        ax.set_ylim([0, 1.1])
        ax.set_xticklabels([])
        ax.set_ylabel("fraction", fontsize=14)
        # ax.set_xlabel('session diff. $\Delta$ s',fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax = plt.axes([0.15, 0.425, 0.8, 0.225])
        self.pl_dat.plot_with_confidence(
            ax,
            self.pl_dat.n_edges - 1,
            np.nanmean(self.recurrence["coding"]["ofactive"], 0),
            1.0 * np.nanstd(self.recurrence["coding"]["ofactive"], 0),
            col="k",
            ls="-",
            label="place cell recurrence (of rec. active)",
        )
        # ax.plot(pl_dat.n_edges-1,np.nanmean(self.recurrence['coding']['all'],0),'b--',label='recurrence of place cells')
        ax.legend(loc="lower right", fontsize=10, bbox_to_anchor=[1.05, 0.8])
        ax.set_xlim([0, t_ses[-1]])
        ax.set_ylim([0, 1.1])
        ax.set_xticklabels([])
        ax.set_ylabel("fraction", fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax = plt.axes([0.15, 0.125, 0.8, 0.225])
        p_corr = np.minimum(
            1,
            self.stability["all"]["mean"][:, 1]
            + (1 - self.stability["all"]["mean"][:, 1])
            * (2 * SD * self.stability["all"]["mean"][:, 2] / nbin),
        )
        self.pl_dat.plot_with_confidence(
            ax,
            self.pl_dat.n_edges - 1,
            p_corr,
            p_corr + np.array([[-1], [1]]) * self.stability["all"]["std"][:, 1] * SD,
            "k",
            "-",
            label="stable place fields (of rec. place cell)",
        )
        ax.set_xlim([0, t_ses[-1]])
        ax.set_ylim([0, 1.1])
        ax.set_ylabel("fraction", fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("session diff. $\Delta$ s", fontsize=14)
        ax.legend(loc="lower right", fontsize=10, bbox_to_anchor=[1.05, 0.8])

        plt.tight_layout()
        plt.show(block=False)

        return

    def plot_stability_dynamics2(self):

        nSes = self.data["nSes"]

        SD = 1.96

        # if sv:
        #     pl_dat.save_fig('stability_dynamics_hierarchy')

        plt.figure(figsize=(4, 2))

        p = self.stability

        ax = plt.subplot(111)
        self.pl_dat.plot_with_confidence(
            ax,
            self.pl_dat.n_edges - 1,
            p["all"]["mean"][:, 2],
            p["all"]["mean"][:, 2] + np.array([[-1], [1]]) * p["all"]["std"][:, 2] * SD,
            "k",
            "--",
        )
        ax.set_xlim([0, 40])
        ax.set_ylim([0, 12])
        ax.set_xlabel("session diff. $\Delta$ s")
        ax.set_ylabel("$\sigma$ [bins]", fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("session diff. $\Delta$ s", fontsize=14)
        # ax.legend(loc='lower right',fontsize=10,bbox_to_anchor=[1.05,0.8])

        plt.tight_layout()
        plt.show(block=False)
        # if sv:
        #     self.pl_dat.save_fig('stability_dynamics_width')

        # plot_shift_distr(p['cont']['mean'],p['cont']['std'],p_ds0)
        # plot_shift_distr(p['silent']['mean'],p['silent']['std'],p_ds0)

        # if sv:
        # pl_dat.save_fig('stability_dynamics_cont')
        ##plot_shift_distr(p['mix']['mean'],p['mix']['std'],p_ds0)
        # plot_shift_distr(p['discont']['mean'],p['discont']['std'],p_ds0)
        # if sv:
        # pl_dat.save_fig('stability_dynamics_disc')

        # f,axs = plt.subplots(2,2,figsize=(6,4),sharex=True)
        plt.figure(figsize=(4, 2.5))
        ax = plt.subplot(111)
        self.pl_dat.plot_with_confidence(
            ax,
            range(nSes),
            self.stability["cont"]["mean"][:, 1],
            self.stability["cont"]["mean"][:, 1]
            + np.array([[-1], [1]]) * self.stability["cont"]["std"][:, 1],
            "b",
            "--",
            label="coding",
        )
        self.pl_dat.plot_with_confidence(
            ax,
            range(nSes),
            self.stability["discont"]["mean"][:, 1],
            self.stability["discont"]["mean"][:, 1]
            + np.array([[-1], [1]]) * self.stability["discont"]["std"][:, 1],
            "r",
            "--",
            label="no coding",
        )
        # pl_dat.plot_with_confidence(ax,range(nSes),self.stability['silent']['mean'][:,1],self.stability['silent']['mean'][:,1]+np.array([[-1],[1]])*self.stability['silent']['std'][:,1]*SD,'g','--',label='silent')
        # ax.set_yscale('log')
        ax.set_ylim([0, 1.1])
        ax.set_xlim([0, 20])
        ax.set_xlabel("$\Delta$ s", fontsize=14)
        ax.set_ylabel("$r_{stable}$", fontsize=14)
        plt.tight_layout()
        plt.legend(loc="lower right", fontsize=10)
        # plt.title('cont')
        plt.show(block=False)
        # if sv:
        #     self.pl_dat.save_fig('stability_dynamics_cont_vs_disc')

        maxSes = 6
        print("what are those stable cells coding for?")
        plt.figure(figsize=(5, 2.5))

        col_arr = [[0.5, 0.5, 1], [0.5, 0.5, 0.5], [1, 0.5, 0.5], [0.5, 1, 0.5]]
        label_arr = ["continuous", "mixed", "non-coding", "silent"]
        key_arr = ["cont", "mix", "discont", "silent"]

        w_bar = 0.2
        nKey = len(key_arr)
        offset_bar = ((nKey + 1) % 2) * w_bar / 2 + (nKey // 2 - 1) * w_bar

        for i, key in enumerate(key_arr):

            plt.bar(
                np.arange(1, maxSes + 1) - offset_bar + i * w_bar,
                self.stability[key]["mean"][:maxSes, 1],
                width=w_bar,
                facecolor=col_arr[i],
                edgecolor="k",
                label=label_arr[i],
            )
            plt.errorbar(
                np.arange(1, maxSes + 1) - offset_bar + i * w_bar,
                self.stability[key]["mean"][:maxSes, 1],
                self.stability[key]["std"][:maxSes, 1],
                fmt="none",
                ecolor="r",
            )

        plt.xlabel("session difference $\Delta s$", fontsize=14)
        plt.ylabel("$\%$ stable fields", fontsize=14)
        plt.ylim([0, 1.1])
        plt.legend(loc="upper right", ncol=2)
        plt.tight_layout()
        plt.show(block=False)

        # if sv:
        #     self.pl_dat.save_fig('intercoding_state')

    def plot_param_dependent_stability(
        self,
        iterate_var=("reliability_thr", [0.1, 0.3, 0.5]),
        n_processes=8,
        stability_in=None,
    ):

        p_keys = ["all", "cont", "mix", "discont", "silent"]

        dsMax = 6
        nSteps = len(iterate_var[1])

        params = {
            "SNR_thr": 2,
            "rval_thr": 0,
            "reliability_thr": 0.3,
            "fr_thr": 0.01,
            # "pm_thr": 0.05,
            # "alpha": 1,
            # "CI_thr": 10,
            "Bayes_thr": 0,
        }

        if stability_in is None:
            stability = {}
            p_stats = {
                "mean": np.full((nSteps, dsMax, 4), np.NaN),
                "CI": np.full((nSteps, dsMax, 2, 4), np.NaN),
                "std": np.full((nSteps, dsMax, 4), np.NaN),
            }
            for key in p_keys:
                stability[key] = copy.deepcopy(p_stats)

            ## iterate over parameter
            for i, val in enumerate(iterate_var[1]):
                params[iterate_var[0]] = val

                ## update activity and field status
                self.update_status(**params)
                self.compareSessions(n_processes=n_processes)

                ## and calculate stability
                p = self.calculate_placefield_stability(
                    dsMax=dsMax,
                    celltype="all",
                    n_processes=n_processes,
                    N_bs=100,
                    p_keys=p_keys,
                )

                ## hand over results to stability dict
                for key in p.keys():
                    for stat in p[key].keys():
                        stability[key][stat][i, ...] = p[key][stat]
        else:
            stability = stability_in

        plt.figure(figsize=(4, 3))
        pop_key = "all"
        for i, ds in enumerate([1, 2, 5]):
            for pop_key in ["all", "cont", "discont", "silent"]:
                # s1_shifts,s2_shifts = np.unravel_index(self.compare['pointer'].col,(nSes,nSes))
                # idx_ds = np.where(s2_shifts-s1_shifts==ds)[0]

                if pop_key == "all":
                    col = [0.2 + 0.3 * i, 0.2 + 0.3 * i, 0.2 + 0.3 * i]
                if pop_key == "cont":
                    col = [0.2 + 0.3 * i, 0.2 + 0.3 * i, 1]
                    # idxes = compare['inter_coding'][idx_ds,1]==1
                if pop_key == "discont":
                    col = [1, 0.2 + 0.3 * i, 0.2 + 0.3 * i]
                    # idxes = (compare['inter_coding'][idx_ds,1]==0) & (compare['inter_active'][idx_ds,1]==1)
                if pop_key == "silent":
                    col = [0.2 + 0.3 * i, 1, 0.2 + 0.3 * i]
                    # idxes = compare['inter_active'][idx_ds,1]==0

                plt.plot(
                    iterate_var[1],
                    stability[pop_key]["mean"][:, ds, 1],
                    color=col,
                    label=pop_key if i == 0 else None,
                )
            # plt.errorbar(val_arr[:-1],stability[pop_key]['mean'][:,ds][:-1],stability[pop_key]['std'][:,ds,1][:-1],fmt='none',ecolor='r')

        plt.xlabel(iterate_var[0], fontsize=14)
        plt.legend(fontsize=10)
        plt.ylim([0, 1.0])
        plt.ylabel("$r_{stable}$", fontsize=14)
        plt.tight_layout()
        plt.show(block=False)

        return stability

        # if sv:
        #     pl_dat.save_fig('stability_impact_pm')
        ##for i in range(nSteps):
        ##col = np.ones(3)*0.2*i
        ##plt.bar(np.arange(1,sesMax+1)-0.1*nSteps+0.2*i,self.stability['cont']['mean'][i,:sesMax,1],width=0.2,facecolor=col,label='continuous')
        ##plt.errorbar(np.arange(1,sesMax+1)-0.1*nSteps+0.2*i,self.stability['cont']['mean'][i,:sesMax,1],self.stability['cont']['std'][i,:sesMax,1],fmt='none',ecolor='r')
        ##plt.show(block=False)
        # return cluster

    def plot_PC_choice(self, sv=False):

        s = 10
        idx_PCs = self.status["activity"][:, :, 2]
        idx_fields = np.where(self.status["fields"])
        plt.figure(figsize=(4, 2.5))
        plt.scatter(
            self.stats["MI_p_value"][idx_fields[0], idx_fields[1]],
            self.fields["Bayes_factor"][self.status["fields"]],
            color="r",
            s=5,
        )

        idx_nfields = np.where(~self.status["fields"])
        plt.scatter(
            self.stats["MI_p_value"][idx_nfields[0], idx_nfields[1]],
            self.fields["Bayes_factor"][~self.status["fields"]],
            color=[0.6, 0.6, 0.6],
            s=3,
        )
        plt.xlabel("p-value (mutual information)", fontsize=14)
        plt.ylabel("log($Z_{PC}$) - log($Z_{nPC}$)", fontsize=14)
        plt.ylim([-10, 200])
        plt.tight_layout()
        plt.show(block=False)
        if sv:
            self.pl_dat.save_fig("PC_choice_s=%d" % s)

    def plot_neuron_movement(self):

        nDisp = 500
        plt.figure()
        ax = plt.subplot(111)  # ,projection='3d')

        for n in range(nDisp):
            sc = ax.plot(
                self.matching["com"][n, ::5, 0], self.matching["com"][n, ::5, 1], "k-"
            )  # ,c=range(0,nSes,5))#,cmap='jet')#,self.matching['com'][n,::5,2]
        # plt.colorbar(sc)

        plt.show(block=False)

    def plot_network_dynamics(self):
        print("### plot dynamics of whole network ###")

        plt.figure()
        for i, s in enumerate(range(0, 10)):

            if self.paths["neuron_detection"][s].exists():
                ld = load_data(self.paths["neuron_detection"][s])
                n_arr = self.matching["IDs"][
                    self.status["activity"][:, s, 1], s
                ].astype("int")

                # print(n_arr)

                if n_arr.size > 0:

                    S = ld["S"][n_arr, :]
                    _, _, S_thr = get_firingrate(S)

                    S_mean = gauss_smooth(S_thr.mean(0), 5)

                    S_ft = np.fft.fft(S_mean)
                    # print(S.shape)
                    frequencies = np.arange(8989 // 2) / 600
                    plt.subplot(5, 2, i + 1)
                    plt.plot(frequencies, S_ft[: 8989 // 2])
                    plt.ylim([0, 50])
        plt.show(block=False)

    def plot_coding_stats(self, sv=False):

        ### get place field max firing rate
        # for c in range(self.params['nC']):
        # for s in range(self.data['nSes']):

        print("test field width as well")
        print("test peak firing rate as well")

        nSes = self.data["nSes"]

        s_bool = np.zeros(nSes, "bool")
        # s_bool[17:87] = True
        s_bool[0:15] = True

        s1, s2, f1, f2 = np.unravel_index(
            self.compare["pointer"].col,
            (
                self.data["nSes"],
                self.data["nSes"],
                self.params["field_count_max"],
                self.params["field_count_max"],
            ),
        )
        idx_ds1 = np.where((s2 - s1 == 1) & s_bool[s1] & s_bool[s2])[0]

        c_ds1 = self.compare["pointer"].row[idx_ds1]
        s1_ds1 = s1[idx_ds1]
        f1_ds1 = f1[idx_ds1]
        idx_shifts_ds1 = self.compare["pointer"].data[idx_ds1].astype("int") - 1
        shifts_ds1 = self.compare["shifts"][idx_shifts_ds1]

        idx_stable_ds1 = np.where(np.abs(shifts_ds1) < 6)[0]
        idx_relocate_ds1 = np.where(np.abs(shifts_ds1) > 12)[0]

        c_stable = c_ds1[idx_stable_ds1]
        s1_stable = s1_ds1[idx_stable_ds1]
        f_stable = f1_ds1[idx_stable_ds1]
        rel_stable = self.fields["reliability"][c_stable, s1_stable, f_stable]
        Isec_stable = self.stats["Isec_value"][c_stable, s1_stable]
        fr_stable = self.stats["firingrate"][c_stable, s1_stable]

        c_relocate = c_ds1[idx_relocate_ds1]
        s1_relocate = s1_ds1[idx_relocate_ds1]
        f_relocate = f1_ds1[idx_relocate_ds1]
        Isec_relocate = self.stats["Isec_value"][c_relocate, s1_relocate]
        fr_relocate = self.stats["firingrate"][c_relocate, s1_relocate]

        idx_loosePC = np.where(
            np.diff(self.status["activity"][..., 2].astype("int"), 1) == -1
        )
        Isec_instable = self.stats["Isec_value"][idx_loosePC]
        fr_instable = self.stats["firingrate"][idx_loosePC]

        idx_nPC = np.where(
            self.status["activity"][..., 1] & ~self.status["activity"][..., 2]
        )
        # rel_instable = np.nanmax(self.fields['reliability'][idx_loosePC[0],idx_loosePC[1],:],-1)
        Isec_nPC = self.stats["Isec_value"][idx_nPC]
        fr_nPC = self.stats["firingrate"][idx_nPC]

        col_stable = [0, 0.5, 0]
        plt.figure(figsize=(7, 2.5))
        ax = plt.subplot(142)
        rel_relocate = self.fields["reliability"][c_relocate, s1_relocate, f_relocate]
        rel_instable = np.nanmax(
            self.fields["reliability"][idx_loosePC[0], idx_loosePC[1], :], -1
        )
        ax.hist(
            rel_stable,
            np.linspace(0, 1, 51),
            alpha=0.5,
            density=True,
            cumulative=True,
            histtype="step",
            color=col_stable,
        )
        ax.hist(
            rel_relocate,
            np.linspace(0, 1, 51),
            alpha=0.5,
            density=True,
            cumulative=True,
            histtype="step",
            color="b",
        )
        ax.hist(
            rel_instable,
            np.linspace(0, 1, 51),
            alpha=0.5,
            density=True,
            cumulative=True,
            histtype="step",
            color="r",
        )
        # plt.hist(rel_nPC,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k',linestyle=':')
        # rel_all = self.fields['reliability']
        # rel_all[~self.status["fields"]] = np.NaN
        # rel_all = rel_all[self.status['activity'][...,2],...]
        # ax.hist(rel_all.flat,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k')
        ax.set_xlabel("reliability [%]", fontsize=14)
        ax.set_xlim([0, 1])
        ax.set_yticks([])

        ax = plt.subplot(143)
        MI_nPC = self.stats["MI_value"][idx_nPC]
        MI_stable = self.stats["MI_value"][c_stable, s1_stable]
        MI_instable = self.stats["MI_value"][idx_loosePC]
        MI_relocate = self.stats["MI_value"][c_relocate, s1_relocate]

        # MI_nPC = self.stats['Isec_value'][idx_nPC]
        # MI_stable = self.stats['Isec_value'][c_stable,s1_stable]
        # MI_instable = self.stats['Isec_value'][idx_loosePC]
        # MI_relocate = self.stats['Isec_value'][c_relocate,s1_relocate]

        plt.hist(
            MI_nPC,
            np.linspace(0, 1, 51),
            alpha=0.5,
            density=True,
            cumulative=True,
            histtype="step",
            color="k",
            linestyle=":",
            label="nPC",
        )
        plt.hist(
            MI_stable,
            np.linspace(0, 1, 51),
            alpha=0.5,
            density=True,
            cumulative=True,
            histtype="step",
            color=col_stable,
            label="stable",
        )
        plt.hist(
            MI_instable,
            np.linspace(0, 1, 51),
            alpha=0.5,
            density=True,
            cumulative=True,
            histtype="step",
            color="r",
            label="instable",
        )
        ax.set_xlabel("MI [bits]", fontsize=14)
        ax.set_xlim([0, 1])
        ax.set_yticks([])
        ax.legend(fontsize=10, loc="lower right")

        ax = plt.subplot(144)
        # key = 'oof_firingrate_adapt'
        key = "firingrate"
        nu_nPC = self.stats[key][idx_nPC]
        nu_stable = self.stats[key][c_stable, s1_stable]
        nu_instable = self.stats[key][idx_loosePC]
        nu_relocate = self.stats[key][c_relocate, s1_relocate]
        plt.hist(
            nu_nPC,
            np.linspace(0, 2, 51),
            alpha=0.5,
            density=True,
            cumulative=True,
            histtype="step",
            color="k",
            linestyle=":",
            label="nPC",
        )
        plt.hist(
            nu_stable,
            np.linspace(0, 2, 51),
            alpha=0.5,
            density=True,
            cumulative=True,
            histtype="step",
            color=col_stable,
            label="stable",
        )
        plt.hist(
            nu_instable,
            np.linspace(0, 2, 51),
            alpha=0.5,
            density=True,
            cumulative=True,
            histtype="step",
            color="r",
            label="instable",
        )
        ax.set_xlabel("$\\nu$ [Hz]")
        ax.set_xlim([0, 1.2])
        ax.set_yticks([])
        ax.legend(fontsize=10, loc="lower right")

        # ax = plt.subplot(133)
        # maxrate_stable = self.fields['max_rate'][c_stable,s1_stable,f_stable]
        ##idx_loosePC = np.where(np.diff(self.status['activity'][...,2].astype('int'),1)==-1)
        # maxrate_instable = np.nanmax(self.fields['max_rate'][idx_loosePC[0],idx_loosePC[1],:],-1)
        # plt.hist(maxrate_stable,np.linspace(0,20,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable,label='stable')
        # plt.hist(maxrate_instable,np.linspace(0,20,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r',label='instable')
        # ax.set_xlabel('$\\nu_{max}$',fontsize=14)
        # ax.set_xlim([0,20])

        ax = plt.subplot(141)
        width_stable = self.fields["width"][c_stable, s1_stable, f_stable, 0]
        # idx_loosePC = np.where(np.diff(self.status['activity'][...,2].astype('int'),1)==-1)
        width_instable = np.nanmax(
            self.fields["width"][idx_loosePC[0], idx_loosePC[1], :, 0], -1
        )
        plt.hist(
            width_stable,
            np.linspace(0, 10, 51),
            alpha=0.5,
            density=True,
            cumulative=True,
            histtype="step",
            color=col_stable,
            label="stable",
        )
        plt.hist(
            width_instable,
            np.linspace(0, 10, 51),
            alpha=0.5,
            density=True,
            cumulative=True,
            histtype="step",
            color="r",
            label="instable",
        )
        ax.set_xlabel("$\sigma$ [bins]", fontsize=14)
        ax.set_xlim([0, 10])
        ax.set_yticks(np.linspace(0, 1, 3))
        ax.set_ylabel("cdf", fontsize=14)

        # ax = plt.subplot(144)
        # A_stable = self.fields['amplitude'][c_stable,s1_stable,f_stable,0]
        # #idx_loosePC = np.where(np.diff(self.status['activity'][...,2].astype('int'),1)==-1)
        # A_instable = np.nanmax(self.fields['amplitude'][idx_loosePC[0],idx_loosePC[1],:,0],-1)
        # plt.hist(A_stable,np.linspace(0,40,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable,label='stable')
        # plt.hist(A_instable,np.linspace(0,40,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r',label='instable')
        # ax.set_xlabel('$\sigma$ [bins]',fontsize=14)
        # ax.set_xlim([0,40])
        # ax.set_yticks(np.linspace(0,1,3))
        # ax.set_ylabel('cdf',fontsize=14)

        # plt.subplot(224)
        # plt.hist(Isec_stable,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable)
        # plt.hist(Isec_stable,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r')
        # MI_all = self.stats['Isec_value']
        # MI_all = MI_all[self.status['activity'][...,2]]
        # plt.hist(MI_all.flat,np.linspace(0,1,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k')
        # plt.xlabel('I/sec')

        # ax = plt.subplot(131)
        ##a,b = bootstrap_data(lambda x : (np.cumsum(np.histogram(x,np.linspace(0,3,51))[0])/len(x),np.NaN),fr_stable,1000)
        ##pl_dat.plot_with_confidence(ax,np.linspace(0,3,51)[:-1],a,b,col='k',ls='-')
        # plt.hist(fr_nPC,np.linspace(0,5,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k',linestyle=':',label='nPC')
        # plt.hist(fr_stable,np.linspace(0,5,51),alpha=0.5,density=True,cumulative=True,histtype='step',color=col_stable,label='stable')
        ##plt.hist(fr_relocate,np.linspace(0,3,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='b')
        # plt.hist(fr_instable,np.linspace(0,5,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='r',label='instable')

        # fr_all = self.stats['firingrate']
        # fr_all = fr_all[self.status['activity'][...,2]]
        ##a,b = bootstrap_data(lambda x : (np.cumsum(np.histogram(x,np.linspace(0,3,51))[0])/len(x),np.NaN),fr_all,1000)
        ##pl_dat.plot_with_confidence(ax,np.linspace(0,3,51)[:-1],a,b,col='r',ls='-')
        ##ax.hist(fr_all.flat,np.linspace(0,3,51),alpha=0.5,density=True,cumulative=True,histtype='step',color='k')
        # ax.set_xlabel('activity [Hz]',fontsize=14)
        # ax.set_ylabel('cdf',fontsize=14)
        # ax.set_xlim([0,5])

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig("codingChange_stats")

    def plot_neuron_remapping(self):

        nSes = self.data["nSes"]
        nbin = self.data["nbin"]

        steps = 100

        fig = plt.figure()
        ax = plt.subplot(111)

        fields = np.zeros((steps, nSes))
        idx_PCs = np.where(
            np.logical_and(
                self.status["activity"][:, :, 2].sum(axis=1) > 5,
                self.status["activity"][:, :, 1].sum(axis=1) > 15,
            )
        )[0]

        print(idx_PCs)
        idx_PC = np.random.choice(idx_PCs)
        print(idx_PC)
        for i, s in enumerate(np.where(self.status["sessions"])[0]):
            # idx_PC = np.where(self.status["fields"][:, s, :] >= 3)
            # fields[s,:] = np.nansum(self.fields['p_x'][:,s,:,:],1).sum(0)
            # fields[:, s] = np.nansum(self.fields["p_x"][idx_PC, s, 0, :], 0)
            fields[:, s] = self.fields["p_x"][idx_PC, s, 0, :]
        fields /= fields.max(0)
        # fields[:,s] /= fields[:,s].sum()
        # print(fields)
        fields = gauss_smooth(fields, (1, 0))
        im = ax.imshow(fields, origin="lower", cmap="jet")  # ,clim=[0,1])
        plt.colorbar(im)
        ax.set_xlabel("session")
        ax.set_ylabel("position [bins]")

        plt.show(block=False)

        # s = 10
        ds = 1
        block_size = 2
        fig = plt.figure(figsize=(8, 6), dpi=self.pl_dat.sv_opt["dpi"])
        # ax = fig.add_subplot(111)
        for s in np.where(self.status["sessions"])[0][:-ds]:  # range(5,15):
            if (s % block_size) == 0:
                if (s // block_size) > 0:

                    ax = plt.subplot(3, 4, s // block_size)
                    remapping /= remapping.max() / 2
                    ax.imshow(remapping, origin="lower", clim=[0, 1], cmap="hot")
                    ax.text(
                        5,
                        90,
                        "Sessions %d-%d" % (s - block_size, s),
                        color="w",
                        fontsize=8,
                    )
                    plt.setp(ax, ylim=[0, nbin])
                    # plt.colorbar()
                remapping = np.zeros((steps, steps))

            for c in np.where(self.status["clusters"])[0]:
                if (
                    self.status["activity"][c, s, 2]
                    & self.status["activity"][c, s + ds, 2]
                ):
                    for f in np.where(
                        np.isfinite(self.fields["amplitude"][c, s, :, 0])
                    )[0]:
                        for ff in np.where(
                            np.isfinite(self.fields["amplitude"][c, s + ds, :])
                        )[0]:
                            remapping[
                                int(self.fields["location"][c, s, f, 0]), :
                            ] += self.fields["p_x"][c, s + ds, ff, :]
        plt.show(block=False)
        # print(remapping.sum(1))

    # print(np.where(self.compare['inter_coding'][:,1]==0)[0])
    # print('search for cases, where the neuron loses its  coding ability -> lower MI / lower fr / ...?')

    def plot_neuron_examples(self, sv=False):
        print("### SNR & CNN examples ###")
        if True:
            plt.figure(figsize=(7, 5), dpi=self.pl_dat.sv_opt["dpi"])

            nSteps = 11
            SNR_arr = np.linspace(1, 11, nSteps)

            margin = 18

            ax = plt.axes([0.1, 0.1, 0.45, 0.85])
            t_arr = np.linspace(0, 8989 / 15, 8989)

            s = 1

            self.paths["neuron_detection"][s + 1]
            if self.paths["neuron_detection"][s + 1].exists():
                ld = load_data(self.paths["neuron_detection"][s + 1])
                # ld = loadmat(pathLoad,variable_names=['C','A','SNR','CNN'],squeeze_me=True)

                offset = 0
                for i in tqdm(range(nSteps - 1)):
                    # idx_SNR = np.where((self.stats['SNR'][:,s] >= SNR_arr[i]) & (self.stats['SNR'][:,s] < SNR_arr[i+1]))
                    idx_SNR = np.where(
                        (ld["SNR_comp"] >= SNR_arr[i])
                        & (ld["SNR_comp"] < SNR_arr[i + 1])
                    )
                    n_idx = len(idx_SNR[0])
                    if n_idx > 0:
                        for j in np.random.choice(n_idx, min(n_idx, 3), replace=False):
                            # c = idx_SNR[0][j]
                            # n = int(self.matching['IDs'][c,s])
                            n = idx_SNR[0][j]
                            C = ld["C"][n, :] / ld["C"][n, :].max()
                            ax.plot(t_arr, -C + offset, linewidth=0.5)
                            # ax.text(600,offset,'%.2f'%self.stats['SNR'][c,s])#'[%f,%f]'%(SNR_arr[i],SNR_arr[i+1]))
                            offset += 1  # = (nSteps-i)

                    offset += 1
                ax.set_yticks(np.linspace(1, offset - 3, nSteps))
                ax.set_yticklabels(["$\\approx %d$" % i for i in SNR_arr])
                ax.set_ylabel("SNR", rotation="horizontal", labelpad=-20, y=1.0)
                ax.set_xlabel("time [s]")
                ax.set_ylim([offset - 1, -1])
                ax.set_xlim([0, 600])
                self.pl_dat.remove_frame(ax, ["top", "right"])

                nSteps = 9
                CNN_arr = np.linspace(0.0, 1.0, nSteps)
                acom = com(ld["A"], 512, 512)
                for i in tqdm(range(nSteps - 1)):
                    # idx_CNN = np.where((self.stats['CNN'][:,s] >= CNN_arr[i]) & (self.stats['CNN'][:,s] < CNN_arr[i+1]))
                    idx_CNN = np.where(
                        (ld["cnn_preds"] >= CNN_arr[i])
                        & (ld["cnn_preds"] < CNN_arr[i + 1])
                        & ((ld["A"] > 0).sum(0) > 50)
                        & np.all(acom > 10, 1)
                        & np.all(acom < 500, 1)
                    )
                    n_idx = len(idx_CNN[0])
                    # print(idx_CNN)
                    if n_idx > 0:
                        for j in np.random.choice(n_idx, min(n_idx, 1), replace=False):
                            # c = idx_CNN[0][j]
                            # n = int(self.matching['IDs'][c,s])
                            n = idx_CNN[1][j]
                            A = ld["A"][:, n].reshape(512, 512).toarray()
                            a_com = com(A.reshape(-1, 1), 512, 512)
                            ax = plt.axes(
                                [
                                    0.6 + (i // (nSteps // 2)) * 0.175,
                                    0.75 - (i % (nSteps // 2)) * 0.23,
                                    0.15,
                                    0.21,
                                ]
                            )
                            if i == (nSteps - 2):
                                sbar = ScaleBar(
                                    530.68 / 512 * 10 ** (-6), location="lower right"
                                )
                                ax.add_artist(sbar)
                            A /= A.max()
                            A[A < 0.001] = np.NaN
                            ax.imshow(A, cmap="viridis", origin="lower")
                            ax.contour(
                                A,
                                levels=[0.3, 0.6, 0.9],
                                colors="w",
                                linewidths=[0.5],
                                linestyles=["dotted", "dashed", "solid"],
                            )

                            (x_ref, y_ref) = a_com[0]  # print(x_ref,y_ref)
                            x_lims = [x_ref - margin, x_ref + margin]
                            y_lims = [y_ref - margin, y_ref + margin]
                            # ax.plot(t_arr,C+nSteps-offset)
                            # ax.text(600,nSteps-offset,'%.2f'%self.stats['SNR'][c,s])#'[%f,%f]'%(SNR_arr[i],SNR_arr[i+1]))
                            ax.set_xlim(x_lims)
                            ax.set_ylim(y_lims)
                            # ax.text(x_ref,y_ref+5,'$CNN = %.3f$'%self.stats['CNN'][c,s],fontsize=8)
                            ax.text(
                                x_ref + 2,
                                y_ref + 12,
                                "$%.3f$" % ld["cnn_preds"][n],
                                fontsize=8,
                            )
                            self.pl_dat.remove_frame(ax)
                            ax.set_xticks([])
                            ax.set_yticks([])
                plt.tight_layout()
                plt.show(block=False)

            if sv:
                self.pl_dat.save_fig("neuron_stat_examples")

        if True:
            s = 1
            margin = 20
            nSteps = 9
            pathLoad = self.paths["neuron_detection"][s]
            if os.path.exists(pathLoad):
                ld1 = load_data(pathLoad)

                pathLoad = self.paths["neuron_detection"][s + 1]
                if os.path.exists(pathLoad):
                    ld2 = load_data(pathLoad)

                    # pathLoad = pathcat([self.params['pathMouse'],'Session%02d/results_redetect.mat'%(s)])
                    # ld1 = loadmat(pathLoad,variable_names=['A'])
                    # pathLoad = pathcat([self.params['pathMouse'],'Session%02d/results_redetect.mat'%(s+1)])
                    # ld2 = loadmat(pathLoad,variable_names=['A'])

                    x_grid, y_grid = np.meshgrid(
                        np.arange(0.0, self.params["dims"][0]).astype(np.float32),
                        np.arange(0.0, self.params["dims"][1]).astype(np.float32),
                    )
                    x_remap = (
                        x_grid
                        - self.alignment["shift"][s - 1, 0]
                        + self.alignment["shift"][s, 0]
                        + self.alignment["flow"][s - 1, 0, :, :]
                        - self.alignment["flow"][s, 0, :, :]
                    ).astype("float32")
                    y_remap = (
                        y_grid
                        - self.alignment["shift"][s - 1, 1]
                        + self.alignment["shift"][s, 1]
                        + self.alignment["flow"][s - 1, 1, :, :]
                        - self.alignment["flow"][s, 1, :, :]
                    ).astype("float32")

                    plt.figure(figsize=(2, 4), dpi=self.pl_dat.sv_opt["dpi"])
                    p_arr = np.linspace(0, 1, nSteps)
                    for i in tqdm(range(nSteps - 1)):
                        idx_p = np.where(
                            (self.matching["score"][:, s, 0] >= p_arr[i])
                            & (self.matching["score"][:, s, 0] < p_arr[i + 1])
                            & (self.status["activity"][:, s - 1, 1])
                        )
                        n_idx = len(idx_p[0])
                        if n_idx > 0:
                            c = np.random.choice(idx_p[0])
                            # s = idx_SNR[1][j]
                            n1 = int(self.matching["IDs"][c, s - 1])
                            n2 = int(self.matching["IDs"][c, s])

                            ax = plt.axes(
                                [
                                    0.05 + (i // (nSteps // 2)) * 0.45,
                                    0.75 - (i % (nSteps // 2)) * 0.23,
                                    0.4,
                                    0.2,
                                ]
                            )
                            # ax = plt.axes([0.7,0.8-0.2*]])
                            # for j in np.random.choice(n_idx,min(n_idx,3),replace=False):
                            # offset += 1#= (nSteps-i)
                            A1 = ld1["A"][:, n1].reshape(512, 512).toarray()
                            A1 = cv2.remap(A1, x_remap, y_remap, cv2.INTER_CUBIC)
                            A2 = ld2["A"][:, n2].reshape(512, 512).toarray()

                            a_com = com(A2.reshape(-1, 1), 512, 512)

                            ax.contour(
                                A1 / A1.max(),
                                levels=[0.3, 0.6, 0.9],
                                colors="k",
                                linewidths=[0.5],
                                linestyles=["dotted", "dashed", "solid"],
                            )
                            ax.contour(
                                A2 / A2.max(),
                                levels=[0.3, 0.6, 0.9],
                                colors="r",
                                linewidths=[0.5],
                                linestyles=["dotted", "dashed", "solid"],
                            )
                            if i == (nSteps - 2):
                                sbar = ScaleBar(
                                    530.68 / 512 * 10 ** (-6),
                                    location="lower right",
                                    box_alpha=0,
                                )
                                ax.add_artist(sbar)

                            (x_ref, y_ref) = a_com[0]  # print(x_ref,y_ref)
                            x_lims = [x_ref - margin, x_ref + margin]
                            y_lims = [y_ref - margin, y_ref + margin]
                            # ax.plot(t_arr,C+nSteps-offset)
                            # ax.text(600,nSteps-offset,'%.2f'%self.stats['SNR'][c,s])#'[%f,%f]'%(SNR_arr[i],SNR_arr[i+1]))
                            ax.set_xlim(x_lims)
                            ax.set_ylim(y_lims)
                            ax.text(
                                x_ref + 2,
                                y_ref + 8,
                                "$%.2f$" % self.matching["score"][c, s, 0],
                                fontsize=8,
                            )
                            self.pl_dat.remove_frame(ax)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            # offset += 1
                    plt.tight_layout()
                    plt.show(block=False)

            if sv:
                self.pl_dat.save_fig("neuron_matches_examples")

    def plot_pv_correlations(self):
        print("## plot population vector correlations etc")

        nSes = self.data["nSes"]
        nC = self.data["nC"]
        nbin = self.data["nbin"]

        def nangauss_filter(array, sigma, truncate):

            V = array.copy()
            V[np.isnan(array)] = 0
            VV = sp.ndimage.gaussian_filter(V, sigma=sigma, truncate=truncate)

            W = 0 * array.copy() + 1
            W[np.isnan(array)] = 0
            WW = sp.ndimage.gaussian_filter(W, sigma=sigma, truncate=truncate)

            return VV / WW

        fmap = nangauss_filter(
            self.stats["firingmap"][self.status["clusters"], :, :], (0, 0, 2), 5
        )
        fmap = np.ma.masked_invalid(fmap)

        # fmap = sp.ndimage.gaussian_filter(fmap,2)

        ## plotting population vector correlation of different ds, calculated for each session
        di = 5

        plt.figure(figsize=(3, 5))
        ax = plt.axes([0.1, 0.6, 0.85, 0.35])
        ax_corr = plt.axes([0.1, 0.15, 0.85, 0.35])

        ds_arr = [1, 2, 3, 5, 10]
        for s0, ds in enumerate(ds_arr):

            col = s0 / len(ds_arr)
            col = [col] * 3

            session_bool = np.where(
                np.pad(self.status["sessions"][ds:], (0, ds), constant_values=False)
                & np.pad(self.status["sessions"][:], (0, 0), constant_values=False)
            )[0]
            s_corr = np.zeros(nSes) * np.NaN
            for s in tqdm(np.where(session_bool)[0]):
                corr = np.zeros(nbin)
                for i in range(nbin):

                    idx = np.zeros(nbin, "bool")
                    idx[max(0, i - di) : min(nbin + 1, i + di)] = True

                    idx_cells = self.status["activity"][self.status["clusters"], s, 1]
                    corr[i] = np.ma.corrcoef(
                        fmap[idx_cells, s, :][:, idx].mean(-1),
                        fmap[idx_cells, s + ds, :][:, idx].mean(-1),
                    )[0, 1]

                # print(corr)
                if s in [10, 20, 40, 60]:
                    ax.plot(corr, color=col)

                s_corr[s] = corr.mean()

            ax_corr.plot(
                gauss_smooth(s_corr, 1, mode="constant"), color=col, label="ds=%d" % ds
            )

        ax.set_ylim([-0.25, 1.0])
        ax_corr.set_ylim([-0.25, 1.0])
        ax_corr.legend()
        # plt.title('ds=%d'%ds)
        plt.show(block=False)

        ## plotting ds-dependence of population vector correlation
        # fmap = gauss_smooth(self.stats['firingmap'],(0,0,2))
        corr = np.zeros((nC, nSes, nSes)) * np.NaN
        for ds in tqdm(range(1, min(nSes - 1, 30))):
            session_bool = np.where(
                np.pad(self.status["sessions"][ds:], (0, ds), constant_values=False)
                & np.pad(self.status["sessions"][:], (0, 0), constant_values=False)
            )[0]
            for s in np.where(session_bool)[0]:
                # for n
                for n in np.where(
                    self.status["activity"][self.status["clusters"], s, 1]
                    & self.status["activity"][self.status["clusters"], s + ds, 1]
                )[0]:
                    corr[n, s, ds] = np.corrcoef(fmap[n, s, :], fmap[n, s + ds, :])[
                        0, 1
                    ]

        plt.figure(figsize=(7, 5), dpi=self.pl_dat.sv_opt["dpi"])
        plt.subplot(121)
        im = plt.imshow(np.nanmean(corr, 0), clim=[0, 0.5])
        plt.colorbar(im)

        plt.subplot(122)
        plt.plot(np.nanmean(np.nanmean(corr, 0), 0))
        plt.ylim([0, 1])
        plt.show(block=False)

    def plot_example_draw(self, sv=False):

        plt.figure(figsize=(3, 2))

        nC1 = 3000
        L = 1000
        K_act1 = 1200
        K_act2 = 1100
        rand_pull_act = (np.random.choice(nC1, (L, K_act1)) < K_act2).sum(1)
        plt.hist(
            rand_pull_act,
            np.linspace(0, 800, 101),
            facecolor="k",
            density=True,
            label="random draws",
        )
        plt.plot([700, 700], [0, 0.2], "r", label="actual value")
        plt.ylim([0, 0.05])
        plt.xlabel("# same activated neurons", fontsize=14)
        plt.yticks([])
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig("example_draw")

    def plot_corr_pairs(self):

        print("### plot within session correlated pairs ###")

        print("neurons whos place field is activated conjointly")
        print("kinda difficult - cant find anything obvious on first sight")

        nC = self.data["nC"]

        plt.figure(figsize=(7, 5), dpi=self.pl_dat.sv_opt["dpi"])
        ## find trials

        self.fields["trial_act"]

        high_corr = np.zeros(nC)
        for i, s in enumerate(range(0, 20), 1):
            idx = np.where(self.status["fields"])

            idx_s = idx[1] == s
            c = idx[0][idx_s]
            f = idx[2][idx_s]
            trials = self.fields["trial_act"][c, s, f, : self.behavior["trial_ct"][s]]

            trial_corr = np.corrcoef(trials)
            trial_corr[np.tril_indices_from(trial_corr)] = np.NaN

            # idx[0][idx_s]# np.fill_diagonal(trial_corr,np.NaN)
            # print(trial_corr)
            # print(trial_corr.shape)
            idx_high_corr = np.where(trial_corr > 0.5)
            # print(self.stats.keys())
            for c1, c2 in zip(idx_high_corr[0], idx_high_corr[1]):
                # print('c: %d, %d'%(c1,c2))
                # print(self.matching['com'][c1,s,:])
                # print(self.matching['com'][c2,s,:])
                # print(np.linalg.norm(self.matching['com'][c1,s,:]-self.matching['com'][c2,s,:]))
                # if np.linalg.norm(self.matching['com'][c1,s,:]-self.matching['com'][c2,s,:])>10:
                high_corr[c1] += 1
                high_corr[c2] += 1

            # plt.subplot(5,4,i)
            # plt.hist(trial_corr.flat,np.linspace(-1,1,51))
        high_corr[high_corr == 0] = np.NaN
        plt.hist(high_corr, np.linspace(0, 400, 51))
        plt.show(block=False)

    def plot_sdep_stability(self, reprocess=False, sv=False):

        print("get session-dependent stability")

        nSes = self.data["nSes"]

        s_bool = np.ones(nSes, "bool")
        # s_bool[17:87] = True
        # s_bool[:] = True
        s_bool[~self.status["sessions"]] = False

        xlim = np.where(s_bool)[0][-1] - np.where(s_bool)[0][0] + 1

        act_stab_thr = [0.1, 0.9]
        r_stab_thr = [0.1, 0.5]

        ds = 2
        if (not ("act_stability_temp" in self.stats.keys())) | reprocess:
            self.stats["act_stability_temp"] = self.get_act_stability_temp(ds=ds)
        if (not ("act_stability" in self.stats.keys())) | reprocess:
            self.stats["act_stability"] = self.get_act_stability(s_bool)
        if (not ("field_stability_temp" in self.stats.keys())) | reprocess:
            self.stats["field_stability_temp"] = self.get_field_stability_temp(
                SD=1.96, ds=ds
            )

        if not ("field_stability" in self.stats.keys()):
            self.stats["field_stability"] = self.get_field_stability(SD=1.96)

        # act_clusters = self.status['clusters']
        act_clusters = np.any(self.status["activity"][:, s_bool, 1], 1)
        r_stab = gauss_smooth(
            self.stats["field_stability_temp"], (0, 1)
        )  # [act_clusters,:]
        # act_stab = self.stats['act_stability_temp'][self.status['clusters'],:,1]
        # act_stab = act_stab[self.status['clusters'],:]
        act_stab = self.stats["act_stability_temp"][
            ..., 1
        ]  # [self.status['clusters'],:,1]

        nC = self.status["activity"].shape[0]  # self.status['clusters'].sum()
        nSes_good = s_bool.sum()

        status = self.status["activity"][..., 1]  # [self.status['clusters'],:,1]
        status_dep = None

        dp_pos, p_pos = get_dp(
            status, status_dep=status_dep, status_session=s_bool, ds=1
        )

        dp_pos_temp = np.zeros((nC, nSes)) * np.NaN
        p_pos_temp = np.zeros((nC, nSes)) * np.NaN
        t_start = time.time()
        for s in range(nSes):  # np.where(s_bool)[0]:
            s_bool_tmp = np.copy(s_bool)
            s_bool_tmp = np.copy(self.status["sessions"])
            s_bool_tmp[:s] = False
            s_bool_tmp[s + ds :] = False
            dp_pos_temp[:, s], p_pos_temp[:, s] = get_dp(
                status, status_dep=status_dep, status_session=s_bool_tmp, ds=1
            )

        # act_stab = p_pos_temp
        # p_pos_temp = act_stab

        fig = plt.figure(figsize=(7, 5), dpi=300)

        locmin = LogLocator(base=10.0, subs=(0, 1), numticks=8)
        locmaj = LogLocator(base=100.0, numticks=8)

        ax = plt.axes([0.1, 0.85, 0.125, 0.08])
        self.pl_dat.add_number(fig, ax, order=1)
        ax.hist(
            p_pos[act_clusters],
            np.linspace(0, 1.0, 21),
            color="k",
            label="$r^{\infty}_{\\alpha^+}$",
        )
        ax.set_xlabel("$r^{\infty}_{\\alpha^+}$")
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax.set_yscale("log")
        ax.set_ylim([0.7, 3000])
        ax.set_ylabel("count")
        # ax.set_yticks([0,1000])
        ax.yaxis.set_major_locator(locmaj)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax = plt.axes([0.3, 0.85, 0.125, 0.08])
        ax.hist(
            np.nanmax(act_stab[act_clusters, :][:, s_bool], 1),
            np.linspace(0, 1, 21),
            color="k",
            label="$max(r^%d_{\\alpha^+})$" % ds,
        )
        ax.set_yscale("log")
        ax.set_ylim([0.7, 3000])
        ylim = ax.get_ylim()[1]
        ax.plot(
            [act_stab_thr[0], act_stab_thr[0]],
            [1, ylim],
            "--",
            color="tab:blue",
            linewidth=0.75,
        )
        ax.plot(
            [act_stab_thr[1], act_stab_thr[1]],
            [1, ylim],
            "--",
            color="tab:red",
            linewidth=0.75,
        )
        ax.text(x=act_stab_thr[0] - 0.05, y=ylim * 1.3, s="low", fontsize=6)
        ax.text(x=act_stab_thr[1] - 0.05, y=ylim * 1.3, s="high", fontsize=6)
        ax.set_xlabel("$max_s(r^%d_{\\alpha^+})$" % ds)
        # ax.legend(fontsize=8,loc='upper right')
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax.yaxis.set_major_locator(locmaj)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax = plt.axes([0.6, 0.85, 0.125, 0.08])
        self.pl_dat.add_number(fig, ax, order=5)
        ax.hist(
            self.stats["field_stability"][act_clusters],
            np.linspace(0, 1.0, 21),
            color="k",
            label="$r^{\infty}_{\\alpha}$",
        )
        ax.set_xlabel("$r^{\infty}_{\\gamma^+}$")
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax.set_ylabel("count")
        ax.set_yscale("log")
        ax.set_ylim([0.7, 3000])
        ax.yaxis.set_major_locator(locmaj)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax = plt.axes([0.8, 0.85, 0.125, 0.08])
        ax.hist(
            np.nanmax(r_stab[act_clusters, :][:, s_bool], 1),
            np.linspace(0, 1.0, 21),
            color="k",
            label="$max(r^%d_{\\gamma^+})$" % ds,
        )
        print(
            np.histogram(
                np.nanmax(r_stab[act_clusters, :][:, s_bool], 1),
                np.linspace(0, 1.0, 21),
            )
        )
        ax.set_yscale("log")
        ax.set_ylim([0.7, 3000])
        ylim = ax.get_ylim()[1]
        ax.plot(
            [r_stab_thr[0], r_stab_thr[0]],
            [1, ylim],
            "--",
            color="tab:blue",
            linewidth=0.75,
        )
        ax.plot(
            [r_stab_thr[1], r_stab_thr[1]],
            [1, ylim],
            "--",
            color="tab:red",
            linewidth=0.75,
        )
        ax.text(x=r_stab_thr[0] - 0.05, y=ylim * 1.3, s="low", fontsize=6)
        ax.text(x=r_stab_thr[1] - 0.05, y=ylim * 1.3, s="high", fontsize=6)
        ax.set_xlabel("$max_s(r^%d_{\\gamma^+})$" % ds)
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax.yaxis.set_major_locator(locmaj)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(NullFormatter())
        # ax.yaxis.set_minor_locator(MultipleLocator(2))

        # plt.show(block=False)
        # return
        # ax = plt.axes([0.4,0.85,0.1,0.125])
        # ax.hist(dp_pos,np.linspace(-1,1,21),color='k',alpha=0.5)
        # ax.hist(np.nanmax(dp_pos_temp[act_clusters,:][:,s_bool],1),np.linspace(-1,1,21),color='r',alpha=0.5)
        # pl_dat.remove_frame(ax,['top','right'])

        ax = plt.axes([0.1, 0.575, 0.175, 0.125])
        self.pl_dat.add_number(fig, ax, order=2)

        Np = np.zeros((nSes, 3))
        for s in range(nSes):
            Np[s, 0] = (
                act_stab[self.status["activity"][:, s, 1], s] < act_stab_thr[0]
            ).sum()
            Np[s, 2] = (
                act_stab[self.status["activity"][:, s, 1], s] > act_stab_thr[1]
            ).sum()
            Np[s, 1] = self.status["activity"][:, s, 1].sum() - Np[s, 0] - Np[s, 2]
            # Np = np.histogram(act_stab[])
        ax.bar(range(nSes), Np[:, 0], width=1, color="tab:blue")
        ax.bar(
            range(nSes),
            Np[:, 1],
            width=1,
            bottom=Np[:, :1].sum(1),
            alpha=0.5,
            color="k",
        )
        ax.bar(range(nSes), Np[:, 2], width=1, bottom=Np[:, :2].sum(1), color="tab:red")
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax.set_xlabel("session")
        ax.set_ylabel("neurons")

        ax = plt.axes([0.6, 0.575, 0.175, 0.125])
        self.pl_dat.add_number(fig, ax, order=6)

        Np = np.zeros((nSes, 3))
        for s in range(nSes):
            Np[s, 0] = (
                r_stab[self.status["activity"][:, s, 1], s] < r_stab_thr[0]
            ).sum()
            Np[s, 2] = (
                r_stab[self.status["activity"][:, s, 1], s] > r_stab_thr[1]
            ).sum()
            Np[s, 1] = self.status["activity"][:, s, 1].sum() - Np[s, 0] - Np[s, 2]
            # Np = np.histogram(act_stab[])
        ax.bar(range(nSes), Np[:, 0], width=1, color="tab:blue")
        ax.bar(
            range(nSes),
            Np[:, 1],
            width=1,
            bottom=Np[:, :1].sum(1),
            alpha=0.5,
            color="k",
        )
        ax.bar(range(nSes), Np[:, 2], width=1, bottom=Np[:, :2].sum(1), color="tab:red")
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax.set_xlabel("session")
        ax.set_ylabel("neurons")

        ax_extremes = plt.axes([0.4, 0.575, 0.05, 0.125])
        # s_arr = np.arange(17,88,1)#[0,5,10,17,25,40,87,97,112]
        # n_int = len(s_arr)-1
        low_p = np.zeros(nSes) * np.NaN
        high_p = np.zeros(nSes) * np.NaN

        for i, s in enumerate(np.where(s_bool)[0]):
            # act_s_range = np.any(self.status['activity'][:,s_arr[i]:s_arr[i+1],1],1)
            c_act = self.status["activity"][:, s, 1]
            p_pos_hist = act_stab[c_act, s]
            low_p[s] = (p_pos_hist < act_stab_thr[0]).sum() / c_act.sum()
            high_p[s] = (p_pos_hist > act_stab_thr[1]).sum() / c_act.sum()
        ax_extremes.set_ylim([0, 0.5])
        ax_extremes.bar(0, np.nanmean(low_p), facecolor="tab:blue")
        ax_extremes.errorbar(0, np.nanmean(low_p), np.nanstd(low_p), color="k")
        ax_extremes.bar(1, np.nanmean(high_p), facecolor="tab:red")
        ax_extremes.errorbar(1, np.nanmean(high_p), np.nanstd(high_p), color="k")
        ax_extremes.set_xticks([0, 1])
        ax_extremes.set_xticklabels(
            ["low $r_{\\alpha^+}^%d$" % ds, "high $r_{\\alpha^+}^%d$" % ds],
            rotation=60,
            fontsize=8,
        )
        self.pl_dat.remove_frame(ax_extremes, ["top", "right"])
        ax_extremes.set_ylabel("fraction")

        ax_extremes = plt.axes([0.9, 0.575, 0.05, 0.125])
        # s_arr = np.arange(17,88,1)#[0,5,10,17,25,40,87,97,112]
        # n_int = len(s_arr)-1
        # color_act = iter(plt.cm.get_cmap('Greys')(np.linspace(0,1,n_int+1)))#s_bool.sum())))
        low_p = np.zeros(nSes) * np.NaN
        high_p = np.zeros(nSes) * np.NaN

        for i, s in enumerate(np.where(s_bool)[0]):
            # col = next(color_act)
            # act_s_range = np.any(self.status['activity'][:,s_arr[i]:s_arr[i+1],1],1)
            c_act = self.status["activity"][:, s, 1]
            p_pos_hist = r_stab[c_act, s]
            low_p[s] = (p_pos_hist < r_stab_thr[0]).sum() / c_act.sum()
            high_p[s] = (p_pos_hist > r_stab_thr[1]).sum() / c_act.sum()
        ax_extremes.set_ylim([0, 1])
        ax_extremes.bar(0, np.nanmean(low_p), facecolor="tab:blue")
        ax_extremes.errorbar(0, np.nanmean(low_p), np.nanstd(low_p), color="k")
        print(low_p)
        print(np.nanmean(low_p), np.nanstd(low_p))
        ax_extremes.bar(1, np.nanmean(high_p), facecolor="tab:red")
        ax_extremes.errorbar(1, np.nanmean(high_p), np.nanstd(high_p), color="k")
        print(np.nanmean(high_p), np.nanstd(high_p))
        ax_extremes.set_xticks([0, 1])
        ax_extremes.set_xticklabels(
            ["low $r_{\\gamma^+}^%d$" % ds, "high $r_{\\gamma^+}^%d$" % ds],
            rotation=60,
            fontsize=8,
        )
        self.pl_dat.remove_frame(ax_extremes, ["top", "right"])
        ax_extremes.set_ylabel("fraction")

        status = self.status["activity"][..., 1]
        status = status[:, s_bool]

        # print(status.sum(1))
        # print(status.shape)
        # print(r_stab.shape)
        # print(act_stab.shape)
        #
        # print((status[np.nanmax(act_stab[:,s_bool],1)>0.9,:].sum(1)))
        # print((status[np.nanmax(r_stab[:,s_bool],1)>0.9,:].sum(1)))

        ax_Na = plt.axes([0.1, 0.3, 0.35, 0.15])
        self.pl_dat.add_number(fig, ax_Na, order=3)
        Na_distr = np.zeros((nSes_good, 3))
        # Na_distr[:,0] = np.histogram(status[np.nanmax(act_stab[:,s_bool],1)<0.1,:].sum(1),np.linspace(0,nSes_good,nSes_good+1))[0]
        Na_distr[:, 1] = np.histogram(
            status[
                (np.nanmax(act_stab[:, s_bool], 1) >= act_stab_thr[0])
                & (np.nanmax(act_stab[:, s_bool], 1) <= act_stab_thr[1]),
                :,
            ].sum(1),
            np.linspace(0, nSes_good, nSes_good + 1),
        )[0]
        Na_distr[:, 2] = np.histogram(
            status[np.nanmax(act_stab[:, s_bool], 1) > act_stab_thr[1], :].sum(1),
            np.linspace(0, nSes_good, nSes_good + 1),
        )[0]
        Na_distr[:, 0] = np.histogram(
            status[act_clusters, :].sum(1), np.linspace(0, nSes_good, nSes_good + 1)
        )[0]
        Na_distr[:, 0] -= Na_distr[:, 1:].sum(1)
        # print(Na_distr)
        # ax_Na.hist(status.sum(1),np.linspace(0,nSes_good,nSes_good+1),facecolor='g')
        ax_Na.bar(range(nSes_good), Na_distr[:, 0], width=1, color="tab:blue")
        ax_Na.bar(
            range(nSes_good),
            Na_distr[:, 1],
            width=1,
            bottom=Na_distr[:, :1].sum(1),
            alpha=0.5,
            color="k",
        )
        ax_Na.bar(
            range(nSes_good),
            Na_distr[:, 2],
            width=1,
            bottom=Na_distr[:, :2].sum(1),
            color="tab:red",
        )
        ax_Na.set_xlabel("$N_{\\alpha^+}$")
        ax_Na.set_ylabel("count")
        self.pl_dat.remove_frame(ax_Na, ["top", "right"])
        ax_Na.set_ylim([0, 350])
        ax_Na.set_xlim([0, xlim])

        ax_Na_inset = plt.axes([0.325, 0.4, 0.075, 0.05])
        ax_Na_inset.plot(
            Na_distr[:, 0] / Na_distr.sum(1), color="tab:blue", linewidth=0.5
        )
        ax_Na_inset.plot(
            Na_distr[:, 2] / Na_distr.sum(1), color="tab:red", linewidth=0.5
        )
        ax_Na_inset.set_ylim([0, 1])
        ax_Na_inset.set_xlabel("$N_{\\alpha^+}$", fontsize=8)
        ax_Na_inset.xaxis.set_label_coords(0.5, -0.8)
        ax_Na_inset.set_ylabel("fraction", fontsize=8)
        self.pl_dat.remove_frame(ax_Na_inset, ["top", "right"])

        # ax.legend(fontsize=8)

        status = self.status["activity"][..., 1]
        status = status[:, s_bool]
        ax_Na = plt.axes([0.6, 0.3, 0.35, 0.15])
        self.pl_dat.add_number(fig, ax_Na, order=7)
        Na_distr = np.zeros((nSes_good, 3))
        # Na_distr[:,0] = np.histogram(status[np.nanmax(r_stab[:,s_bool],1)<0.1,:].sum(1),np.linspace(0,nSes_good,nSes_good+1))[0]
        Na_distr[:, 1] = np.histogram(
            status[
                (np.nanmax(r_stab[:, s_bool], 1) >= r_stab_thr[0])
                & (np.nanmax(r_stab[:, s_bool], 1) <= r_stab_thr[1]),
                :,
            ].sum(1),
            np.linspace(0, nSes_good, nSes_good + 1),
        )[0]
        Na_distr[:, 2] = np.histogram(
            status[np.nanmax(r_stab[:, s_bool], 1) > r_stab_thr[1], :].sum(1),
            np.linspace(0, nSes_good, nSes_good + 1),
        )[0]
        Na_distr[:, 0] = np.histogram(
            status[act_clusters, :].sum(1), np.linspace(0, nSes_good, nSes_good + 1)
        )[0]
        Na_distr[:, 0] -= Na_distr[:, 1:].sum(1)
        # print(Na_distr)

        # ax_Na.hist(status.sum(1),np.linspace(0,nSes_good,nSes_good+1),facecolor='g')
        ax_Na.bar(range(nSes_good), Na_distr[:, 0], width=1, color="tab:blue")
        ax_Na.bar(
            range(nSes_good),
            Na_distr[:, 1],
            width=1,
            bottom=Na_distr[:, :1].sum(1),
            alpha=0.5,
            color="k",
        )
        ax_Na.bar(
            range(nSes_good),
            Na_distr[:, 2],
            width=1,
            bottom=Na_distr[:, :2].sum(1),
            color="tab:red",
        )
        ax_Na.set_xlabel("$N_{\\alpha^+}$")
        ax_Na.set_ylabel("count")
        self.pl_dat.remove_frame(ax_Na, ["top", "right"])
        ax_Na.set_ylim([0, 350])
        ax_Na.set_xlim([0, xlim])

        ax_Na_inset = plt.axes([0.75, 0.4, 0.075, 0.05])
        ax_Na_inset.plot(
            Na_distr[:, 0] / Na_distr.sum(1), color="tab:blue", linewidth=0.5
        )
        ax_Na_inset.plot(
            Na_distr[:, 2] / Na_distr.sum(1), color="tab:red", linewidth=0.5
        )
        ax_Na_inset.set_ylim([0, 1])
        ax_Na_inset.set_ylabel("fraction", fontsize=8)
        ax_Na_inset.set_xlabel("$N_{\\alpha^+}$", fontsize=8)
        ax_Na_inset.xaxis.set_label_coords(0.5, -0.8)
        self.pl_dat.remove_frame(ax_Na_inset, ["top", "right"])

        status = self.status["activity"][..., 2]
        status = status[:, s_bool]
        Na_distr = np.zeros((nSes_good, 3))
        # Na_distr[:,0] = np.histogram(status[np.nanmax(r_stab[:,s_bool],1)<0.1,:].sum(1),np.linspace(0,nSes_good,nSes_good+1))[0]
        Na_distr[:, 1] = np.histogram(
            status[
                (np.nanmax(r_stab[:, s_bool], 1) >= r_stab_thr[0])
                & (np.nanmax(r_stab[:, s_bool], 1) <= r_stab_thr[1]),
                :,
            ].sum(1),
            np.linspace(0, nSes_good, nSes_good + 1),
        )[0]
        Na_distr[:, 2] = np.histogram(
            status[np.nanmax(r_stab[:, s_bool], 1) > r_stab_thr[1], :].sum(1),
            np.linspace(0, nSes_good, nSes_good + 1),
        )[0]
        Na_distr[:, 0] = np.histogram(
            status.sum(1), np.linspace(0, nSes_good, nSes_good + 1)
        )[0]
        Na_distr[:, 0] -= Na_distr[:, 1:].sum(1)

        ax_Na_inset = plt.axes([0.875, 0.4, 0.075, 0.05])
        ax_Na_inset.plot(
            Na_distr[:, 0] / Na_distr.sum(1), color="tab:blue", linewidth=0.5
        )
        ax_Na_inset.plot(
            Na_distr[:, 2] / Na_distr.sum(1), color="tab:red", linewidth=0.5
        )
        ax_Na_inset.set_ylim([0, 1])
        ax_Na_inset.set_xlabel("$N_{\\beta^+}$", fontsize=8)
        ax_Na_inset.xaxis.set_label_coords(0.5, -0.8)
        # ax_Na_inset.set_ylabel('fraction',fontsize=8)
        self.pl_dat.remove_frame(ax_Na_inset, ["top", "right"])

        # ax.legend(fontsize=8)

        ax_act = plt.axes([0.1, 0.1, 0.35, 0.1])
        self.pl_dat.add_number(fig, ax_act, order=4)

        ax_stab = plt.axes([0.6, 0.1, 0.35, 0.1])
        self.pl_dat.add_number(fig, ax_stab, order=8)

        color_t = iter(plt.cm.get_cmap("Greys")(np.linspace(1, 0, 5)))
        # for ds in [1,3,5]:
        print(ds)
        col = next(color_t)

        # act_stab = get_act_stability_temp(cluster,ds=ds)[...,1]
        # r_stab = gauss_smooth(get_field_stability_temp(cluster,SD=1.96,ds=ds),(0,1))

        p_pos_high = act_stab > act_stab_thr[1]
        p_pos_high_recurr = np.zeros((nSes, nSes)) * np.NaN
        for s in np.where(s_bool)[0]:
            p_pos_high_recurr[s, : nSes - s] = (
                p_pos_high[p_pos_high[:, s], s:].sum(0) / p_pos_high[:, s].sum()
            )
            p_pos_high_recurr[s, np.where(~s_bool[s:])[0]] = np.NaN

        # ax_act.plot(np.nanmean(p_pos_high_recurr,0),color=col)
        self.pl_dat.plot_with_confidence(
            ax_act,
            np.arange(nSes),
            np.nanmean(p_pos_high_recurr, 0),
            np.nanstd(p_pos_high_recurr, 0),
            col=col,
        )

        r_stab_high = r_stab > r_stab_thr[1]
        r_stab_high_recurr = np.zeros((nSes, nSes)) * np.NaN
        for s in np.where(s_bool)[0]:
            r_stab_high_recurr[s, : nSes - s] = (
                r_stab_high[r_stab_high[:, s], s:].sum(0) / r_stab_high[:, s].sum()
            )
            r_stab_high_recurr[s, np.where(~s_bool[s:])[0]] = np.NaN
        self.pl_dat.plot_with_confidence(
            ax_stab,
            np.arange(nSes),
            np.nanmean(r_stab_high_recurr, 0),
            np.nanstd(r_stab_high_recurr, 0),
            col=col,
            label="$\delta s = %d$" % ds,
        )

        for axx in [ax_act, ax_stab]:
            axx.set_ylim([0, 1])
            axx.set_xlim([0, xlim])
            axx.set_ylabel("overlap")
            axx.set_xlabel("$\Delta sessions$")
            self.pl_dat.remove_frame(axx, ["top", "right"])
        ax_stab.legend(fontsize=8, loc="upper right", bbox_to_anchor=[1.1, 1.3])

        # ax = plt.axes([0.6,0.6,0.35,0.25])
        # ax.plot(act_stab+0.05*np.random.rand(nC,nSes_good),act_stab+0.05*np.random.rand(nC,nSes_good),'k.',markersize=1,markeredgecolor='none')

        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig("individuals1")

        return

        status_La = np.zeros((nC, nSes, 2), "int")
        status_Lb = np.zeros_like(self.status["activity"][..., 2], "int")

        highCode_thr = 0.5
        IPI_La = np.zeros(nSes)
        IPI_Lb = np.zeros(nSes)
        La_highCode = np.zeros(nSes)

        IPI_La_start = np.zeros_like(self.status["activity"][..., 1], "bool")
        IPI_Lb_start = np.zeros_like(self.status["activity"][..., 2], "bool")

        idx_fields = np.where(self.status["fields"])

        for c in range(nC):
            s0_act = 0
            s0_code = 0
            inAct = False
            inCode = False
            for s in np.where(self.status["sessions"])[0]:
                if inAct:
                    if ~self.status["activity"][c, s, 1]:
                        La = self.status["sessions"][s0_act:s].sum()
                        status_La[c, s0_act:s, 0] = La
                        status_La[c, s0_act:s, 1] = self.status["activity"][
                            c, s0_act:s, 2
                        ].sum()
                        if (
                            self.status["activity"][c, s0_act:s, 2].sum() / La
                        ) > highCode_thr:
                            La_highCode[La] += 1
                        IPI_La[La] += 1
                        inAct = False
                else:
                    if self.status["activity"][c, s, 1]:
                        s0_act = s
                        inAct = True
                        IPI_La_start[c, s] = True

                if inCode:
                    if ~self.status["activity"][c, s, 2]:
                        Lb = self.status["sessions"][s0_code:s].sum()
                        status_Lb[c, s0_code:s] = Lb
                        IPI_Lb[Lb] += 1
                        inCode = False
                else:
                    if self.status["activity"][c, s, 2]:
                        s0_code = s
                        inCode = True
                        IPI_Lb_start[c, s] = True

            if inAct:
                La = self.status["sessions"][s0_act : s + 1].sum()
                status_La[c, s0_act : s + 1, 0] = La
                status_La[c, s0_act : s + 1, 1] = self.status["activity"][
                    c, s0_act : s + 1, 2
                ].sum()
                if (self.status["activity"][c, s0_act:s, 2].sum() / La) > highCode_thr:
                    La_highCode[La] += 1
                IPI_La[La] += 1
            if inCode:
                Lb = self.status["sessions"][s0_code : s + 1].sum()
                status_Lb[c, s0_code : s + 1] = Lb
                IPI_Lb[Lb] += 1

        status_La[:, ~self.status["sessions"], :] = 0
        status_Lb[:, ~self.status["sessions"]] = 0
        # L_code = status_La[self.status['activity'][...,2],0]

        status, status_dep = get_status_arr(cluster)

        # plt.figure(figsize=(7,5),dpi=pl_dat.sv_opt['dpi'])
        #
        # s_arr = [0,5,10,17,25,40,87,97,112]
        # s_arr = np.arange(0,112,1)#[0,5,10,17,25,40,87,97,112]
        # n_int = len(s_arr)-1
        # ax = plt.subplot(221)
        # s_interval = 10
        # fstab_mean = np.zeros((n_int,2))
        # astab_mean = np.zeros((n_int,2))
        #
        # color_field = iter(plt.cm.get_cmap('rainbow')(np.linspace(0,1,n_int)))
        # color_act = iter(plt.cm.get_cmap('Greys')(np.linspace(0,1,n_int)))
        # for i in range(n_int):
        #     col = next(color_field)
        #     ax.hist(self.stats['field_stability'][self.status['clusters'],s_arr[i]:s_arr[i+1]].flat,np.linspace(0,1,51),cumulative=True,density=True,histtype='step',color=col)
        #
        #     fstab_mean[i,0] = np.nanmean(self.stats['field_stability'][self.status['clusters'],s_arr[i]:s_arr[i+1]])
        #     fstab_mean[i,1] = np.nanstd(self.stats['field_stability'][self.status['clusters'],s_arr[i]:s_arr[i+1]])
        #
        #
        #     col = next(color_act)
        #     ax.hist(self.stats['act_stability'][self.status['clusters'],s_arr[i]:s_arr[i+1]].flat,np.linspace(0,1,51),cumulative=True,density=True,histtype='step',color=col)
        #
        #     astab_mean[i,0] = np.nanmean(self.stats['act_stability'][self.status['clusters'],s_arr[i]:s_arr[i+1]])
        #     astab_mean[i,1] = np.nanstd(self.stats['act_stability'][self.status['clusters'],s_arr[i]:s_arr[i+1]])
        # ax = plt.subplot(222)
        # pl_dat.plot_with_confidence(ax,np.arange(n_int)-0.1,fstab_mean[:,0],fstab_mean[:,1],col='r')
        # pl_dat.plot_with_confidence(ax,np.arange(n_int)+0.1,astab_mean[:,0],astab_mean[:,1],col='k')
        # plt.show(block=False)

        fig = plt.figure(figsize=(7, 5), dpi=pl_dat.sv_opt["dpi"])

        ax = plt.subplot(331)
        ax.plot(self.status["activity"][..., 2].sum(0), "k.", markersize=1)
        # ax.hist(np.nanmax(self.stats['field_stability'],1),np.linspace(0,2,101))

        ax = plt.subplot(332)
        idx_med = (self.stats["field_stability"] > 0.5)[..., np.newaxis] & (
            self.status["fields"]
        )
        idx_high = (self.stats["field_stability"] > 0.9)[..., np.newaxis] & (
            self.status["fields"]
        )
        ax.hist(
            self.fields["location"][idx_med, 0],
            np.linspace(0, 100, 101),
            color="tab:blue",
            density=True,
            alpha=0.5,
        )
        ax.hist(
            self.fields["location"][idx_high, 0],
            np.linspace(0, 100, 101),
            color="tab:red",
            density=True,
            alpha=0.5,
        )
        # ax.scatter(np.nanmax(field_stability,1)+0.02*np.random.rand(nC),self.stats['p_post_c']['code']['code'][:,1,0,0]+0.02*np.random.rand(nC),s=self.status['activity'][...,2].sum(1)/10,c='k',edgecolor='none')

        ax = plt.subplot(333)
        ax.plot(np.nanmean(self.stats["field_stability"], 0), "r")
        ax.plot(np.nanmean(self.stats["act_stability"], 0), "k")

        ax = plt.subplot(334)
        ax.plot(
            (self.stats["field_stability"] > 0.5).sum(0)
            / self.status["activity"][..., 2].sum(0),
            "r",
        )
        ax.plot(
            (self.stats["act_stability"][..., 1] > 0.5).sum(0)
            / self.status["activity"][..., 1].sum(0),
            "k",
        )

        ax = plt.subplot(335)
        for thr in np.linspace(0, 1, 51):
            ax.plot(
                thr,
                np.any(self.stats["field_stability"] > thr, 1).sum(),
                "k.",
                markersize=1,
            )

        ax = plt.subplot(336)
        La_mean = np.zeros((nSes, 3)) * np.NaN
        Lb_mean = np.zeros((nSes, 3)) * np.NaN
        for s in np.where(self.status["sessions"])[0]:
            La_mean[s, 0] = status_La[status_La[:, s, 0] > 0, s, 0].mean()
            La_mean[s, 1:] = np.percentile(
                status_La[status_La[:, s, 0] > 0, s, 0], [5, 95]
            )

            Lb_mean[s, 0] = status_Lb[status_Lb[:, s] > 0, s].mean()
            Lb_mean[s, 1:] = np.percentile(status_Lb[status_Lb[:, s] > 0, s], [5, 95])

        pl_dat.plot_with_confidence(
            ax, range(nSes), La_mean[:, 0], La_mean[:, 1:].T, col="k"
        )
        pl_dat.plot_with_confidence(
            ax, range(nSes), Lb_mean[:, 0], Lb_mean[:, 1:].T, col="b"
        )

        ax = plt.subplot(337)
        # print(stat)
        ax.plot(status["stable"][:, :, 1].sum(0), "k", linewidth=0.5)
        # ax = plt.subplot(337)
        # ax.plot(status['stable'][:,:,2].sum(0),'b',linewidth=0.5)
        # ax.plot(status['stable'][:,:,3].sum(0),'r',linewidth=0.5)

        ax = plt.subplot(338)
        ax.plot(self.stats["p_post_s"]["code"]["stable"][:, 1, 0])

        ax = plt.subplot(339)
        ax.hist(
            status["stable"][self.status["clusters"], :, 1].sum(1),
            np.linspace(0, 100, 101),
        )

        plt.show(block=False)

        # return field_stability

    def plot_field_stats(self):

        nSes = self.data["nSes"]
        s_bool = np.zeros(nSes, "bool")
        # s_bool[17:87] = True
        s_bool[0:15] = True

        idx_PF = self.status["fields"] & s_bool[np.newaxis, :, np.newaxis]

        plt.figure(figsize=(7, 5), dpi=300)
        plt.subplot(221)
        plt.plot(
            self.stats["MI_value"][np.where(idx_PF)[0], np.where(idx_PF)[1]],
            self.stats["Isec_value"][np.where(idx_PF)[0], np.where(idx_PF)[1]],
            "k.",
            markersize=1,
        )

        plt.subplot(223)
        plt.plot(
            self.fields["reliability"][idx_PF],
            self.stats["MI_value"][np.where(idx_PF)[0], np.where(idx_PF)[1]],
            "k.",
            markersize=1,
        )
        plt.subplot(224)
        plt.plot(
            self.fields["reliability"][idx_PF],
            self.stats["Isec_value"][np.where(idx_PF)[0], np.where(idx_PF)[1]],
            "r.",
            markersize=1,
        )

        plt.show(block=False)

    def plot_alt_pf_detection(self):

        print("### get alternative place field detection from place maps directly ###")

        nSes = self.data["nSes"]
        nbin = self.data["nbin"]
        nC = self.data["nC"]

        ### smooth firingmap
        ### find, if some part is > 4*SD+baseline
        ### find if part is > 4 bins width
        ### center is place field! (highest only)
        SD = 3
        MI_thr = 1  # 0.1
        loc = np.zeros((nC, nSes)) * np.NaN
        width = np.zeros((nC, nSes)) * np.NaN
        fields = np.zeros((nSes, nbin)) * np.NaN
        PC_ct = np.zeros(nSes)
        # plt.figure(figsize=(7,5),dpi=300)
        for s in np.where(self.status["sessions"])[0]:
            fmaps = np.copy(self.stats["firingmap"][:, s, :])
            baseline = np.percentile(fmaps, 20, axis=1)
            fmap_thr = np.zeros(self.data["nC"])
            # SD =
            fmaps_base_subtracted = fmaps - baseline[:, np.newaxis]
            N = (fmaps_base_subtracted <= 0).sum(1)
            fmaps_base_subtracted *= -1.0 * (fmaps_base_subtracted <= 0)
            noise = np.sqrt((fmaps_base_subtracted**2).sum(1) / (N * (1 - 2 / np.pi)))
            fmap_thr = baseline + SD * noise
            # print(fmap_thr)
            fmaps = gauss_smooth(fmaps, (0, 4))
            for c in np.where(self.status["activity"][:, s, 1])[0]:
                surp = (fmaps[c, :] >= fmap_thr[c]).sum()
                if (self.stats["MI_p_value"][c, s] < MI_thr) & (surp > 4):
                    loc[c, s] = np.argmax(fmaps[c, :])
                    width[c, s] = surp
                    # print('this is placecell!')
                    PC_ct[s] += 1
            # print('session %d'%s)
            # print('Place cells detected: %d'%PC_ct)
            # print(loc[c,:])
            # plt.subplot(5,5,s+1)
            # plt.title('PCs: %d'%PC_ct)
            # plt.hist(loc[:,s],np.linspace(0,100,41))
            # print(np.histogram(loc[c,np.isfinite(loc[c,:])],np.linspace(0,100,101),density=True)[0])
            fields[s, :] = np.histogram(
                loc[:, s], np.linspace(0, nbin, nbin + 1), density=True
            )[0]
        # plt.show(block=False)

        plt.figure(figsize=(7, 5), dpi=300)
        plt.subplot(211)
        plt.plot(PC_ct, "k")
        plt.subplot(212)
        plt.imshow(fields, aspect="auto", origin="lower", cmap="hot", clim=[0, 0.03])
        plt.show(block=False)

        plt.figure(figsize=(7, 5), dpi=300)
        # s_arr = [0,17,40,87,97,107]
        s_arr = [0, 5, 10, 15, 20]
        n_int = len(s_arr) - 1
        color_t = iter(plt.cm.get_cmap("Greys")(np.linspace(0, 1, n_int + 1)))
        for i in range(n_int):
            col = next(color_t)
            plt.plot(np.nanmean(fields[s_arr[i] : s_arr[i + 1], :], 0), color=col)
        plt.show(block=False)

    def plot_multi_modes(self, sv=False):

        print("### whats up with multiple peaks? ###")

        nSes = self.data["nSes"]
        nbin = self.data["nbin"]
        nC = self.data["nC"]

        nFields = np.ma.masked_array(
            self.status["fields"].sum(-1), mask=~self.status["activity"][..., 2]
        )
        idx = nFields > 1
        nMultiMode = nFields.mean(0)

        dLoc = np.zeros((nC, nSes)) * np.NaN
        corr = np.zeros((nC, nSes)) * np.NaN
        overlap = np.zeros((nC, nSes)) * np.NaN
        for c, s in zip(np.where(idx)[0], np.where(idx)[1]):
            # pass
            loc = self.fields["location"][c, s, self.status["fields"][c, s, :], 0]
            dLoc[c, s] = np.abs(
                np.mod(loc[1] - loc[0] + nbin / 2, nbin) - nbin / 2
            )  # loc[1]-loc[0]

            idx_loc = np.where(self.status["fields"][c, s, :])[0]

            corr[c, s] = np.corrcoef(
                self.fields["trial_act"][
                    c, s, idx_loc[0], : self.behavior["trial_ct"][s]
                ],
                self.fields["trial_act"][
                    c, s, idx_loc[1], : self.behavior["trial_ct"][s]
                ],
            )[0, 1]

            overlap[c, s] = (
                self.fields["trial_act"][
                    c, s, idx_loc[0], : self.behavior["trial_ct"][s]
                ]
                & self.fields["trial_act"][
                    c, s, idx_loc[1], : self.behavior["trial_ct"][s]
                ]
            ).sum()

        fig = plt.figure(figsize=(7, 5), dpi=self.pl_dat.sv_opt["dpi"])
        ax = plt.axes([0.1, 0.75, 0.35, 0.175])
        self.pl_dat.add_number(fig, ax, order=1)
        ax.plot(nMultiMode, "k")
        ax.set_ylim([0.98, 1.2])
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax.set_xlabel("session")
        ax.set_ylabel("$\left \langle \# fields \\right \\rangle$")

        ax = plt.axes([0.55, 0.75, 0.35, 0.175])
        self.pl_dat.add_number(fig, ax, order=2, offset=[-50, 50])

        ax.hist(
            self.fields["location"][nFields == 1, :, 0].flat,
            np.linspace(0, nbin, 101),
            facecolor="k",
            density=True,
            label="1 field",
        )
        ax.hist(
            self.fields["location"][idx, :, 0].flat,
            np.linspace(0, nbin, 101),
            facecolor="tab:orange",
            alpha=0.5,
            density=True,
            label="2 fields",
        )
        self.pl_dat.remove_frame(ax, ["top", "right", "left"])
        ax.set_yticks([])
        ax.set_xlabel("position [bins]")
        ax.legend(
            fontsize=8, loc="upper right", bbox_to_anchor=[0.85, 1.2], handlelength=1
        )

        ax = plt.axes([0.65, 0.1, 0.3, 0.42])
        self.pl_dat.add_number(fig, ax, order=5)
        ax.plot(dLoc[overlap == 0], corr[overlap == 0], "k.", markersize=1, zorder=10)
        ax.plot(
            dLoc[overlap > 0],
            corr[overlap > 0],
            ".",
            color="tab:red",
            markersize=1,
            zorder=12,
        )
        ax.set_xlim([0, 50])
        ax.set_ylim([-1, 1])
        ax.set_yticks(np.linspace(-1, 1, 5))
        ax.set_xlabel("$\Delta \\theta [bins]$")
        ax.set_ylabel("$c_a$")

        ax2 = ax.twiny()
        ax2.hist(
            corr.flat,
            np.linspace(-1, 1, 51),
            orientation="horizontal",
            facecolor="tab:orange",
            alpha=0.5,
            zorder=0,
        )
        ax2.set_xlim([0, ax2.get_xlim()[1] * 4])
        ax2.set_xticks([])

        ax3 = ax.twinx()
        ax3.hist(
            dLoc.flat,
            np.linspace(0, 50, 51),
            orientation="vertical",
            facecolor="tab:orange",
            alpha=0.5,
            zorder=0,
        )
        ax3.set_ylim([ax3.get_ylim()[1] * 4, 0])
        ax3.set_yticks([])

        ### plot "proper" 2-field
        idx = np.where(dLoc > 10)
        i = np.random.randint(len(idx[0]))
        c = idx[0][i]
        s = idx[1][i]
        # c,s = [12,73]
        print(c, s)

        ax_fmap = plt.axes([0.1, 0.4, 0.35, 0.15])
        self.pl_dat.add_number(fig, ax_fmap, order=3)
        ax_fmap.bar(
            np.linspace(1, nbin, nbin),
            gauss_smooth(self.stats["firingmap"][c, s, :], 1),
            width=1,
            facecolor="k",
        )
        ax_fmap.set_ylabel("$\\bar{\\nu}$")

        loc = self.fields["location"][c, s, self.status["fields"][c, s, :], 0]
        ax_trial = plt.axes([0.375, 0.525, 0.125, 0.1])
        idx_loc = np.where(self.status["fields"][c, s, :])[0]
        self.pl_dat.remove_frame(ax_fmap, ["top", "right"])

        col_arr = ["tab:green", "tab:blue"]
        for i, f in enumerate(idx_loc):
            ax_fmap.plot(loc[i], 1, "v", color=col_arr[i], markersize=5)
            ax_trial.bar(
                range(self.behavior["trial_ct"][s]),
                self.fields["trial_act"][c, s, f, : self.behavior["trial_ct"][s]],
                bottom=i,
                color=col_arr[i],
                alpha=0.5,
            )

        ax_fmap.arrow(
            x=loc.min(),
            y=ax_fmap.get_ylim()[1] * 0.95,
            dx=loc.max() - loc.min(),
            dy=0,
            shape="full",
            color="tab:orange",
            width=0.02,
            head_width=0.4,
            head_length=2,
            length_includes_head=True,
        )  # "$\Delta \\theta$",
        ax_fmap.arrow(
            x=loc.max(),
            y=ax_fmap.get_ylim()[1] * 0.95,
            dx=loc.min() - loc.max(),
            dy=0,
            shape="full",
            color="tab:orange",
            width=0.02,
            head_width=0.4,
            head_length=2,
            length_includes_head=True,
        )  # "$\Delta \\theta$",
        ax_fmap.text(
            loc.min() / 2 + loc.max() / 2,
            ax_fmap.get_ylim()[1],
            "$\Delta \\theta$",
            color="tab:orange",
            fontsize=10,
            ha="center",
        )

        self.pl_dat.remove_frame(ax_trial, ["top", "right", "left"])
        ax_trial.set_yticks([])
        ax_trial.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax_trial.set_xlabel("trial", fontsize=10)  # ,labelpad=-5,x=0.4)
        ax_trial.xaxis.set_label_coords(0.3, -0.3)
        ax_trial.text(-10, 1.2, s="$c_a=%.2f$" % corr[c, s], fontsize=6)

        ### plot "improper" 2-field
        idx = np.where(dLoc < 10)
        i = np.random.randint(len(idx[0]))
        c = idx[0][i]
        s = idx[1][i]
        # c,s = [490,44]
        print(c, s)

        ax_fmap = plt.axes([0.1, 0.1, 0.35, 0.15])
        self.pl_dat.add_number(fig, ax_fmap, order=4)
        ax_fmap.bar(
            np.linspace(1, nbin, nbin),
            gauss_smooth(self.stats["firingmap"][c, s, :], 1),
            width=1,
            facecolor="k",
        )
        self.pl_dat.remove_frame(ax_fmap, ["top", "right"])
        ax_fmap.set_ylabel("$\\bar{\\nu}$")
        ax_fmap.set_xlabel("position [bins]")

        loc = self.fields["location"][c, s, self.status["fields"][c, s, :], 0]
        ax_trial = plt.axes([0.375, 0.225, 0.125, 0.1])
        idx_loc = np.where(self.status["fields"][c, s, :])[0]
        for i, f in enumerate(idx_loc):

            ax_fmap.plot(loc[i], 1, "v", color=col_arr[i], markersize=5)
            ax_trial.bar(
                range(self.behavior["trial_ct"][s]),
                self.fields["trial_act"][c, s, f, : self.behavior["trial_ct"][s]],
                bottom=i,
                color=col_arr[i],
                alpha=0.5,
            )

        ax_fmap.arrow(
            x=loc.min(),
            y=ax_fmap.get_ylim()[1] * 0.95,
            dx=loc.max() - loc.min(),
            dy=0,
            shape="full",
            color="tab:orange",
            width=0.015,
            head_width=0.2,
            head_length=2,
            length_includes_head=True,
        )  # "$\Delta \\theta$",
        ax_fmap.arrow(
            x=loc.max(),
            y=ax_fmap.get_ylim()[1] * 0.95,
            dx=loc.min() - loc.max(),
            dy=0,
            shape="full",
            color="tab:orange",
            width=0.015,
            head_width=0.2,
            head_length=2,
            length_includes_head=True,
        )  # "$\Delta \\theta$",
        ax_fmap.text(
            loc.min() / 2 + loc.max() / 2,
            ax_fmap.get_ylim()[1],
            "$\Delta \\theta$",
            color="tab:orange",
            fontsize=10,
            ha="center",
        )
        self.pl_dat.remove_frame(ax_trial, ["top", "right", "left"])
        ax_trial.set_yticks([])
        ax_trial.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax_trial.set_xlabel("trial", fontsize=10)  # ,labelpad=-5,x=0.4)
        ax_trial.xaxis.set_label_coords(0.3, -0.3)
        ax_trial.text(-10, 1.2, s="$c_a=%.2f$" % corr[c, s], fontsize=6)

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig("multi_modes")

    def plot_coding_overlap(self):

        nSes = self.data["nSes"]
        s_bool = np.zeros(nSes, "bool")
        # s_bool[17:87] = True
        s_bool[:] = True
        s_bool[~self.status["sessions"]] = False

        t_ct_max = max(self.behavior["trial_ct"])

        nsteps = 4
        perc_act = np.zeros((nsteps, nSes, t_ct_max)) * np.NaN
        perc_coding = np.zeros((nsteps, nSes, t_ct_max)) * np.NaN
        perc_act_overlap = np.zeros((nsteps, nSes, t_ct_max)) * np.NaN
        perc_coding_overlap = np.zeros((nsteps, nSes, t_ct_max)) * np.NaN

        field_var = np.zeros((nSes, 2)) * np.NaN

        coding_fit = np.zeros((nsteps, nSes, 2, 3)) * np.NaN
        coding_overlap_fit = np.zeros((nsteps, nSes, 2, 3)) * np.NaN
        coding_overlap = np.zeros((nSes, 2, 2)) * np.NaN

        fig = plt.figure(figsize=(7, 5), dpi=300)

        # plt.show(block=False)
        # return
        ax_overlap = plt.axes([0.1, 0.125, 0.2, 0.5])
        # pl_dat.add_number(fig,ax_fit_overlap,order=9)
        # ax_fit_coact = plt.axes([0.725,0.425,0.225,0.2])
        # pl_dat.add_number(fig,ax_fit_coact,order=6)
        j = 0
        # color_t = plt.cm.rainbow(np.linspace(0,1,nsteps))

        dt = 5

        for s in tqdm(np.where(s_bool)[0][:-1]):

            if s_bool[s + 1]:
                t_start = min(self.behavior["trial_ct"][s], dt)
                t_end = max(0, self.behavior["trial_ct"][s] - dt)
                coding_s1_start = (
                    np.any(self.fields["trial_act"][:, s, :, :t_start], -1)
                    & self.status["fields"][:, s, :]
                )
                coding_s1_end = (
                    np.any(self.fields["trial_act"][:, s, :, t_end:], -1)
                    & self.status["fields"][:, s, :]
                )

                ### get first dt trials and last dt trials
                t_start = self.behavior["trial_ct"][
                    s + 1
                ]  # min(self.behavior['trial_ct'][s+1],dt)
                t_end = 0  # max(0,self.behavior['trial_ct'][s+1]-dt)
                coding_s2_start = (
                    np.any(self.fields["trial_act"][:, s + 1, :, :t_start], -1)
                    & self.status["fields"][:, s + 1, :]
                )
                coding_s2_end = (
                    np.any(self.fields["trial_act"][:, s + 1, :, t_end:], -1)
                    & self.status["fields"][:, s + 1, :]
                )

                coding_overlap[s, 0, 0] = (
                    coding_s2_start[coding_s1_start].sum() / coding_s1_start.sum()
                )
                coding_overlap[s, 0, 1] = (
                    coding_s2_end[coding_s1_start].sum() / coding_s1_start.sum()
                )
                coding_overlap[s, 1, 0] = (
                    coding_s2_start[coding_s1_end].sum() / coding_s1_end.sum()
                )
                coding_overlap[s, 1, 1] = (
                    coding_s2_end[coding_s1_end].sum() / coding_s1_end.sum()
                )

        ax_overlap.plot(
            np.random.rand(nSes) * 0.5, coding_overlap[:, 0, 0], "k.", markersize=1
        )
        ax_overlap.errorbar(
            0.25,
            np.nanmean(coding_overlap[:, 0, 0]),
            np.nanstd(coding_overlap[:, 0, 0]),
            fmt="r.",
            markersize=5,
            linestyle="none",
        )
        ax_overlap.plot(
            1 + np.random.rand(nSes) * 0.5, coding_overlap[:, 1, 0], "k.", markersize=1
        )
        ax_overlap.errorbar(
            1.25,
            np.nanmean(coding_overlap[:, 1, 0]),
            np.nanstd(coding_overlap[:, 1, 0]),
            fmt="r.",
            markersize=5,
            linestyle="none",
        )

        ax_overlap.plot(
            2 + np.random.rand(nSes) * 0.5, coding_overlap[:, 0, 1], "k.", markersize=1
        )
        ax_overlap.errorbar(
            2.25,
            np.nanmean(coding_overlap[:, 0, 1]),
            np.nanstd(coding_overlap[:, 0, 1]),
            fmt="r.",
            markersize=5,
            linestyle="none",
        )
        ax_overlap.plot(
            3 + np.random.rand(nSes) * 0.5, coding_overlap[:, 1, 1], "k.", markersize=1
        )
        ax_overlap.errorbar(
            3.25,
            np.nanmean(coding_overlap[:, 1, 1]),
            np.nanstd(coding_overlap[:, 1, 1]),
            fmt="r.",
            markersize=5,
            linestyle="none",
        )
        ax_overlap.set_ylim([0, 0.5])

        res = sstats.kruskal(
            coding_overlap[:, 0, 0], coding_overlap[:, 1, 0], nan_policy="omit"
        )
        print(res)

        res = sstats.kruskal(
            coding_overlap[:, 1, 0], coding_overlap[:, 1, 1], nan_policy="omit"
        )
        print(res)

        plt.show(block=False)

    def plot_location_dependent_stability(self, sv=False):

        print("### plot location-specific stability ###")

        nSes = self.data["nSes"]
        nbin = self.data["nbin"]

        self.params["zone_mask"] = {
            "gate": np.zeros(nbin, "bool"),
            "reward": np.zeros(nbin, "bool"),
            "others": np.zeros(nbin, "bool"),
        }

        self.params["zone_mask"]["gate"][20:35] = True
        self.params["zone_mask"]["reward"][50:65] = True
        self.params["zone_mask"]["others"] = (
            ~self.params["zone_mask"]["gate"] & ~self.params["zone_mask"]["reward"]
        )
        # p_rec = {'all':     np.zeros(nSes)*np.NaN,
        #          'gate':    np.zeros(nSes)*np.NaN,
        #          'reward':  np.zeros(nSes)*np.NaN,
        #          'others':  np.zeros(nSes)*np.NaN}

        if nSes > 50:
            s_arr = np.array([0, 5, 17, 30, 87])
        # s_arr = np.array([0,16,60,87,96,107])
        else:
            s_arr = np.array([0, 5, 10, 15, 20])
        s_arr += np.where(self.status["sessions"])[0][0]
        print(s_arr)
        # s_arr = np.array([0,10,21])
        n_int = len(s_arr) - 1

        ds = 1
        session_bool = np.where(
            np.pad(self.status["sessions"][ds:], (0, ds), constant_values=False)
            & np.pad(self.status["sessions"][:], (0, 0), constant_values=False)
        )[0]
        # print(session_bool)
        loc_stab = np.zeros((nSes, nbin + 2, nbin + 2))
        loc_stab_p = np.zeros((nSes, nbin + 2, nbin + 2))
        for s in session_bool:  # range(nSes):#np.where(self.status['sessions'])[0]:
            ### assign bin-specific remapping to rows, active nPC (nbins+1) & silent (nbins+2)
            for c in np.where(self.status["activity"][:, s, 2])[0]:
                ## find belonging fields
                if self.status["activity"][c, s + ds, 2]:
                    d = np.abs(
                        np.mod(
                            self.fields["location"][c, s, :, 0][:, np.newaxis]
                            - self.fields["location"][c, s + ds, :, 0]
                            + nbin / 2,
                            nbin,
                        )
                        - nbin / 2
                    )
                    d[np.isnan(d)] = nbin
                    f1, f2 = sp.optimize.linear_sum_assignment(d)
                    for f in zip(f1, f2):
                        if d[f] < nbin:
                            loc_stab[
                                s,
                                int(round(self.fields["location"][c, s, f[0], 0])),
                                int(round(self.fields["location"][c, s + ds, f[1], 0])),
                            ] += 1
                            loc_stab_p[
                                s,
                                int(round(self.fields["location"][c, s, f[0], 0])),
                                :nbin,
                            ] += self.fields["p_x"][c, s + ds, f[1], :]

        loc_stab = loc_stab[:, :nbin, :nbin]
        loc_stab_p = loc_stab_p[:, :nbin, :nbin]

        p_rec_loc = np.zeros((n_int, nbin, nSes)) * np.NaN

        s1_shifts, s2_shifts, f1, f2 = np.unravel_index(
            self.compare["pointer"].col,
            (
                nSes,
                nSes,
                self.params["field_count_max"],
                self.params["field_count_max"],
            ),
        )
        c_shifts = self.compare["pointer"].row
        sig_theta = self.stability["all"]["mean"][0, 2]
        print(sig_theta)
        di = 3
        SD = 2
        for ds in range(1, min(nSes, 21)):
            # session_bool = np.where(np.pad(self.status['sessions'][ds:],(0,ds),constant_values=False) & np.pad(self.status['sessions'][:],(0,0),constant_values=False))[0]

            ### somehow condition this on the location
            # for s1 in session_bool:
            #     overlap = self.status['activity'][self.status['activity'][:,s1,1],s1+ds,1].sum(0).astype('float')
            #     N_ref = self.status['activity'][:,s1,1].sum(0)
            #     p_rec['act'][ds,s1] = (overlap/N_ref)
            #
            #     overlap = self.status['activity'][self.status['activity'][:,s1,2],s1+ds,2].sum(0).astype('float')
            #     N_ref = self.status['activity'][self.status['activity'][:,s1,2],s1+ds,1].sum(0)
            #     p_rec['PC'][ds,s1] = (overlap/N_ref)

            Ds = s2_shifts - s1_shifts
            idx = np.where(Ds == ds)[0]
            idx_shifts = self.compare["pointer"].data[idx].astype("int") - 1
            shifts = self.compare["shifts"][idx_shifts]

            s = s1_shifts[idx]
            f = f1[idx]
            c = c_shifts[idx]
            loc_shifts = np.round(self.fields["location"][c, s, f, 0]).astype("int")

            for j in range(len(s_arr) - 1):
                for i in range(nbin):
                    i_min = max(0, i - di)
                    i_max = min(nbin, i + di)
                    idx_loc = (
                        (loc_shifts >= i_min)
                        & (loc_shifts < i_max)
                        & ((s >= s_arr[j]) & (s < s_arr[j + 1]))
                    )

                    shifts_loc = shifts[idx_loc]
                    N_data = len(shifts_loc)
                    N_stable = (np.abs(shifts_loc) < (SD * sig_theta)).sum()

                    p_rec_loc[j, i, ds] = N_stable / N_data

        plt.figure(figsize=(7, 5), dpi=self.pl_dat.sv_opt["dpi"])

        gate = np.any(self.params["zone_mask"]["gate"])
        if gate:
            ax_GT = plt.axes([0.4, 0.8, 0.25, 0.175])
            ax_GT.bar(
                range(nbin),
                1000.0 * self.params["zone_mask"]["gate"],
                width=1,
                facecolor="tab:green",
                alpha=0.3,
            )
            ax_GT.set_ylim([0, 0.1])
            self.pl_dat.remove_frame(ax_GT, ["top", "right"])

        ax_RW = plt.axes([0.1, 0.8, 0.25, 0.175])
        ax_RW.bar(
            range(nbin),
            1000.0 * self.params["zone_mask"]["reward"],
            width=1,
            facecolor="tab:red",
            alpha=0.3,
        )
        ax_RW.set_ylim([0, 0.1])
        self.pl_dat.remove_frame(ax_RW, ["top", "right"])
        ax_RW.set_xlabel("position [bins]")

        ax_nRnG = plt.axes([0.7, 0.8, 0.25, 0.175])
        ax_nRnG.bar(
            range(nbin),
            1000.0 * self.params["zone_mask"]["others"],
            width=1,
            facecolor="tab:blue",
            alpha=0.3,
        )
        ax_nRnG.set_ylim([0, 0.1])
        self.pl_dat.remove_frame(ax_nRnG, ["top", "right"])
        ax_nRnG.set_xlabel("position [bins]")

        for j in range(n_int):
            col = [1, 0.2 * j, 0.2 * j]
            occ = (
                loc_stab_p[
                    s_arr[j] : s_arr[j + 1], self.params["zone_mask"]["reward"], :
                ]
                .sum(0)
                .sum(0)
            )
            occ /= occ.sum()
            ax_RW.plot(
                range(nbin),
                occ,
                "-",
                color=col,
                label="Sessions %d-%d" % (s_arr[j] + 1, s_arr[j + 1]),
            )
            # ax.bar(range(nbin),loc_stab[:20,self.params['zone_mask']['reward'],:].sum(0).sum(0),width=1,facecolor='k',alpha=0.5)

            col = [0.2 * j, 0.2 * j, 1]
            occ = (
                loc_stab_p[
                    s_arr[j] : s_arr[j + 1], self.params["zone_mask"]["others"], :
                ]
                .sum(0)
                .sum(0)
            )
            occ /= occ.sum()
            ax_nRnG.plot(range(nbin), occ, "-", color=col)
            # ax.bar(range(nbin),loc_stab[:20,self.params['zone_mask']['others'],:].sum(0).sum(0),width=1,facecolor='k',alpha=0.5)

            if gate:
                col = [0.2 * j, 0.8, 0.2 * j]
                occ = (
                    loc_stab_p[
                        s_arr[j] : s_arr[j + 1], self.params["zone_mask"]["gate"], :
                    ]
                    .sum(0)
                    .sum(0)
                )
                occ /= occ.sum()
                ax_GT.plot(range(nbin), occ, "-", color=col)
                # ax.bar(range(nbin),loc_stab[:20,self.params['zone_mask']['others'],:].sum(0).sum(0),width=1,facecolor='k',alpha=0.5)
        ax_RW.legend(fontsize=6, loc="upper left", bbox_to_anchor=[0.05, 1.1])
        props = dict(boxstyle="round", facecolor="w", alpha=0.8)

        for j in range(n_int):
            ax_im = plt.axes([0.1, 0.525 - j * 0.15, 0.15, 0.1])
            im = ax_im.imshow(
                gauss_smooth(p_rec_loc[j, ...], (1, 0)),
                clim=[0.25, 0.75],
                interpolation="None",
                origin="lower",
                aspect="auto",
            )
            plt.colorbar(im)
            ax_im.set_xlim([0.5, 10.5])
            ax_im.text(
                x=6,
                y=107,
                s="Sessions %d-%d" % (s_arr[j] + 1, s_arr[j + 1]),
                ha="left",
                va="bottom",
                bbox=props,
                fontsize=8,
            )
            ax_im.set_ylabel("pos.")

            ax = plt.axes([0.375, 0.525 - j * 0.15, 0.225, 0.1])
            for i, ds in enumerate([1, 3]):
                col = [0.35 * i, 0.35 * i, 0.35 * i]
                if j == 0:
                    ax_im.annotate(
                        "",
                        xy=(ds, 100),
                        xytext=(ds, 115),
                        fontsize=6,
                        annotation_clip=False,
                        arrowprops=dict(arrowstyle="->", color=col),
                    )
                ax.plot(gauss_smooth(p_rec_loc[j, :, ds], 1), color=col)
            ax.set_ylim([0, 1])
            ax.set_ylabel("$r_{s}$")
            self.pl_dat.remove_frame(ax, ["top", "right"])
            if j < (n_int - 1):
                ax_im.set_xticklabels([])
                ax.set_xticklabels([])
            else:
                ax_im.set_xlabel("$\Delta s$ [sessions]")
                ax.set_xlabel("position [bins]")

            # ax = plt.axes([0.725,0.525-j*0.15,0.25,0.1])

        status, status_dep = get_status_arr(self)

        # idx_c = np.where(self.status['clusters'])[0]

        # nC_good,nSes_good = status['act'].shape
        # ds_max = 1

        # need session average, not cluster average
        # fields = np.any(self.status["fields"][self.status['clusters'],...] & (self.fields['location'][self.status['clusters'],...,0]>self.params['zone_idx']['reward'][0]) & (self.fields['location'][self.status['clusters'],...,0]<self.params['zone_idx']['reward'][1]),2)
        # fields = np.any(self.status["fields"][self.status['clusters'],...] & (self.fields['location'][self.status['clusters'],...,0]<self.params['zone_idx']['reward'][0]) | (self.fields['location'][self.status['clusters'],...,0]>self.params['zone_idx']['reward'][1]),2)

        ax = plt.axes([0.7, 0.1, 0.25, 0.1])
        ax.plot(gauss_smooth(self.stats["p_post_s"]["act"]["act"][:, 1, 0], 1), "k")
        ax.plot(gauss_smooth(self.stats["p_post_s"]["code"]["code"][:, 1, 0], 1), "r")
        ax.plot(gauss_smooth(self.stats["p_post_s"]["stable"]["code"][:, 1, 0], 1), "b")
        ax.set_ylim([0, 1])

        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig("change_of_stability")

        plt.figure()
        p_rec_loc = np.zeros((nSes, nbin)) * np.NaN
        # for ds in range(1,min(nSes,41)):
        ds = 1
        session_bool = np.where(
            np.pad(self.status["sessions"][ds:], (0, ds), constant_values=False)
            & np.pad(self.status["sessions"][:], (0, 0), constant_values=False)
        )[0]

        s1_shifts, s2_shifts, f1, f2 = np.unravel_index(
            self.compare["pointer"].col,
            (
                nSes,
                nSes,
                self.params["field_count_max"],
                self.params["field_count_max"],
            ),
        )
        c_shifts = self.compare["pointer"].row
        sig = 6
        di = 3

        Ds = s2_shifts - s1_shifts
        idx = np.where(Ds == ds)[0]
        idx_shifts = self.compare["pointer"].data[idx].astype("int") - 1
        shifts = self.compare["shifts"][idx_shifts]

        s = s1_shifts[idx]
        f = f1[idx]
        c = c_shifts[idx]
        loc_shifts = np.round(self.fields["location"][c, s, f, 0]).astype("int")
        for s0 in np.where(session_bool)[0]:
            for i in range(nbin):
                i_min = max(0, i - di)
                i_max = min(nbin, i + di)
                idx_loc = (loc_shifts >= i_min) & (loc_shifts < i_max) & (s == s0)

                shifts_loc = shifts[idx_loc]
                N_data = len(shifts_loc)
                N_stable = (np.abs(shifts_loc) < (SD * sig)).sum()

                p_rec_loc[s0, i] = N_stable / N_data

        plt.subplot(212)
        ## find location specific stabilization
        RW_stab = np.nanmean(p_rec_loc[:, self.params["zone_mask"]["reward"]], 1)
        plt.plot(gauss_smooth(RW_stab, 1), color="tab:red")
        non_start = np.copy(self.params["zone_mask"]["others"])
        non_start[:13] = False
        nRnG_stab = np.nanmean(p_rec_loc[:, non_start], 1)
        plt.plot(gauss_smooth(nRnG_stab, 1), color="tab:blue")
        START_stab = np.nanmean(p_rec_loc[:, 15:35], 1)
        plt.plot(gauss_smooth(START_stab, 1), color="tab:green")

        plt.show(block=False)

        # maxSes = 20
        # print('what are those stable cells coding for?')
        # plt.figure(figsize=(5,2.5))
        #
        # col_arr = ['k',[0.5,1,0.5],[1,0.5,0.5],[0.5,0.5,1]]
        # label_arr = ['all','GT','RW','nRG']
        # key_arr = ['all','gate','reward','others']
        #
        # w_bar = 0.2
        # nKey = len(key_arr)
        # offset_bar = ((nKey+1)%2)*w_bar/2 + (nKey//2 - 1)*w_bar
        #
        # arr = np.arange(1,min(40,nSes),2)
        # for i,key in enumerate(key_arr):

        # plt.bar(arr-offset_bar+i*w_bar,p_rec[key][arr],width=w_bar,facecolor=col_arr[i],edgecolor='k',label=label_arr[i])
        # plt.errorbar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,self.loc_stability[key]['mean'][:maxSes,1],self.loc_stability[key]['std'][:maxSes,1],fmt='none',ecolor='r')

        # plt.xlabel('session difference $\Delta s$')
        # plt.ylabel('$\%$ stable fields')
        # plt.ylim([0,1.1])
        # plt.legend(loc='upper right',ncol=2)
        # plt.tight_layout()
        # plt.show(block=False)

    def plot_XY(self):

        nSes = self.data["nSes"]

        mode = "PC"
        # s_bool = np.zeros(self.data['nSes'],'bool')
        # s_bool[17:87] = True
        # s_bool[~self.status['sessions']] = False
        s_bool = self.status["sessions"]
        state_label = "alpha" if (mode == "act") else "beta"
        status_act = self.status["activity"][self.status["clusters"], :, 1]
        status_act = status_act[:, s_bool]
        # status_act = status_act[:,session_bool]
        status_PC = self.status["activity"][self.status["clusters"], :, 2]
        status_PC = status_PC[:, s_bool]
        nC_good, nSes_good = status_act.shape
        nSes_max = np.where(s_bool)[0][-1]

        active_neurons = status_act.mean(0)
        silent_neurons = (~status_act).mean(0)
        print(
            "active neurons: %.3g +/- %.3g"
            % (active_neurons.mean() * 100, active_neurons.std() * 100)
        )
        print(
            "silent neurons: %.3g +/- %.3g"
            % (silent_neurons.mean() * 100, silent_neurons.std() * 100)
        )

        coding_neurons = status_PC.sum(0) / status_act.sum(0)
        ncoding_neurons = ((~status_PC) & status_act).sum(0) / status_act.sum(0)
        print(
            "coding neurons: %.3g +/- %.3g"
            % (coding_neurons.mean() * 100, coding_neurons.std() * 100)
        )
        print(
            "non-coding neurons: %.3g +/- %.3g"
            % (ncoding_neurons.mean() * 100, coding_neurons.std() * 100)
        )

        p_act = np.count_nonzero(status_act) / (nC_good * nSes_good)
        p_PC = np.count_nonzero(status_PC) / np.count_nonzero(status_act)
        # print(p_PC)
        rnd_var_act = np.random.random(status_act.shape)
        rnd_var_PC = np.random.random(status_PC.shape)
        status_act_test = np.zeros(status_act.shape, "bool")
        status_act_test_rnd = np.zeros(status_act.shape, "bool")
        status_PC_test = np.zeros(status_PC.shape, "bool")
        status_PC_test_rnd = np.zeros(status_PC.shape, "bool")
        for c in range(nC_good):

            # status_act_test[c,:] = rnd_var_act[c,:] < (np.count_nonzero(status_act[c,:])/nSes_good)
            nC_act = status_act[c, :].sum()
            status_act_test[c, np.random.choice(nSes_good, nC_act, replace=False)] = (
                True
            )
            status_act_test_rnd[c, :] = rnd_var_act[c, :] < p_act

            status_PC_test[
                c,
                np.where(status_act[c, :])[0][
                    np.random.choice(nC_act, status_PC[c, :].sum(), replace=False)
                ],
            ] = True
            status_PC_test_rnd[c, status_act[c, :]] = (
                rnd_var_PC[c, status_act[c, :]] < p_PC
            )

        for mode in ["act", "code"]:

            fig = plt.figure(figsize=(3, 2), dpi=self.pl_dat.sv_opt["dpi"])
            status = status_act if mode == "act" else status_PC
            status_test = status_act_test if mode == "act" else status_PC_test

            recurr = np.zeros((nSes_good, nSes_good)) * np.NaN
            N_active = status_act.sum(0)

            for s in range(nSes_good):  # np.where(s_bool)[0]:
                overlap = status[status[:, s], :].sum(0).astype("float")
                N_ref = (
                    N_active if mode == "act" else status_act[status_PC[:, s], :].sum(0)
                )
                recurr[s, 1 : nSes_good - s] = (overlap / N_ref)[s + 1 :]

            recurr_test = np.zeros((nSes_good, nSes_good)) * np.NaN
            N_active_test = status_test.sum(0)
            tmp = []
            for s in range(nSes_good):
                # overlap_act_test = status_test[status_test[:,s],:].sum(0).astype('float')
                overlap_test = status_test[status_test[:, s], :].sum(0).astype("float")
                N_ref = (
                    N_active_test
                    if mode == "act"
                    else status_act_test[status_PC_test[:, s], :].sum(0)
                )
                recurr_test[s, 1 : nSes_good - s] = (overlap_test / N_ref)[s + 1 :]
                if (~np.isnan(recurr_test[s, :])).sum() > 1:
                    tmp.append(recurr_test[s, ~np.isnan(recurr_test[s, :])])

            rec_mean = np.nanmean(np.nanmean(recurr, 0))
            rec_var = np.sqrt(np.nansum(np.nanvar(recurr, 0)) / (recurr.shape[1] - 1))

            ax = plt.axes([0.2, 0.3, 0.75, 0.65])
            # pl_dat.add_number(fig,ax,order=4,offset=[-250,250])

            p = status.sum() / (nSes_good * nC_good)

            # ax.plot([0,self.data['nSes']],[p,p],'k--')
            # ax.text(10,p+0.05,'$p^{(0)}_{\\%s^+}$'%(state_label),fontsize=8)
            SD = 1
            # ax.plot([1,nSes_good],[rec_mean,rec_mean],'k--',linewidth=0.5)
            recurr[:, 0] = 1
            # ax.plot([0,1],[1,np.nanmean(recurr[:,0])],'-k')
            self.pl_dat.plot_with_confidence(
                ax,
                np.linspace(0, nSes_good - 1, nSes_good),
                np.nanmean(recurr, 0),
                SD * np.nanstd(recurr, 0),
                col="k",
                ls="-",
                label="emp. data",
            )
            # pl_dat.plot_with_confidence(ax,np.linspace(0,nSes_good-1,nSes_good),np.nanmean(recurr_test,0),SD*np.nanstd(recurr_test,0),col='tab:red',ls='-',label='rnd. data')
            ax.set_ylim([0, 1.1])
            ax.set_xlabel("$\Delta$ sessions")
            if mode == "act":
                ax.set_ylabel("act. recurr.")
            else:
                ax.set_ylabel("code recurr.")
            # ax.set_ylabel('$p(\\%s^+_{s+\Delta s} | \\%s^+_s)$'%(state_label,state_label))#'p(recurr.)')
            ax.set_xlim([0, nSes_good])
            ax.legend(
                fontsize=8, loc="upper right", bbox_to_anchor=[0.9, 1], handlelength=1
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout()
            plt.show(block=False)

            # if sv:
            #     pl_dat.save_fig('defense_%s_nornd_recurr'%mode)

        SD = 1.96
        fig = plt.figure(figsize=(3, 2), dpi=self.pl_dat.sv_opt["dpi"])
        ax = plt.axes([0.2, 0.3, 0.75, 0.65])
        N_data = np.zeros(self.data["nSes"]) * np.NaN

        D_KS = np.zeros(self.data["nSes"]) * np.NaN
        N_stable = np.zeros(self.data["nSes"]) * np.NaN
        N_total = (
            np.zeros(self.data["nSes"]) * np.NaN
        )  ### number of PCs which could be stable
        # fig = plt.figure()
        p_rec_alt = np.zeros(self.data["nSes"]) * np.NaN

        s1_shifts, s2_shifts, f1, f2 = np.unravel_index(
            self.compare["pointer"].col,
            (
                self.data["nSes"],
                nSes,
                self.params["field_count_max"],
                self.params["field_count_max"],
            ),
        )
        c_shifts = self.compare["pointer"].row

        for ds in range(1, self.data["nSes"]):  # min(self.data['nSes'],30)):
            Ds = s2_shifts - s1_shifts
            idx_ds = np.where((Ds == ds) & s_bool[s1_shifts] & s_bool[s2_shifts])[0]
            N_data[ds] = len(idx_ds)

            idx_shifts = self.compare["pointer"].data[idx_ds].astype("int") - 1
            shifts = self.compare["shifts"][idx_shifts]
            N_stable[ds] = (
                np.abs(shifts) < (SD * self.stability["all"]["mean"][0, 2])
            ).sum()

            p_rec_alt[ds] = N_stable[ds] / N_data[ds]

        p_rec_alt[0] = 1
        ax.plot(range(self.data["nSes"]), p_rec_alt, "-", color="k")
        # ax.plot(0,1,'ok')
        r_random = 2 * SD * self.stability["all"]["mean"][0, 2] / 100
        ax.plot(
            [1, self.data["nSes"]],
            [r_random, r_random],
            "--",
            color="tab:red",
            linewidth=1,
        )
        ax.set_ylim([0, 1.1])
        ax.set_xlim([0, nSes_good])
        ax.set_ylabel("place field recurr.", fontsize=12)
        ax.set_xlabel("$\Delta$ sessions")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.show(block=False)

        if sv:
            pl_dat.save_fig("defense_pf_recurr")

        p_shift = np.zeros(nbin)
        for s in np.where(s_bool)[0]:
            idx_field = np.where(self.status["fields"][:, s, :])
            for c, f in zip(idx_field[0], idx_field[1]):
                roll = round(
                    (-self.fields["location"][c, s, f, 0] + self.data["nbin"] / 2)
                    / L_track
                    * self.data["nbin"]
                )
                p_shift += np.roll(self.fields["p_x"][c, s, f, :], roll)
        p_shift /= p_shift.sum()

        PC_idx = np.where(self.status["activity"][..., 2])
        N_data = len(PC_idx[0])

        p_ds0, p_cov = fit_shift_model(p_shift)

        p = self.stability
        fig = plt.figure(figsize=(3, 1.5), dpi=300)
        ax = plt.axes([0.2, 0.3, 0.75, 0.65])

        ax.plot(
            [0, self.data["nSes"]],
            [p_ds0[2], p_ds0[2]],
            linestyle="--",
            color=[0.6, 0.6, 0.6],
        )
        ax.text(10, p_ds0[2] + 1, "$\sigma_0$", fontsize=8)

        sig_theta = self.stability["all"]["mean"][0, 2]

        self.pl_dat.plot_with_confidence(
            ax,
            range(1, nSes + 1),
            p["all"]["mean"][:, 2],
            p["all"]["CI"][..., 2].T,
            "k",
            "-",
        )

        ax.set_ylim([0, 12])
        ax.set_xlim([0, nSes_good])
        ax.set_ylabel("$\sigma_{\Delta \\theta}$", fontsize=12)
        ax.set_xlabel("$\Delta$ sessions")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.show(block=False)
        if sv:
            pl_dat.save_fig("defense_sig_shift")

    def plot_parameter_dependent_neuron_numbers(self):

        # need:
        # neurons detected:   SNR, rval, CNN, (p_m) <- stats
        # place fields:       Bayes_factor, reliability, A0, A, pmass <- fields
        print("### plot test of thresholds for neuron number ###")

        nSes = self.data["nSes"]

        fig = plt.figure(figsize=(7, 2.5), dpi=300)
        ax_SNR = plt.axes([0.125, 0.2, 0.225, 0.65])
        ax_rval = plt.axes([0.4, 0.2, 0.225, 0.65])
        ax_pm = plt.axes([0.675, 0.2, 0.225, 0.65])
        # self.pl_dat.add_number(fig,ax,order=1)

        def calculate_neuron_numbers(ax, key, vals, val_thr):
            ## method to calculate and plot neuron and PC numbers
            ## for different parameter thresholds

            width = np.diff(vals)[0] * 0.4
            nROI = np.zeros((len(vals), nSes))
            nPC = np.zeros((len(vals), nSes))

            ## iterate through parameter values
            for i, val in enumerate(vals):

                thresholds[key + "_thr"] = val

                self.update_status(**thresholds)
                nROI[i, :] = (self.status["activity"][..., 1]).sum(axis=0)
                nPC[i, :] = (self.status["activity"][..., 2]).sum(axis=0)

                ax[0].scatter(
                    val
                    - width / 2
                    + width * np.random.rand(nSes)[self.status["sessions"]],
                    nROI[i, self.status["sessions"]],
                    s=2,
                    c=[[0.8, 0.8, 0.8]],
                )
                ax[0].scatter(
                    val
                    - width / 2
                    + width * np.random.rand(nSes)[self.status["sessions"]],
                    nPC[i, self.status["sessions"]],
                    s=2,
                    c=[[0.8, 0.8, 1]],
                )

            ax[0].plot(
                vals,
                nROI[:, self.status["sessions"]].mean(1),
                "k^",
                markersize=4,
                markeredgewidth=0.5,
                label="neurons",
            )  # ,label='neurons ($p_m\geq0.05$)')
            ax[0].plot(
                vals,
                nPC[:, self.status["sessions"]].mean(1),
                "^",
                color="tab:blue",
                markersize=4,
                markeredgewidth=0.5,
                label="PCs",
            )

            self.pl_dat.remove_frame(ax[0], ["top"])
            ax[0].axvline(val_thr, color="k", linestyle="--")
            ax[0].legend(fontsize=10, bbox_to_anchor=[0.1, 1.15], loc="upper left")

            ax[1].plot(
                vals,
                nPC[:, self.status["sessions"]].mean(1)
                / nROI[:, self.status["sessions"]].mean(1),
                "r-",
                linewidth=0.5,
            )

            ax[1].set_ylim([0, 0.54])
            self.pl_dat.remove_frame(ax[1], ["top"])

        thresholds = {
            "SNR_thr": 0.5,
            "rval_thr": 0.5,
            "CNN_thr": 0.6,
            "A0_thr": 1,
            "A_thr": 3,
            "pm_thr": 0.95,
            "Bayes_thr": 10,
            "reliability_thr": 0.1,
        }

        ylim = [0, 500]
        ax_SNR = [ax_SNR, ax_SNR.twinx()]
        calculate_neuron_numbers(ax_SNR, "SNR", np.linspace(1, 10, 10), 3.0)
        thresholds["SNR_thr"] = 3.0
        plt.setp(ax_SNR[0], ylim=ylim, xlabel=r"$\Theta_{SNR}$", ylabel="# neurons")
        # plt.setp(ax_SNR[1],yticklabels=[])

        ax_rval = [ax_rval, ax_rval.twinx()]
        calculate_neuron_numbers(ax_rval, "rval", np.linspace(-1, 1, 11), 0.0)
        thresholds["rval_thr"] = 0.0
        plt.setp(
            ax_rval[0],
            ylim=ylim,
            yticklabels=[],
            xlabel=r"$\Theta_{r}$",
        )
        plt.setp(ax_SNR[1], yticklabels=[])

        ax_pm = [ax_pm, ax_pm.twinx()]
        calculate_neuron_numbers(ax_pm, "pm", np.linspace(0, 1, 11), 0.95)
        thresholds["pm_thr"] = 0.95
        plt.setp(
            ax_pm[0],
            ylim=ylim,
            yticklabels=[],
            xlabel=r"$\Theta_{p^*}$",
        )
        plt.setp(ax_pm[1], ylabel="% place cells")

        plt.tight_layout()
        plt.show(block=False)

        # if sv:
        #     pl_dat.save_fig('neuronNumbers_test')
        # return

        print("whats with MI of place cells, only?")
        ### get SNR dependence of detected neurons
        # idx_other = (self.stats['r_values'] > 0) & (self.matching['score'][...,0]>0.05)
        # idx_other_certain = (self.stats['r_values'] > 0) & (self.matching['score'][...,0]>0.95)
        SNR_arr = np.linspace(2, 20, 5)
        MI = np.zeros(len(SNR_arr))
        # nPC = np.zeros(SNR_arr.shape + (nSes,2))
        plt.figure(figsize=(4, 2.5))
        width = 0.6
        ax = plt.axes([0.2, 0.2, 0.65, 0.65])
        for i, SNR_thr in enumerate(SNR_arr):
            idx = (
                (self.stats["SNR_comp"] >= (SNR_thr - 0.5))
                & (self.stats["SNR_comp"] < (SNR_thr + 0.5))
                & ~np.isnan(self.stats["MI_value"])
            )  # & idx_other
            # print(self.stats['MI_value'][idx])
            MI[i] = np.nanmean(self.stats["MI_value"][idx], axis=0)
            # idx = (self.stats['SNR_comp'] >= (SNR_thr-0.5)) & (self.stats['SNR_comp'] < (SNR_thr+0.5)) & idx_other_certain
            # MI[i,1] = self.stats['MI_value'][idx].mean(0)

            ax.boxplot(
                self.stats["MI_value"][idx],
                positions=[SNR_thr],
                widths=width,
                whis=[5, 95],
                notch=True,
                bootstrap=100,
                flierprops=dict(
                    marker=".",
                    markeredgecolor="None",
                    markerfacecolor=[0.5, 0.5, 0.5],
                    markersize=2,
                ),
            )

        ax.plot(SNR_arr, MI, "k^", markersize=5, label="neurons")

        # ax.set_ylim([0,0.6])
        ax.set_xlabel("$\Theta_{SNR}$")
        ax.set_ylabel("MI")
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.show(block=False)

        # if sv:
        #     pl_dat.save_fig('MI_SNR')

    def plot_hierarchy_interaction(self):

        print("### plot interaction of different dynamical hierarchies ###")

        nSes = self.data["nSes"]
        nC = self.data["nC"]
        nbin = self.data["nbin"]

        SD = 1
        sig_theta = self.stability["all"]["mean"][0, 2]
        ### find, if coding sessions usually belong to a period of longer activity
        status_La = np.zeros((nC, nSes, 2), "int")
        status_Lb = np.zeros_like(self.status["activity"][..., 2], "int")

        highCode_thr = 0.5
        IPI_La = np.zeros(nSes)
        IPI_Lb = np.zeros(nSes)
        La_highCode = np.zeros(nSes)

        IPI_La_start = np.zeros_like(self.status["activity"][..., 1], "bool")
        IPI_Lb_start = np.zeros_like(self.status["activity"][..., 2], "bool")

        idx_fields = np.where(self.status["fields"])

        for c in range(nC):
            s0_act = 0
            s0_code = 0
            inAct = False
            inCode = False
            for s in np.where(self.status["sessions"])[0]:
                if inAct:
                    if ~self.status["activity"][c, s, 1]:
                        La = self.status["sessions"][s0_act:s].sum()
                        status_La[c, s0_act:s, 0] = La
                        status_La[c, s0_act:s, 1] = self.status["activity"][
                            c, s0_act:s, 2
                        ].sum()
                        if (
                            self.status["activity"][c, s0_act:s, 2].sum() / La
                        ) > highCode_thr:
                            La_highCode[La] += 1
                        IPI_La[La] += 1
                        inAct = False
                else:
                    if self.status["activity"][c, s, 1]:
                        s0_act = s
                        inAct = True
                        IPI_La_start[c, s] = True

                if inCode:
                    if ~self.status["activity"][c, s, 2]:
                        Lb = self.status["sessions"][s0_code:s].sum()
                        status_Lb[c, s0_code:s] = Lb
                        IPI_Lb[Lb] += 1
                        inCode = False
                else:
                    if self.status["activity"][c, s, 2]:
                        s0_code = s
                        inCode = True
                        IPI_Lb_start[c, s] = True

            if inAct:
                La = self.status["sessions"][s0_act : s + 1].sum()
                status_La[c, s0_act : s + 1, 0] = La
                status_La[c, s0_act : s + 1, 1] = self.status["activity"][
                    c, s0_act : s + 1, 2
                ].sum()
                if (self.status["activity"][c, s0_act:s, 2].sum() / La) > highCode_thr:
                    La_highCode[La] += 1
                IPI_La[La] += 1
            if inCode:
                Lb = self.status["sessions"][s0_code : s + 1].sum()
                status_Lb[c, s0_code : s + 1] = Lb
                IPI_Lb[Lb] += 1

        status_La[:, ~self.status["sessions"], :] = 0
        status_Lb[:, ~self.status["sessions"]] = 0
        L_code = status_La[self.status["activity"][..., 2], 0]

        mean_La = np.zeros((nSes, 2)) * np.NaN
        mean_Lb = np.zeros((nSes, 2)) * np.NaN
        for s in np.where(self.status["sessions"])[0]:
            mean_La[s, 0] = np.nanmean(status_La[status_La[:, s, 0] > 0, s, 0])
            mean_La[s, 1] = np.nanstd(status_La[status_La[:, s, 0] > 0, s, 0])

            mean_Lb[s, 0] = np.nanmean(status_Lb[status_Lb[:, s] > 0, s])
            mean_Lb[s, 1] = np.nanstd(status_Lb[status_Lb[:, s] > 0, s])

        plt.figure()
        ax = plt.subplot(111)
        self.pl_dat.plot_with_confidence(
            ax, np.arange(nSes), mean_La[:, 0], mean_La[:, 1], "k"
        )
        self.pl_dat.plot_with_confidence(
            ax, np.arange(nSes), mean_Lb[:, 0], mean_Lb[:, 1], "r"
        )
        plt.show(block=False)

        status_stable = np.zeros_like(self.status["activity"][..., 2], "int")
        for c in range(nC):
            for s in np.where(self.status["sessions"])[0]:
                if self.status["activity"][c, s, 2]:
                    ds_ref = np.inf
                    idxes = (idx_fields[0] == c) & (idx_fields[1] < s)
                    idx_s = idx_fields[1][idxes]

                    for f in np.where(self.status["fields"][c, s, :])[0]:
                        dLoc = np.abs(
                            np.mod(
                                self.fields["location"][c, s, f, 0]
                                - self.fields["location"][
                                    c, idx_fields[1][idxes], idx_fields[2][idxes], 0
                                ]
                                + nbin / 2,
                                nbin,
                            )
                            - nbin / 2
                        )

                        stable_s = idx_s[np.where(dLoc < (SD * sig_theta))[0]]
                        if len(stable_s) > 0:
                            ds_ref = np.min(s - stable_s)
                            status_stable[c, s] = min(s - stable_s[-1], ds_ref)

        # print(p_pre['stable_code'][:,ds,0,:])
        # res = sstats.ttest_ind_from_stats(np.nanmean(p_pre[:,0,0]),np.nanstd(p_pre[:,0,0]),(~np.isnan(p_pre[:,0,0])).sum(),np.nanmean(p_pre[:,1,0]),np.nanstd(p_pre[:,1,0]),(~np.isnan(p_pre[:,1,0])).sum(),equal_var=True)
        # print(res)
        # res = sstats.ttest_ind_from_stats(np.nanmean(p_pre[:,0,1]),np.nanstd(p_pre[:,0,1]),(~np.isnan(p_pre[:,0,1])).sum(),np.nanmean(p_pre[:,1,1]),np.nanstd(p_pre[:,1,1]),(~np.isnan(p_pre[:,1,1])).sum(),equal_var=True)
        # print(res)

        # print('sessions before and after coding are more probable to be active')

        # res = sstats.ttest_ind_from_stats(np.nanmean(p_pre[:,0,0]),np.nanstd(p_pre[:,0,0]),(~np.isnan(p_pre[:,1,1])).sum(),np.nanmean(p_pre[:,1,1]),np.nanstd(p_pre[:,1,1]),(~np.isnan(p_pre[:,1,1])).sum(),equal_var=True)
        # print(res)
        # print(p_post)

        fig = plt.figure(figsize=(7, 6), dpi=self.pl_dat.sv_opt["dpi"])

        plt.figtext(0.15, 0.8, "activation", fontsize=14)

        ax = plt.axes([0.1, 0.5, 0.08, 0.15])

        ax = plt.axes([0.4, 0.8, 0.175, 0.15])
        print(IPI_La)
        print(IPI_La.shape)
        La_sessions = IPI_La * np.arange(len(IPI_La))
        count = np.zeros(nSes)
        for item in Counter(status_La[self.status["activity"][..., 2], 0]).items():
            count[item[0]] = item[1]

        nAct = self.status["activity"][..., 1].sum(1)
        nPC = self.status["activity"][..., 2].sum(1)
        rate = nPC / nAct
        mean_r = np.zeros((nSes, 3)) * np.NaN
        tmp = []
        print("get CI from bootstrapping")
        for i in range(1, nSes):
            if np.any(nAct == i):
                mean_r[i, 0] = rate[nAct == i].mean()
                mean_r[i, 1:] = np.percentile(rate[nAct == i], [15.8, 84.1])

        c_pa = self.status["activity"][..., 1].sum(1) / self.status["sessions"].sum()
        c_pb = self.status["activity"][..., 2].sum(1) / self.status["activity"][
            ..., 1
        ].sum(1)

        # print(c_pb[self.status['clusters']])
        y0, res, rank, tmp = np.linalg.lstsq(
            self.status["activity"][self.status["clusters"], :, 1],
            c_pb[self.status["clusters"]],
        )
        # print(y0)

        # ax.plot(c_pb[self.status['clusters']])
        # ax.plot(self.status['activity'][...,1].sum(1)+0.5*np.random.rand(nC),c_pb+0.01*np.random.rand(nC),'r.',markersize=1.5,markeredgecolor='none')
        pb = np.nanmean(
            self.status["activity"][self.status["clusters"], :, 2].sum(0)
            / self.status["activity"][self.status["clusters"], :, 1].sum(0)
        )

        ax.plot([0, 80], [pb, pb], "k--")
        ax.plot(
            gauss_smooth(count[: len(IPI_La)] / La_sessions, 1),
            label="$p(\\beta^+| \in \mathcal{L}_{\\alpha})$",
        )
        self.pl_dat.plot_with_confidence(
            ax,
            range(nSes),
            mean_r[:, 0],
            mean_r[:, 1:].T,
            col="r",
            label="$p(\\beta^+| \in N_{\\alpha})$",
        )

        ax.set_xlim([0, 70])
        ax.set_ylim([0, 0.5])
        ax.set_ylabel("p")
        ax.set_xlabel("$N_{\\alpha} / \mathcal{L}_{\\alpha}$")
        ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=[1.2, 1.3])
        self.pl_dat.remove_frame(ax, ["top", "right"])
        # plt.hist(status_La[self.status['activity'][...,2],0],np.linspace(0,80,81),color='tab:blue',alpha=0.5,density=True,cumulative=True,histtype='step')
        # plt.hist(status_La[self.status['activity'][...,1]&(~self.status['activity'][...,2]),0],np.linspace(0,80,81),color='tab:red',alpha=0.5,density=True,cumulative=True,histtype='step')

        # res = sstats.f_oneway(*tmp)
        # print(res)

        # ax.plot(nAct+0.7*np.random.rand(nC),nPC/nAct+0.03*np.random.rand(nC),'k.',markersize=1,markeredgecolor='none')
        # ax.plot(range(nSes),mean_r,'r-')
        # ax = plt.axes([0.7,0.7,0.25,0.25])

        plt.figtext(0.45, 0.65, "coding", fontsize=14)
        plt.figtext(0.75, 0.25, "  field \nstability", fontsize=14)
        #
        # plt.show(block=False)

        ### coding -> activity

        p_rec = {
            "all": np.zeros(nSes) * np.NaN,
            "cont": np.zeros(nSes) * np.NaN,
            "mix": np.zeros(nSes) * np.NaN,
            "discont": np.zeros(nSes) * np.NaN,
            "silent_mix": np.zeros(nSes) * np.NaN,
            "silent": np.zeros(nSes) * np.NaN,
        }

        key_arr = ["all", "cont", "mix", "discont", "silent_mix", "silent"]
        s1_shifts, s2_shifts, f1, f2 = np.unravel_index(
            self.compare["pointer"].col, (nSes, nSes, 5, 5)
        )
        Ds = s2_shifts - s1_shifts

        ### coding -> field stability
        for ds in range(1, nSes):
            session_bool = np.where(
                np.pad(self.status["sessions"][ds:], (0, ds), constant_values=False)
                & np.pad(self.status["sessions"][:], (0, 0), constant_values=False)
            )[0]

            # for s1 in session_bool:
            idx_ds = np.where(Ds == ds)[0]

            idx_shifts = self.compare["pointer"].data[idx_ds].astype("int") - 1
            shifts = self.compare["shifts"][idx_shifts]

            for pop in key_arr:

                if pop == "all":
                    idxes = np.ones(len(idx_ds), "bool")
                elif pop == "cont":
                    idxes = self.compare["inter_coding"][idx_ds, 1] == 1
                elif pop == "mix":
                    idxes = (
                        (self.compare["inter_coding"][idx_ds, 1] > 0)
                        & (self.compare["inter_coding"][idx_ds, 1] < 1)
                    ) & (self.compare["inter_active"][idx_ds, 1] == 1)
                elif pop == "discont":
                    idxes = (self.compare["inter_coding"][idx_ds, 1] == 0) & (
                        self.compare["inter_active"][idx_ds, 1] == 1
                    )
                # elif pop=='silent_mix':
                # idxes =(self.compare['inter_active'][idx_ds,1]>0) & (self.compare['inter_active'][idx_ds,1]<1)
                elif pop == "silent":
                    idxes = self.compare["inter_active"][idx_ds, 1] < 1

                N_data = idxes.sum()
                N_stable = (np.abs(shifts[idxes]) < (SD * sig_theta)).sum()

                p_rec[pop][ds] = N_stable / N_data

        ax = plt.axes([0.7, 0.5, 0.25, 0.15])

        maxSes = 10
        col_arr = [[0.5, 0.5, 1], [0.5, 0.5, 0.5], [1, 0.5, 0.5], [0.5, 1, 0.5]]
        label_arr = ["continuous", "mixed", "non-coding", "silent"]
        # key_arr = ['cont','mix','discont','silent']
        key_arr = ["cont", "mix", "discont", "silent"]

        w_bar = 0.2
        nKey = len(key_arr)
        offset_bar = ((nKey + 1) % 2) * w_bar / 2 + ((nKey - 1) // 2) * w_bar

        for i, key in enumerate(key_arr):
            # ax.bar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,self.stability[key]['mean'][:maxSes,1],width=w_bar,facecolor=col_arr[i],edgecolor='k',label=label_arr[i])
            # ax.errorbar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,self.stability[key]['mean'][:maxSes,1],self.stability[key]['std'][:maxSes,1],fmt='none',ecolor='r')
            ax.bar(
                np.arange(maxSes) - offset_bar + i * w_bar,
                p_rec[key][:maxSes],
                width=w_bar,
                facecolor=col_arr[i],
                edgecolor="k",
                label=label_arr[i],
            )
            # ax.errorbar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,p_rec[key][:maxSes],self.stability[key]['std'][:maxSes,1],fmt='none',ecolor='r')
        p_rec_chance = (2 * SD * sig_theta) / 100
        ax.plot(
            [0.5, maxSes + 0.5],
            [p_rec_chance, p_rec_chance],
            "--",
            color=[0.5, 0.5, 0.5],
        )
        ax.set_xlabel("session difference $\Delta s$")
        ax.set_ylabel("$p(\\gamma_{\Delta s}^+)$")
        ax.set_xlim([0.5, maxSes - 0.5])
        ax.set_ylim([0, 1.1])
        ax.legend(loc="upper right", bbox_to_anchor=[0.7, 1.25], fontsize=8, ncol=2)
        self.pl_dat.remove_frame(ax, ["top", "right"])

        # ax_sig = ax.twinx()
        # for j,p in enumerate(key_arr):
        #     pl_dat.plot_with_confidence(ax_sig,range(1,nSes+1),self.stability[p]['mean'][:,2],self.stability[p]['CI'][:,:,2].T,col=col_arr[j])

        # ax_sig.set_ylim([0,100/(2*SD)])
        # pl_dat.remove_frame(ax_sig,['top'])
        # plt.show(block=False)

        ### field stability -> coding
        print(
            "find neurons, which are stable and get statistics of reactivation, recoding, enhanced prob of coding, ..."
        )

        ### activity -> field stability
        ax = plt.axes([0.7, 0.8, 0.25, 0.15])
        # maxSes = 6
        col_arr = [[1, 0.5, 0.5], [0.5, 1, 0.5]]
        label_arr = ["non-coding", "silent"]
        # key_arr = ['cont','mix','discont','silent']
        key_arr = ["discont", "silent"]

        w_bar = 0.2
        nKey = len(key_arr)
        offset_bar = ((nKey + 1) % 2) * w_bar / 2 + ((nKey - 1) // 2) * w_bar
        offset_bar = 0.1
        for i, key in enumerate(key_arr):
            # ax.bar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,self.stability[key]['mean'][:maxSes,1],width=w_bar,facecolor=col_arr[i],edgecolor='k',label=label_arr[i])
            # ax.errorbar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,self.stability[key]['mean'][:maxSes,1],self.stability[key]['std'][:maxSes,1],fmt='none',ecolor='r')
            ax.bar(
                np.arange(maxSes) - offset_bar + i * w_bar,
                p_rec[key][:maxSes],
                width=w_bar,
                facecolor=col_arr[i],
                edgecolor="k",
                label=label_arr[i],
            )

        ax.plot(
            [0.5, maxSes + 0.5],
            [p_rec_chance, p_rec_chance],
            "--",
            color=[0.5, 0.5, 0.5],
        )
        ax.set_xlabel("session difference $\Delta s$")
        ax.set_ylabel("$p(\\gamma_{\Delta s}^+)$")
        ax.set_ylim([0, 1.1])
        ax.set_xlim([0.5, maxSes - 0.5])
        ax.legend(loc="upper right", bbox_to_anchor=[0.7, 1.25], fontsize=8)
        self.pl_dat.remove_frame(ax, ["top", "right"])

        # ax_sig = ax.twinx()
        # for j,p in enumerate(key_arr):
        #     pl_dat.plot_with_confidence(ax_sig,range(nSes),self.stability[p]['mean'][:,2],self.stability[p]['CI'][:,:,2].T,col=col_arr[j])
        # ax_sig.set_ylim([0,100/(2*SD)])
        # pl_dat.remove_frame(ax_sig,['top'])

        plt.tight_layout
        plt.show(block=False)

        # if sv:
        #     self.pl_dat.save_fig('dynamics_interaction')

        # plt.figure()
        # for ds in range(1,ds_max):
        #     plt.hist(N_stable[:,ds],np.linspace(0,30,31),alpha=0.5,histtype='step',color=[0.2*ds,0.2*ds,0.2*ds],cumulative=True,density=True)
        # plt.show(block=False)

    def plot_transition_probs(self, N_bs=100):

        plt.figure(figsize=(7, 3), dpi=self.pl_dat.sv_opt["dpi"])
        ds_max = 7

        def plot_transitions(ax, state_pre, state_post, state_ref, label_str):

            ## probability of being active after coding
            ax[0].axhline(
                np.nanmean(
                    self.stats["p_post_s"][state_post][state_ref][
                        self.status["sessions"], 1, 0
                    ],
                    0,
                ),
                color="k",
                ls="--",
                lw=0.5,
            )

            p_bs = np.zeros((2, 2))
            for p_key, sign, col in zip(
                ["p_pre_s", "p_post_s"], ["-", "+"], ["tab:blue", "tab:red"]
            ):
                p_bs[0, 0], p_bs[0, 1] = bootstrap_data(
                    np.nanmean,
                    self.stats[p_key][state_pre][state_post][
                        self.status["sessions"], 1, 0
                    ],
                    N_bs,
                )
                p_bs[1, 0], p_bs[1, 1] = bootstrap_data(
                    np.nanmean,
                    self.stats[p_key][state_pre][state_post][
                        self.status["sessions"], 1, 1
                    ],
                    N_bs,
                )
                ax[0].errorbar(
                    [1, 2],
                    p_bs[:, 0],
                    p_bs[:, 1],
                    fmt="-",
                    color=col,
                    linewidth=0.5,
                    label=label_str(sign, 1, "\Delta s"),
                )

            plt.setp(
                ax[0],
                xticks=[1, 2],
                xticklabels=["$\\beta^+$", "$\\beta^-$"],
                yticks=np.linspace(0, 1, 3),
                ylim=[0, 1],
                ylabel="$p$",
            )
            self.pl_dat.remove_frame(ax[0], ["top", "right"])
            # ax.legend(fontsize=8,loc='lower left')

            if len(ax) > 2:
                ax[2].axhline(
                    np.nanmean(
                        self.stats["p_post_s"]["code"]["act"][
                            self.status["sessions"], 1, 0
                        ],
                        0,
                    ),
                    color="k",
                    ls="--",
                    linewidth=0.5,
                )
                p_bs = np.full((ds_max, 2, 2), np.NaN)

                for p_key, sign, col in zip(
                    ["p_pre_s", "p_post_s"], ["-", "+"], ["tab:blue", "tab:red"]
                ):
                    for ds in range(1, ds_max):
                        # # p_pre_bs[ds,0,:] = bootstrap_data(np.nanmean,p_pre['stable_act'][:,ds,0,0,0],N_bs)
                        # # p_pre_bs[ds,1,:] = bootstrap_data(np.nanmean,p_pre['stable_act'][:,ds,1,0,0],N_bs)
                        # p_pre_bs[ds,0,0],p_pre_bs[ds,0,1] = bootstrap_data(np.nanmean,self.stats['p_pre_s']['stable']['act'][self.status['sessions'],ds,0,1],N_bs)
                        # p_pre_bs[ds,1,0],p_pre_bs[ds,1,1] = bootstrap_data(np.nanmean,self.stats['p_pre_s']['stable']['act'][self.status['sessions'],ds,1,1],N_bs)

                        p_bs[ds, 0, 0], p_bs[ds, 0, 1] = bootstrap_data(
                            np.nanmean,
                            self.stats[p_key][state_pre][state_post][
                                self.status["sessions"], ds, 0
                            ],
                            N_bs,
                        )
                        p_bs[ds, 1, 0], p_bs[ds, 1, 1] = bootstrap_data(
                            np.nanmean,
                            self.stats[p_key][state_pre][state_post][
                                self.status["sessions"], ds, 1
                            ],
                            N_bs,
                        )
                    ax[2].errorbar(
                        range(ds_max),
                        p_bs[:, 0, 0],
                        p_bs[:, 0, 1],
                        fmt="-",
                        color=col,
                        linewidth=0.5,
                        label=label_str(sign, "\Delta s", 1),
                    )

                # ax.errorbar(range(ds_max),p_pre_bs[:,0,0],p_pre_bs[:,0,1],fmt='-',color='tab:blue',linewidth=0.5,label='$p(\\alpha_{s-1}^+|\\gamma_{\Delta s,s}^{+})$')
                self.pl_dat.remove_frame(ax[2], ["top", "right"])
                plt.setp(
                    ax[2],
                    yticks=np.linspace(0, 1, 3),
                    ylim=[0, 1],
                    xticklabels=[],
                )
                # ax.set_xlabel('$\Delta s$')
                ax[2].legend(fontsize=8, loc="lower right", bbox_to_anchor=[1.4, 0])

            ax[1].axhline(
                np.nanmean(
                    self.stats["p_post_s"][state_ref][state_post][
                        self.status["sessions"], 1, 0
                    ],
                    0,
                ),
                c="k",
                ls="--",
                lw=0.5,
            )

            p_bs = np.full((ds_max, 2, 2), np.NaN)

            for p_key, sign, col in zip(
                ["p_pre_s", "p_post_s"], ["-", "+"], ["tab:blue", "tab:red"]
            ):

                for ds in range(1, ds_max):
                    p_bs[ds, 0, 0], p_bs[ds, 0, 1] = bootstrap_data(
                        np.nanmean,
                        self.stats[p_key][state_pre][state_post][
                            self.status["sessions"], ds, 0
                        ],
                        N_bs,
                    )
                    p_bs[ds, 1, 0], p_bs[ds, 1, 1] = bootstrap_data(
                        np.nanmean,
                        self.stats[p_key][state_pre][state_post][
                            self.status["sessions"], ds, 1
                        ],
                        N_bs,
                    )

                ax[1].errorbar(
                    range(ds_max),
                    p_bs[:, 0, 0],
                    p_bs[:, 0, 1],
                    fmt="-",
                    color=col,
                    linewidth=0.5,
                    label=label_str(sign, 1, "\Delta s"),
                )

            plt.setp(
                ax[1],
                yticks=[],
                ylim=[0, 1],
            )
            self.pl_dat.remove_frame(ax[1], ["top", "left", "right"])
            # ax.set_xlabel('$\Delta s$')
            ax[1].legend(fontsize=8, loc="lower right", bbox_to_anchor=[1.4, 0])

        ax1 = plt.axes([0.1, 0.15, 0.08, 0.35])
        ax2 = plt.axes([0.2, 0.15, 0.125, 0.35])
        plot_transitions(
            [ax1, ax2],
            "code",
            "act",
            state_ref="act",
            label_str=lambda sign, ds, ds2: "$p(\\alpha_{s%s%s}^+|\\beta_s^{\pm})$"
            % (sign, ds),
        )

        ax1 = plt.axes([0.4, 0.15, 0.08, 0.35])
        ax2 = plt.axes([0.5, 0.6, 0.125, 0.35])
        ax3 = plt.axes([0.5, 0.15, 0.125, 0.35])
        plot_transitions(
            [ax1, ax2, ax3],
            "stable",
            "act",
            state_ref="code",
            label_str=lambda sign, ds, ds2: "$p(\\alpha_{s%s%s}^+|\\gamma_{%s,s}^{\pm})$"
            % (sign, ds, ds2),
        )

        ax1 = plt.axes([0.7, 0.15, 0.08, 0.35])
        ax2 = plt.axes([0.8, 0.6, 0.125, 0.35])
        ax3 = plt.axes([0.8, 0.15, 0.125, 0.35])
        plot_transitions(
            [ax1, ax2, ax3],
            "stable",
            "code",
            state_ref="code",
            label_str=lambda sign, ds, ds2: "$p(\\beta_{s%s%s}^+|\\gamma_{%s,s}^{\pm})$"
            % (sign, ds, ds2),
        )

        return

        ### field stability -> activity
        # p(a_s+1|gamma_s,s-ds)

        ax.plot(
            [0.75, 2.25],
            [
                np.nanmean(
                    self.stats["p_post_s"]["code"]["act"][
                        self.status["sessions"], 1, 0
                    ],
                    0,
                ),
                np.nanmean(self.stats["p_post_s"]["code"]["act"][:, 1, 0], 0),
            ],
            "--",
            color="k",
            linewidth=0.5,
        )

        # # p_bs[0,:] = bootstrap_data(np.nanmean,p_pre['stable_act'][:,1,0,0,0],N_bs)
        # # p_bs[1,:] = bootstrap_data(np.nanmean,p_pre['stable_act'][:,1,1,0,0],N_bs)
        # p_bs[0,:] = bootstrap_data(np.nanmean,self.stats['p_pre_s']['stable']['act'][self.status['sessions'],1,0,0],N_bs)
        # p_bs[1,:] = bootstrap_data(np.nanmean,self.stats['p_pre_s']['stable']['act'][self.status['sessions'],1,1,0],N_bs)
        # ax.errorbar([1,2],p_bs[:,0],p_bs[:,1],fmt='-',color='tab:blue',linewidth=0.5,label='$p(\\alpha_{s-1}^+|\\gamma_{\Delta s,s}^{\pm})$')

        p_bs[0, 0], p_bs[0, 1] = bootstrap_data(
            np.nanmean,
            self.stats["p_post_s"]["stable"]["act"][self.status["sessions"], 1, 0],
            N_bs,
        )
        p_bs[1, 0], p_bs[1, 1] = bootstrap_data(
            np.nanmean,
            self.stats["p_post_s"]["stable"]["act"][self.status["sessions"], 1, 1],
            N_bs,
        )
        ax.errorbar(
            [1, 2],
            p_bs[:, 0],
            p_bs[:, 1],
            fmt="-",
            color="tab:red",
            linewidth=0.5,
            label="$p(\\alpha_{s+1}^+|\\gamma_{\Delta s,s}^{\pm})$",
        )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["$\\gamma_{1}^+$", "$\\gamma_{1}^-$"])
        ax.set_ylim([0, 1])
        ax.set_yticks(np.linspace(0, 1, 3))
        ax.set_ylabel("p")
        self.pl_dat.remove_frame(ax, ["top", "right"])

        ax = plt.axes([0.5, 0.6, 0.125, 0.35])
        ax.plot(
            [0, ds_max + 0.5],
            [
                np.nanmean(
                    self.stats["p_post_s"]["code"]["act"][
                        self.status["sessions"], 1, 0
                    ],
                    0,
                ),
                np.nanmean(
                    self.stats["p_post_s"]["code"]["act"][
                        self.status["sessions"], 1, 0
                    ],
                    0,
                ),
            ],
            "--",
            color="k",
            linewidth=0.5,
        )
        p_pre_bs = np.zeros((ds_max, 2, 2)) * np.NaN
        p_post_bs = np.zeros((ds_max, 2, 2)) * np.NaN
        for ds in range(1, ds_max):
            # # p_pre_bs[ds,0,:] = bootstrap_data(np.nanmean,p_pre['stable_act'][:,ds,0,0,0],N_bs)
            # # p_pre_bs[ds,1,:] = bootstrap_data(np.nanmean,p_pre['stable_act'][:,ds,1,0,0],N_bs)
            # p_pre_bs[ds,0,0],p_pre_bs[ds,0,1] = bootstrap_data(np.nanmean,self.stats['p_pre_s']['stable']['act'][self.status['sessions'],ds,0,1],N_bs)
            # p_pre_bs[ds,1,0],p_pre_bs[ds,1,1] = bootstrap_data(np.nanmean,self.stats['p_pre_s']['stable']['act'][self.status['sessions'],ds,1,1],N_bs)

            p_post_bs[ds, 0, 0], p_post_bs[ds, 0, 1] = bootstrap_data(
                np.nanmean,
                self.stats["p_post_s"]["stable"]["act"][self.status["sessions"], ds, 0],
                N_bs,
            )
            p_post_bs[ds, 1, 0], p_post_bs[ds, 1, 1] = bootstrap_data(
                np.nanmean,
                self.stats["p_post_s"]["stable"]["act"][self.status["sessions"], ds, 1],
                N_bs,
            )

        # ax.errorbar(range(ds_max),p_pre_bs[:,0,0],p_pre_bs[:,0,1],fmt='-',color='tab:blue',linewidth=0.5,label='$p(\\alpha_{s-1}^+|\\gamma_{\Delta s,s}^{+})$')
        ax.errorbar(
            range(ds_max),
            p_post_bs[:, 0, 0],
            p_post_bs[:, 0, 1],
            fmt="-",
            color="tab:red",
            linewidth=0.5,
            label="$p(\\alpha_{s+1}^+|\\gamma_{\Delta s,s}^{+})$",
        )
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax.set_yticks(np.linspace(0, 1, 3))
        ax.set_ylim([0, 1])
        ax.set_xticklabels([])
        # ax.set_xlabel('$\Delta s$')
        ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=[1.4, 0])

        ax = plt.axes([0.5, 0.15, 0.125, 0.35])
        ax.plot(
            [0, ds_max + 0.5],
            [
                np.nanmean(
                    self.stats["p_post_s"]["code"]["act"][
                        self.status["sessions"], 1, 0
                    ],
                    0,
                ),
                np.nanmean(
                    self.stats["p_post_s"]["code"]["act"][
                        self.status["sessions"], 1, 0
                    ],
                    0,
                ),
            ],
            "--",
            color="k",
            linewidth=0.5,
        )
        p_pre_bs = np.zeros((ds_max, 2, 2)) * np.NaN
        p_post_bs = np.zeros((ds_max, 2, 2)) * np.NaN
        for ds in range(1, ds_max):
            # # p_pre_bs[ds,0,:] = bootstrap_data(np.nanmean,p_pre['stable_act'][:,ds,0,1,0],N_bs)
            # # p_pre_bs[ds,1,:] = bootstrap_data(np.nanmean,p_pre['stable_act'][:,ds,1,1,0],N_bs)
            # p_pre_bs[ds,0,:] = bootstrap_data(np.nanmean,self.stats['p_pre_s']['stable']['act'][self.status['sessions'],ds,0,0],N_bs)
            # p_pre_bs[ds,1,:] = bootstrap_data(np.nanmean,self.stats['p_pre_s']['stable']['act'][self.status['sessions'],ds,1,0],N_bs)

            p_post_bs[ds, 0, 0], p_post_bs[ds, 0, 1] = bootstrap_data(
                np.nanmean,
                self.stats["p_post_s"]["stable"]["act"][self.status["sessions"], ds, 0],
                N_bs,
            )
            p_post_bs[ds, 1, 0], p_post_bs[ds, 1, 1] = bootstrap_data(
                np.nanmean,
                self.stats["p_post_s"]["stable"]["act"][self.status["sessions"], ds, 1],
                N_bs,
            )

        # ax.errorbar(range(ds_max),p_pre_bs[:,0,0],p_pre_bs[:,0,1],fmt='-',color='tab:blue',linewidth=0.5,label='$p(\\alpha_{s-\Delta s}^+|\\gamma_{1,s}^{+})$')
        ax.errorbar(
            range(ds_max),
            p_post_bs[:, 0, 0],
            p_post_bs[:, 0, 1],
            fmt="-",
            color="tab:red",
            linewidth=0.5,
            label="$p(\\alpha_{s+\Delta s}^+|\\gamma_{1,s}^{+})$",
        )
        self.pl_dat.remove_frame(ax, ["top", "left", "right"])
        ax.set_yticks([])
        ax.set_ylim([0, 1])
        ax.set_xlabel("$\Delta s$")
        ax.legend(fontsize=8, loc="lower right")

        ax = plt.axes([0.7, 0.15, 0.08, 0.35])
        ax.plot(
            [0.75, 2.25],
            [
                np.nanmean(
                    self.stats["p_post_s"]["code"]["code"][
                        self.status["sessions"], 1, 0
                    ],
                    0,
                ),
                np.nanmean(
                    self.stats["p_post_s"]["code"]["code"][
                        self.status["sessions"], 1, 0
                    ],
                    0,
                ),
            ],
            "--",
            color="k",
            linewidth=0.5,
        )
        # # p_bs[0,:] = bootstrap_data(np.nanmean,p_pre['stable_code'][:,1,0,0,0],N_bs)
        # # p_bs[1,:] = bootstrap_data(np.nanmean,p_pre['stable_code'][:,1,1,0,0],N_bs)
        # p_bs[0,:] = bootstrap_data(np.nanmean,self.stats['p_pre_s']['stable']['code'][self.status['sessions'],1,0,0],N_bs)
        # p_bs[1,:] = bootstrap_data(np.nanmean,self.stats['p_pre_s']['stable']['code'][self.status['sessions'],1,1,0],N_bs)
        # ax.errorbar([1,2],p_bs[:,0],p_bs[:,1],fmt='-',color='tab:blue',linewidth=0.5,label='$p(\\beta_{s-1}^+|\\gamma_{\Delta s,s}^{\pm})$')

        p_bs[0, 0], p_bs[0, 1] = bootstrap_data(
            np.nanmean,
            self.stats["p_post_s"]["stable"]["code"][self.status["sessions"], 1, 0],
            N_bs,
        )
        p_bs[1, 0], p_bs[1, 1] = bootstrap_data(
            np.nanmean,
            self.stats["p_post_s"]["stable"]["code"][self.status["sessions"], 1, 1],
            N_bs,
        )
        ax.errorbar(
            [1, 2],
            p_bs[:, 0],
            p_bs[:, 1],
            fmt="-",
            color="tab:red",
            linewidth=0.5,
            label="$p(\\beta_{s+1}^+|\\gamma_{\Delta s,s}^{\pm})$",
        )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["$\\gamma_{1}^+$", "$\\gamma_{1}^-$"])
        ax.set_ylim([0, 1])
        ax.set_yticks(np.linspace(0, 1, 3))
        self.pl_dat.remove_frame(ax, ["top", "right"])

        ax = plt.axes([0.8, 0.6, 0.125, 0.35])
        ax.plot(
            [0, ds_max + 0.5],
            [
                np.nanmean(
                    self.stats["p_post_s"]["code"]["code"][
                        self.status["sessions"], 1, 0
                    ],
                    0,
                ),
                np.nanmean(
                    self.stats["p_post_s"]["code"]["code"][
                        self.status["sessions"], 1, 0
                    ],
                    0,
                ),
            ],
            "--",
            color="k",
            linewidth=0.5,
        )
        for ds in range(1, ds_max):
            # # p_pre_bs[ds,0,:] = bootstrap_data(np.nanmean,p_pre['stable_code'][:,ds,0,0,0],N_bs)
            # # p_pre_bs[ds,1,:] = bootstrap_data(np.nanmean,p_pre['stable_code'][:,ds,1,0,0],N_bs)
            # p_pre_bs[ds,0,:] = bootstrap_data(np.nanmean,self.stats['p_pre_s']['stable']['code'][self.status['sessions'],ds,0,1],N_bs)
            # p_pre_bs[ds,1,:] = bootstrap_data(np.nanmean,self.stats['p_pre_s']['stable']['code'][self.status['sessions'],ds,1,1],N_bs)

            p_post_bs[ds, 0, 0], p_post_bs[ds, 0, 1] = bootstrap_data(
                np.nanmean,
                self.stats["p_post_s"]["stable"]["code"][
                    self.status["sessions"], ds, 0
                ],
                N_bs,
            )
            p_post_bs[ds, 1, 0], p_post_bs[ds, 1, 1] = bootstrap_data(
                np.nanmean,
                self.stats["p_post_s"]["stable"]["code"][
                    self.status["sessions"], ds, 1
                ],
                N_bs,
            )

        # ax.errorbar(range(ds_max),p_pre_bs[:,0,0],p_pre_bs[:,0,1],fmt='-',color='tab:blue',linewidth=0.5,label='$p(\\beta_{s-1}^+|\\gamma_{\Delta s,s}^{+})$')
        ax.errorbar(
            range(ds_max),
            p_post_bs[:, 0, 0],
            p_post_bs[:, 0, 1],
            fmt="-",
            color="tab:red",
            linewidth=0.5,
            label="$p(\\beta_{s+1}^+|\\gamma_{\Delta s,s}^{+})$",
        )
        # ax.set_xlabel('$\Delta s$')
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=[1.4, 1.1])

        ax.set_xticklabels([])
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax.set_yticks(np.linspace(0, 1, 3))

        ax = plt.axes([0.8, 0.15, 0.125, 0.35])
        ax.plot(
            [0, ds_max + 0.5],
            [
                np.nanmean(
                    self.stats["p_post_s"]["code"]["code"][
                        self.status["sessions"], 1, 0
                    ],
                    0,
                ),
                np.nanmean(
                    self.stats["p_post_s"]["code"]["code"][
                        self.status["sessions"], 1, 0
                    ],
                    0,
                ),
            ],
            "--",
            color="k",
            linewidth=0.5,
        )
        for ds in range(1, ds_max):
            # # p_pre_bs[ds,0,:] = bootstrap_data(np.nanmean,p_pre['stable_code'][:,ds,0,1,0],N_bs)
            # # p_pre_bs[ds,1,:] = bootstrap_data(np.nanmean,p_pre['stable_code'][:,ds,1,1,0],N_bs)
            # p_pre_bs[ds,0,:] = bootstrap_data(np.nanmean,self.stats['p_pre_s']['stable']['code'][self.status['sessions'],ds,0,0],N_bs)
            # p_pre_bs[ds,1,:] = bootstrap_data(np.nanmean,self.stats['p_pre_s']['stable']['code'][self.status['sessions'],ds,1,0],N_bs)

            p_post_bs[ds, 0, 0], p_post_bs[ds, 0, 1] = bootstrap_data(
                np.nanmean,
                self.stats["p_post_s"]["stable"]["code"][
                    self.status["sessions"], ds, 0
                ],
                N_bs,
            )
            p_post_bs[ds, 1, 0], p_post_bs[ds, 1, 1] = bootstrap_data(
                np.nanmean,
                self.stats["p_post_s"]["stable"]["code"][
                    self.status["sessions"], ds, 1
                ],
                N_bs,
            )

        # ax.errorbar(range(ds_max),p_pre_bs[:,0,0],p_pre_bs[:,0,1],fmt='-',color='tab:blue',linewidth=0.5,label='$p(\\beta_{s-\Delta s}^+|\\gamma_{1,s}^{+})$')
        ax.errorbar(
            range(ds_max),
            p_post_bs[:, 0, 0],
            p_post_bs[:, 0, 1],
            fmt="-",
            color="tab:red",
            linewidth=0.5,
            label="$p(\\beta_{s+\Delta s}^+|\\gamma_{1,s}^{+})$",
        )
        ax.set_xlabel("$\Delta s$")
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=[1.4, 1.1])

        self.pl_dat.remove_frame(ax, ["top", "left", "right"])
        ax.set_yticks([])

        plt.show(block=False)

    def plot_timedependent_transition_probs(self):

        print("### plot time-dependent probabilities ###")

        nSes = self.data["nSes"]
        nbin = self.data["nbin"]

        SD = 1.96
        if nSes > 50:
            s_arr = np.array([0, 5, 17, 30, 87])
        # s_arr = np.array([0,16,60,87,96,107])
        else:
            s_arr = np.array([0, 5, 10, 15, 20])
        # s_arr += np.where(self.status['sessions'])[0][0]

        n_int = len(s_arr) - 1

        # status_arr = ['act','code','stable']
        # p_post = np.zeros((4,n_int,3,3,2))
        # suffix_arr = ['','_RW','_GT','_nRnG']
        # for k in range(n_int):

        #     for j,key in enumerate(status_arr):
        #         for i,key2 in enumerate(status_arr):
        #             for l,sf in enumerate(suffix_arr):
        #                 p_post[l,k,j,i,0] = np.nanmean(self.stats['p_post%s_s'%sf][key][key2][s_arr[k]:s_arr[k+1],1,0])
        #                 p_post[l,k,j,i,1] = np.nanstd(self.stats['p_post%s_s'%sf][key][key2][s_arr[k]:s_arr[k+1],1,0])

        plt.figure(figsize=(7, 5), dpi=300)

        label_arr = ["RW", "GT", "nRnG"]
        ax = plt.axes([0.1, 0.8, 0.225, 0.175])
        ax.plot(gauss_smooth(self.stats["p_post_s"]["act"]["stable"][:, 1, 0], 1), "k")
        ax.plot(gauss_smooth(self.stats["p_post_s"]["code"]["stable"][:, 1, 0], 1), "r")
        ax.plot(
            gauss_smooth(self.stats["p_post_s"]["stable"]["stable"][:, 1, 0], 1), "b"
        )
        ax.set_ylim([0.0, 1])
        ax.set_xlim([0, np.where(self.status["sessions"])[0][-1]])
        ax.set_xlabel("session")
        ax.set_ylabel("$p(\\gamma_{s+1}^+|\cdot^+)$")

        self.pl_dat.remove_frame(ax, ["top", "right"])

        status_key = ["alpha", "beta", "gamma"]

        ## plotting location-dependent transition rates
        # # col_arr = []
        # idx = 1
        # for l,sf in enumerate(suffix_arr[1:]):
        #     ax2 = plt.axes([0.1+0.125*l,0.55,0.075,0.1])
        #     for j in range(len(s_arr)-1):
        #         ax.plot([s_arr[j],s_arr[j]],[0,2],'--',color=[0.5,0.5,0.5],linewidth=0.5,zorder=0)
        #         ax2.errorbar(np.arange(n_int),p_post[0,:,idx,2,0],p_post[l+1,:,idx,2,1],fmt='o',mec=[0.6,0.6,0.6],linewidth=0.5,markersize=1)
        #         ax2.errorbar(np.arange(n_int),p_post[l+1,:,idx,2,0],p_post[l+1,:,idx,2,1],fmt='ko',linewidth=0.5,markersize=2)
        #     ax2.set_xticks(np.arange(n_int))
        #     ax2.set_xticklabels(['s%d-%d'%(s_arr[j]+1,s_arr[j+1]) for j in range(n_int)],rotation=60,fontsize=8)
        #     ax2.set_ylim([0,1.05])
        #     pl_dat.remove_frame(ax2,['top','right'])
        #     ax2.set_title(label_arr[l])
        #     if l==0:
        #         ax2.set_ylabel('$p(\\gamma_{s+1}^+|\\%s_s^+)$'%status_key[idx])
        #     else:
        #         ax2.set_yticklabels([])
        # pl_dat.remove_frame(ax2,['top','right'])

        ax = plt.axes([0.6, 0.8, 0.225, 0.175])
        ax.plot(gauss_smooth(self.stats["p_post_s"]["act"]["act"][:, 1, 0], 1), "k")
        ax.plot(gauss_smooth(self.stats["p_post_s"]["code"]["act"][:, 1, 0], 1), "r")
        ax.plot(gauss_smooth(self.stats["p_post_s"]["stable"]["act"][:, 1, 0], 1), "b")
        ax.set_ylim([0.5, 1])
        ax.set_xlim([0, np.where(self.status["sessions"])[0][-1]])
        ax.set_xlabel("session")
        ax.set_ylabel("$p(\\alpha_{s+1}^+|\cdot^+)$")
        self.pl_dat.remove_frame(ax, ["top", "right"])

        ## plotting location dependent transition-rates
        # idx = 1
        # for l,sf in enumerate(suffix_arr[1:]):
        #     ax2 = plt.axes([0.6+0.125*l,0.55,0.075,0.1])
        #     for j in range(len(s_arr)-1):
        #         # if np.any(self.session_data['RW_pos'])
        #         ax.plot([s_arr[j],s_arr[j]],[0,2],'--',color=[0.5,0.5,0.5],linewidth=0.5,zorder=0)
        #         ax2.errorbar(np.arange(n_int),p_post[0,:,idx,0,0],p_post[0,:,idx,0,1],fmt='o',mec=[0.6,0.6,0.6],linewidth=0.5,markersize=1)
        #         ax2.errorbar(np.arange(n_int),p_post[l+1,:,idx,0,0],p_post[l+1,:,idx,0,1],fmt='ko',linewidth=0.5,markersize=2)
        #     ax2.set_ylim([0,1.05])
        #     ax2.set_xticks(np.arange(n_int))
        #     ax2.set_xticklabels(['s%d-%d'%(s_arr[j]+1,s_arr[j+1]) for j in range(n_int)],rotation=60,fontsize=8)
        #     pl_dat.remove_frame(ax2,['top','right'])
        #     ax2.set_title(label_arr[l])
        #     if l==0:
        #         ax2.set_ylabel('$p(\\alpha_{s+1}^+|\\%s_s^+)$'%status_key[idx])
        #     else:
        #         ax2.set_yticklabels([])
        # pl_dat.remove_frame(ax2,['top','right'])

        ds_max = 11
        p_rec_loc = np.zeros((n_int, nbin, ds_max)) * np.NaN
        N_rec_loc = np.zeros((n_int, nbin, ds_max))

        s1_shifts, s2_shifts, f1, f2 = np.unravel_index(
            self.compare["pointer"].col,
            (
                nSes,
                nSes,
                self.params["field_count_max"],
                self.params["field_count_max"],
            ),
        )
        c_shifts = self.compare["pointer"].row

        sig_theta = self.stability["all"]["mean"][0, 2]
        di = 3

        for ds in range(ds_max):
            Ds = s2_shifts - s1_shifts
            idx = np.where(Ds == ds)[0]
            idx_shifts = self.compare["pointer"].data[idx].astype("int") - 1
            shifts = self.compare["shifts"][idx_shifts]

            s = s1_shifts[idx]
            f = f1[idx]
            c = c_shifts[idx]
            loc_shifts = np.round(self.fields["location"][c, s, f, 0]).astype("int")

            for j in range(len(s_arr) - 1):
                for i in range(nbin):
                    i_min = max(0, i - di)
                    i_max = min(nbin, i + di)
                    idx_loc = (
                        (loc_shifts >= i_min)
                        & (loc_shifts < i_max)
                        & ((s >= s_arr[j]) & (s < s_arr[j + 1]))
                    )

                    shifts_loc = shifts[idx_loc]
                    N_data = len(shifts_loc)
                    N_stable = (np.abs(shifts_loc) < (SD * sig_theta)).sum()

                    p_rec_loc[j, i, ds] = N_stable / N_data
                    N_rec_loc[j, i, ds] = N_stable

        p_act_loc = np.zeros((nSes, nbin, ds_max)) * np.NaN
        N_act_loc = np.zeros((nSes, nbin, ds_max))
        for s in np.where(self.status["sessions"])[0]:
            for ds in range(min(nSes - s, ds_max)):
                if self.status["sessions"][s + ds]:
                    loc = self.fields["location"][:, s, :]

                    for i in range(nbin):
                        i_min = max(0, i - di)
                        i_max = min(nbin, i + di)
                        idx_loc = np.where((i_min <= loc) & (loc < i_max))
                        p_act_loc[s, i, ds] = self.status["activity"][
                            idx_loc[0], s + ds, 1
                        ].mean()
                        N_act_loc[s, i, ds] = self.status["activity"][
                            idx_loc[0], s + ds, 1
                        ].sum()

        props = dict(boxstyle="round", facecolor="w", alpha=0.8)

        for j in range(n_int):
            ax_im = plt.axes([0.1 + 0.2 * j, 0.25, 0.15, 0.1])
            im = ax_im.imshow(
                gauss_smooth(p_rec_loc[j, ...], (1, 0)),
                clim=[0.25, 0.75],
                interpolation="None",
                origin="lower",
                aspect="auto",
            )
            ax_im.set_xlim([0.5, 10.5])
            ax_im.set_xticklabels([1, 5, 10])
            ax_im.text(
                x=0.5,
                y=110,
                s="Sessions %d-%d" % (s_arr[j] + 1, s_arr[j + 1]),
                ha="left",
                va="bottom",
                bbox=props,
                fontsize=8,
            )
            ax_im.set_ylim([0, 100])
            ax_im.set_xticks([])
            if j == 0:
                ax_im.set_ylabel("pos.")
            else:
                ax_im.set_yticklabels([])
            if j == (n_int - 1):
                cb = plt.colorbar(im)
                cb.set_label("$p(\\gamma_{\Delta s}^+|\\beta^+)$", fontsize=8)
            # else:
            # ax_im.set_xlabel('$\Delta s$ [sessions]')

            p_act_range = np.nanmean(p_act_loc[s_arr[j] : s_arr[j + 1], ...], 0)
            ax_im = plt.axes([0.1 + 0.2 * j, 0.1, 0.15, 0.1])
            im = ax_im.imshow(
                gauss_smooth(p_act_range, (1, 0)),
                clim=[0.25, 0.75],
                interpolation="None",
                origin="lower",
                aspect="auto",
            )
            ax_im.set_xlim([0.5, 10])
            ax_im.set_xticks([1, 5, 10])
            ax_im.set_ylim([0, 100])
            # ax_im.text(x=6,y=107,s='Sessions %d-%d'%(s_arr[j]+1,s_arr[j+1]),ha='left',va='bottom',bbox=props,fontsize=8)
            if j == 0:
                ax_im.set_ylabel("pos.")
            else:
                ax_im.set_yticklabels([])
            if j == (n_int - 1):
                cb = plt.colorbar(im)
                cb.set_label("$p(\\alpha_{\Delta s}^+|\\beta^+)$", fontsize=8)
            ax_im.set_xlabel("$\Delta s$")

        ax_rec = plt.axes([0.375, 0.8, 0.1, 0.175])
        ax_rec.plot(
            [0, n_int],
            [0.2, 0.2],
            color=[0.5, 0.5, 0.5],
            linestyle="--",
            linewidth=0.5,
            zorder=0,
        )
        ax_rec.set_ylim([0, 1])

        # ax_act = plt.axes([0.875,0.8,0.1,0.175])
        # ax_act.plot([0,n_int],[0.2,0.2],color=[0.5,0.5,0.5],linestyle='--',linewidth=0.5,zorder=0)
        # ax_act.set_ylim([0,1])

        # for j in range(n_int):
        #     RW_pos = self.session_data['RW_pos'][s_arr[j],:].astype('int')
        #     GT_pos = self.session_data['GT_pos'][s_arr[j],:].astype('int')
        #     N_all = N_rec_loc[j,:,1].sum()
        #     N_RW = N_rec_loc[j,RW_pos[0]:RW_pos[1],1].sum()
        #     N_GT = N_rec_loc[j,GT_pos[0]:GT_pos[1],1].sum()

        #     ax_rec.plot(j,N_RW/N_all,'o',color='tab:red',markersize=2)
        #     ax_rec.plot(j,N_GT/N_all,'o',color='tab:green',markersize=2)
        #     ax_rec.plot(j,(N_all-N_RW-N_GT)/N_all,'o',color='tab:blue',markersize=2)

        #     N_all = N_act_loc[s_arr[j]:s_arr[j+1],:,1].sum(axis=(0,1))
        #     N_RW = N_act_loc[s_arr[j]:s_arr[j+1],RW_pos[0]:RW_pos[1],1].sum(axis=(0,1))
        #     N_GT = N_act_loc[s_arr[j]:s_arr[j+1],GT_pos[0]:GT_pos[1],1].sum(axis=(0,1))

        #     ax_act.plot(j,N_RW/N_all,'o',color='tab:red',markersize=2)
        #     ax_act.plot(j,N_GT/N_all,'o',color='tab:green',markersize=2)
        #     ax_act.plot(j,(N_all-N_RW-N_GT)/N_all,'o',color='tab:blue',markersize=2)

        # pl_dat.remove_frame(ax_rec,['top','right'])
        # pl_dat.remove_frame(ax_act,['top','right'])

        # ax_rec.set_xticks(np.arange(n_int))
        # ax_rec.set_xticklabels(['s%d-%d'%(s_arr[j]+1,s_arr[j+1]) for j in range(n_int)],rotation=60,fontsize=8)
        # ax_act.set_xticks(np.arange(n_int))
        # ax_act.set_xticklabels(['s%d-%d'%(s_arr[j]+1,s_arr[j+1]) for j in range(n_int)],rotation=60,fontsize=8)

        # ax.set_xlabel('session')
        plt.show(block=False)

        # if sv:
        #     self.pl_dat.save_fig('timedep_dynamics')

    def plot_performancedependent_statistics(self):

        print("### plot performance-dependent statistics ###")
        print("### this needs some work!!! ###")

        nSes = self.data["nSes"]

        RW_reception = np.zeros(nSes) * np.NaN
        slowing = np.zeros(nSes) * np.NaN
        for s in range(nSes):
            if s in self.behavior["performance"].keys():
                RW_reception[s] = self.behavior["performance"][s]["RW_reception"].mean()
                slowing[s] = self.behavior["performance"][s]["slowDown"].mean()

        perf = slowing  # self.sessions['time_active']/600
        plt.figure(figsize=(7, 5), dpi=300)
        ax = plt.axes([0.1, 0.7, 0.15, 0.25])
        ax.plot(RW_reception)
        ax.set_ylim([0, 1])
        ax = plt.axes([0.3, 0.7, 0.15, 0.25])
        ax.plot(slowing)
        ax.set_ylim([0, 1])
        ax = plt.axes([0.5, 0.7, 0.15, 0.25])
        ax.plot(RW_reception, slowing, "k.", markersize=2)
        ax.set_ylim([0, 1])
        ax = plt.axes([0.7, 0.7, 0.15, 0.25])
        ax.plot(
            RW_reception, self.status["activity"][..., 2].sum(0), "k.", markersize=2
        )
        ax.plot(slowing, self.status["activity"][..., 2].sum(0), "r.", markersize=2)
        ax.plot(perf, self.status["activity"][..., 2].sum(0), "b.", markersize=2)
        # ax.set_ylim([0,1])

        ds = 1
        ax = plt.axes([0.1, 0.1, 0.25, 0.35])
        ax.plot(
            perf[ds:],
            self.stats["p_post_s"]["stable"]["stable"][: nSes - ds, 1, 0],
            "k.",
            markersize=2,
        )
        ax.plot(
            perf[ds:],
            self.stats["p_post_s"]["stable"]["code"][: nSes - ds, 1, 0],
            "r.",
            markersize=2,
        )
        ax.plot(
            perf[ds:],
            self.stats["p_post_s"]["stable"]["act"][: nSes - ds, 1, 0],
            "b.",
            markersize=2,
        )
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        ax = plt.axes([0.4, 0.1, 0.25, 0.35])
        ax.plot(
            perf[ds:],
            self.stats["p_post_s"]["code"]["stable"][: nSes - ds, 1, 0],
            "k.",
            markersize=2,
        )
        ax.plot(
            perf[ds:],
            self.stats["p_post_s"]["code"]["code"][: nSes - ds, 1, 0],
            "r.",
            markersize=2,
        )
        ax.plot(
            perf[ds:],
            self.stats["p_post_s"]["code"]["act"][: nSes - ds, 1, 0],
            "b.",
            markersize=2,
        )
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        ax = plt.axes([0.7, 0.1, 0.25, 0.35])
        ax.plot(
            perf[ds:],
            self.stats["p_post_s"]["act"]["stable"][: nSes - ds, 1, 0],
            "k.",
            markersize=2,
        )
        ax.plot(
            perf[ds:],
            self.stats["p_post_s"]["act"]["code"][: nSes - ds, 1, 0],
            "r.",
            markersize=2,
        )
        ax.plot(
            perf[ds:],
            self.stats["p_post_s"]["act"]["act"][: nSes - ds, 1, 0],
            "b.",
            markersize=2,
        )
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.show(block=False)

    def plot_location_dependent_statistics(self, sv=False):

        print("### plot location-specific, static statistics ###")

        nSes = self.data["nSes"]
        nbin = self.data["nbin"]

        ## initialize some arrays
        loc = np.round(self.fields["location"][..., 0]).astype("int")

        # par_keys = ['width','reliability','firingrate','max_rate','MI_value']
        par_keys = [
            "width",
            "oof_firingrate_adapt",
            "if_firingrate_adapt",
            "reliability",
            "MI_value",
        ]
        par_labels = ["$\sigma$", "$\\nu^-$", "$\\nu^*$", "a", "MI"]
        ranges = np.array([[0, 20], [0, 2.0], [0, 6], [0, 1], [0, 1.0]])

        distr = {}
        for key in par_keys:
            distr[key] = np.zeros((nbin, 2)) * np.NaN

        fig = plt.figure(figsize=(7, 4), dpi=self.pl_dat.sv_opt["dpi"])

        ### place field density
        ax_im = plt.axes([0.1, 0.35, 0.325, 0.2])
        ax = plt.axes([0.1, 0.12, 0.325, 0.11])
        self.pl_dat.add_number(fig, ax_im, order=2)
        s_range = 20

        # ax = plt.axes([0.525,0.1,0.375,0.275])
        fields = np.zeros((nbin, nSes))
        for i, s in enumerate(np.where(self.status["sessions"])[0]):
            # idx_PC = np.where(self.fields['status'][:,s,:]>=3)
            idx_PC = np.where(self.status["fields"][:, s, :])
            # idx_PC = np.where(~np.isnan(self.fields['location'][:,s,:]))
            # fields[s,:] = np.nansum(self.fields['p_x'][:,s,:,:],1).sum(0)
            fields[:, s] = np.nansum(self.fields["p_x"][idx_PC[0], s, idx_PC[1], :], 0)
            # fields[:,s] /= fields[:,s].sum()
            ax.plot(
                gauss_smooth(fields[:, s] / fields[:, s].sum(), 1),
                "-",
                color=[0.5, 0.5, 0.5],
                linewidth=0.3,
                alpha=0.5,
            )
        fields = gauss_smooth(fields, (2, 0))

        im = ax_im.imshow(
            fields / fields.max(0), origin="lower", aspect="auto", cmap="hot"
        )  # ,clim=[0,1])
        ax_im.set_xlim([-0.5, nSes - 0.5])
        ax_im.set_xlim(
            [
                -0.5 + np.where(self.status["sessions"])[0][0],
                np.where(self.status["sessions"])[0][-1] - 0.5,
            ]
        )
        ax_im.set_ylim([0, 100])

        cbaxes = plt.axes([0.44, 0.35, 0.01, 0.2])
        h_cb = plt.colorbar(im, cax=cbaxes)
        h_cb.set_label("place field \ndensity", fontsize=8)
        h_cb.set_ticks([])

        ax_im.set_xlabel("session")
        ax_im.set_ylabel("position")

        self.pl_dat.add_number(fig, ax, order=3)
        s_arr = [24, 44, 74]
        # s_arr = [0,5,17,88]
        # s_arr += np.where(self.status['sessions'])[0][0]
        for i in range(len(s_arr)):
            col = [0.7 - 0.35 * i, 0.7 - 0.35 * i, 0.7 - 0.35 * i]
            ax_im.annotate(
                text="",
                xy=(s_arr[i], 100),
                xytext=(s_arr[i], 110),
                fontsize=6,
                annotation_clip=False,
                arrowprops=dict(arrowstyle="->", color=col),
            )
            # ax_im.annotate(s='',xy=(s_arr[i+1]-1,100),xytext=(s_arr[i+1]-1,110),fontsize=6,annotation_clip=False,arrowprops=dict(arrowstyle='->',color=col))

        ax.plot(np.nanmean(fields[:, 1:15] / fields[:, 1:15].max(0), 1), color="k")

        ax.set_xlim([0, 100])
        ax.set_yticks([])
        ax.set_xlabel("position")
        ax.set_ylabel("density")
        self.pl_dat.remove_frame(ax, ["top", "right"])

        # s_arr2 = np.array([1,14,34])
        # s_arr2 += np.where(self.status['sessions'])[0][0]

        props = dict(boxstyle="round", facecolor="w", alpha=0.8)
        for j, s in enumerate(s_arr):
            if s < nSes:
                ax = plt.axes([0.075 + 0.12 * j, 0.65, 0.1, 0.275])
                if j == 0:
                    self.pl_dat.add_number(fig, ax, order=1, offset=[-100, 50])
                idxes_tmp = np.where(
                    self.status["fields"][:, s, :]
                    & (self.stats["SNR_comp"][:, s] > 2)[..., np.newaxis]
                    & (self.stats["r_values"][:, s] > 0)[..., np.newaxis]
                    & (self.matching["score"][:, s, 0] > 0.5)[..., np.newaxis]
                )
                idxes = idxes_tmp[0]
                sort_idx = np.argsort(
                    self.fields["location"][idxes_tmp[0], s, idxes_tmp[1], 0]
                )
                sort_idx = idxes[sort_idx]
                nID = len(sort_idx)

                firingmap = self.stats["firingmap"][sort_idx, s, :]
                firingmap = gauss_smooth(firingmap, [0, 2])
                firingmap = firingmap - np.nanmin(firingmap, 1)[:, np.newaxis]
                # firingmap = firingmap / np.nanmax(firingmap,1)[:,np.newaxis]
                im = ax.imshow(
                    firingmap, aspect="auto", origin="upper", cmap="jet", clim=[0, 5]
                )
                ax.text(
                    5, nID * 0.95, "n = %d" % nID, bbox=props, color="k", fontsize=6
                )
                ax.text(
                    95,
                    nID / 10,
                    "Session %d" % (s + 1),
                    bbox=props,
                    color="k",
                    fontsize=6,
                    ha="right",
                )
                self.pl_dat.remove_frame(ax)
                ax.set_xticks([])
                ax.set_yticks([])

        cbaxes = plt.axes([0.425, 0.825, 0.01, 0.1])
        h_cb = plt.colorbar(im, cax=cbaxes)
        h_cb.set_label("$Ca^{2+}$", fontsize=8)
        h_cb.set_ticks([0, 5])
        h_cb.set_ticklabels(["low", "high"])

        ### location-specific parameters
        ## width, rel, MI, max_rate
        for j, key in enumerate(par_keys):
            ax = plt.axes([0.6, 0.8 - j * 0.17, 0.375, 0.13])
            if j == 0:
                self.pl_dat.add_number(fig, ax, order=4)
            if key in ["oof_firingrate_adapt", "if_firingrate_adapt", "MI_value"]:
                dat = self.stats[key]
            elif key == "width":
                dat = self.fields[key][..., 0]
            else:
                dat = self.fields[key]

            for i in range(nbin):
                idx = (
                    (loc == i)
                    & self.status["fields"]
                    & ((np.arange(nSes) < 15) & (np.arange(nSes) > 5))[
                        np.newaxis, :, np.newaxis
                    ]
                )
                if key in ["oof_firingrate_adapt", "if_firingrate_adapt", "MI_value"]:
                    idx = np.any(idx, -1)
                distr[key][i, 0] = np.nanmean(dat[idx])
                distr[key][i, 1] = np.nanstd(dat[idx])
            idx = np.where(self.status["fields"])
            if key in ["oof_firingrate_adapt", "if_firingrate_adapt", "MI_value"]:
                ax.plot(
                    self.fields["location"][idx[0], idx[1], idx[2], 0],
                    dat[idx[0], idx[1]],
                    ".",
                    color=[0.6, 0.6, 0.6],
                    markersize=1,
                    markeredgewidth=0,
                    zorder=0,
                )
            else:
                ax.plot(
                    self.fields["location"][idx[0], idx[1], idx[2], 0],
                    dat[idx[0], idx[1], idx[2]],
                    ".",
                    color=[0.6, 0.6, 0.6],
                    markersize=1,
                    markeredgewidth=0,
                    zorder=0,
                )
            self.pl_dat.plot_with_confidence(
                ax,
                np.linspace(0, nbin - 1, nbin),
                distr[key][:, 0],
                distr[key][:, 1],
                col="tab:red",
            )
            ax.set_ylabel(par_labels[j], rotation="vertical", ha="left", va="center")
            ax.yaxis.set_label_coords(-0.175, 0.5)
            self.pl_dat.remove_frame(ax, ["top", "right"])
            if j < 4:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("position [bins]")
            ax.set_ylim(ranges[j, :])
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        # plt.tight_layout()

        # ax = plt.axes([0.6,0.1,0.35,0.2])
        # s1_shifts,s2_shifts,f1,f2 = np.unravel_index(self.compare['pointer'].col,(nSes,nSes,self.params['field_count_max'],self.params['field_count_max']))
        # c_shifts = self.compare['pointer'].row
        #
        # Ds = s2_shifts-s1_shifts
        # idx = np.where(Ds==1)[0]
        # idx_shifts = self.compare['pointer'].data[idx].astype('int')-1
        # shifts = np.abs(self.compare['shifts'][idx_shifts])
        #
        # loc_ref = self.fields['location'][c_shifts[idx],s1_shifts[idx],f1[idx],0].astype('int')
        # shift_dist = np.zeros((nbin,2))
        # for i in range(nbin):
        #     shift_dist[i,0] = shifts[loc_ref==i].mean()
        #     shift_dist[i,1] = shifts[loc_ref==i].std()
        #
        # pl_dat.plot_with_confidence(ax,range(nbin),shift_dist[:,0],shift_dist[:,1],col='k')
        # ax.set_xlabel('position')

        # ax = plt.axes([0.6,0.4,0.35,0.2])

        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig("PC_locationStats")

        fields /= fields.sum(0)[np.newaxis, :]

        plt.figure(figsize=(3, 2), dpi=self.pl_dat.sv_opt["dpi"])

        density = {}
        density["reward"] = np.zeros(nSes)
        density["gate"] = np.zeros(nSes)
        density["others"] = np.zeros(nSes)
        for s in np.where(self.status["sessions"])[0]:
            zone_mask = {}
            zone_mask["reward"] = np.zeros(nbin).astype("bool")
            zone_mask["gate"] = np.zeros(nbin).astype("bool")
            zone_mask["others"] = np.ones(nbin).astype("bool")

            RW_pos = self.session_data["RW_pos"][s, :].astype("int")
            GT_pos = self.session_data["GT_pos"][s, :].astype("int")
            print("session %d" % s)
            print(RW_pos)
            zone_mask["reward"][RW_pos[0] : RW_pos[1]] = True
            zone_mask["others"][zone_mask["reward"]] = False
            if ~np.isnan(self.session_data["GT_pos"][s, 0]):
                zone_mask["gate"][GT_pos[0] : GT_pos[1]] = True
                zone_mask["others"][zone_mask["gate"]] = False
            zone_mask["others"][:10] = False
            zone_mask["others"][-10:] = False
            for key in ["reward", "gate", "others"]:
                density[key][s] = np.nanmean(fields[zone_mask[key], s], 0)

        # ax = plt.axes([0.1,0.6,0.25,0.35])
        ax = plt.subplot(111)
        self.pl_dat.add_number(fig, ax, order=1)
        # print(fields.sum(0))
        # print(fields[self.params['zone_mask']['reward'],:],0)
        ax.plot(gauss_smooth(density["reward"], 1), color="tab:red")
        ax.plot(gauss_smooth(density["gate"], 1), color="tab:green")
        ax.plot(gauss_smooth(density["others"], 1), color="tab:blue")
        ax.set_ylim([0, 0.02])
        ax.set_xlim([0, np.where(self.status["sessions"])[0][-1]])
        self.pl_dat.remove_frame(ax, ["top", "right"])

        plt.tight_layout()
        plt.show(block=False)

        # ax_sig = plt.axes([0.1,0.15,0.15,0.2])
        # pl_dat.add_number(fig,ax_sig,order=2)
        # ax_MI = plt.axes([0.325,0.15,0.15,0.2])
        # ax_rate1 = plt.axes([0.5,0.15,0.15,0.2])
        # ax_rate2 = plt.axes([0.675,0.15,0.15,0.2])
        # ax_rel = plt.axes([0.85,0.15,0.15,0.2])
        # # s_arr = np.arange(0,nSes+s_range,s_range)
        # # s_arr = np.array([0,5,17,50,87])
        #
        # for j in range(len(s_arr)-1):
        #     idx = (self.status["fields"] & (self.stats['SNR']>2)[...,np.newaxis] & (self.stats['r_values']>0)[...,np.newaxis] & (self.matching['score'][...,0]>0.9)[...,np.newaxis] & ((np.arange(nSes)>=s_arr[j]) & (np.arange(nSes)<s_arr[j+1]))[np.newaxis,:,np.newaxis])
        #     density = np.histogram(self.fields['location'][idx,0],np.linspace(0,nbin,nbin+1),density=True)
        #     print(idx.shape)
        #     col = [0.1+0.225*j,0.1+0.225*j,1]
        #     # ax.plot(np.linspace(0,nbin-1,nbin),density[0],color=col,label='s %d-%d'%(s_arr[j]+1,s_arr[j+1]))
        #
        #     _,_,patches = ax_sig.hist(self.fields['width'][idx,0],np.linspace(0,20,51),color=col,cumulative=True,density=True,histtype='step')
        #     patches[0].set_xy(patches[0].get_xy()[:-1])
        #     _,_,patches =ax_rel.hist(self.fields['reliability'][idx],np.linspace(0,1,51),color=col,cumulative=True,density=True,histtype='step')
        #     patches[0].set_xy(patches[0].get_xy()[:-1])
        #     _,_,patches = ax_rate1.hist(self.stats['firingrate_adapt'][np.any(idx,-1)],np.linspace(0,0.5,51),color=col,cumulative=True,density=True,histtype='step')
        #     patches[0].set_xy(patches[0].get_xy()[:-1])
        #     _,_,patches =ax_rate2.hist(self.fields['max_rate'][idx],np.linspace(0,50,51),color=col,cumulative=True,density=True,histtype='step')
        #     patches[0].set_xy(patches[0].get_xy()[:-1])
        #     _,_,patches = ax_MI.hist(self.stats['MI_value'][np.any(idx,-1)],np.linspace(0,1,51),color=col,cumulative=True,density=True,histtype='step')
        #     patches[0].set_xy(patches[0].get_xy()[:-1])
        #     # ax.hist(self.fields['location'][idx,0],np.linspace(0,nbin-1,nbin),density=True,histtype='step')
        # pl_dat.remove_frame(ax_sig,['top','right'])
        # pl_dat.remove_frame(ax_MI,['top','right'])
        # pl_dat.remove_frame(ax_rate1,['top','right'])
        # pl_dat.remove_frame(ax_rate2,['top','right'])
        # pl_dat.remove_frame(ax_rel,['top','right'])
        # ax_sig.set_ylabel('fraction')
        # ax_sig.set_xlabel('$\sigma$ [bins]')
        # ax_MI.set_yticklabels([])
        # ax_MI.set_xlabel('MI [bit]')
        # ax_rate1.set_yticklabels([])
        # ax_rate1.set_xlabel('$\\nu^-$ [Hz]')
        # ax_rate2.set_yticklabels([])
        # ax_rate2.set_xlabel('$\\nu^*$ [Hz]')
        # ax_rel.set_yticklabels([])
        # ax_rel.set_xlabel('a')
        # # ax.set_xlabel('position [bins]')
        # # ax.set_ylabel('density')
        # # ax.legend(fontsize=8,loc='upper right',bbox_to_anchor=[1.25,1.2],handlelength=1)
        # # pl_dat.remove_frame(ax,['top','right'])
        # plt.show(block=False)

        if sv:
            self.pl_dat.save_fig("PC_timeStats")

    def plot_time_dependent_statistics(self, sv=False):

        print("### plot time-specific, static statistics ###")

        nSes = self.data["nSes"]
        nbin = self.data["nbin"]

        fig = plt.figure(figsize=(7, 5), dpi=self.pl_dat.sv_opt["dpi"])

        ax_im = plt.axes([0.1, 0.35, 0.325, 0.2])
        ax = plt.axes([0.1, 0.12, 0.325, 0.11])
        self.pl_dat.add_number(fig, ax_im, order=2)
        s_range = 20

        # ax = plt.axes([0.525,0.1,0.375,0.275])
        fields = np.zeros((nbin, nSes))
        for i, s in enumerate(np.where(self.status["sessions"])[0]):
            # idx_PC = np.where(self.fields['status'][:,s,:]>=3)
            idx_PC = np.where(self.status["fields"][:, s, :])
            # idx_PC = np.where(~np.isnan(self.fields['location'][:,s,:]))
            # fields[s,:] = np.nansum(self.fields['p_x'][:,s,:,:],1).sum(0)
            fields[:, s] = np.nansum(self.fields["p_x"][idx_PC[0], s, idx_PC[1], :], 0)
            fields[:, s] /= fields[:, s].sum()
            # ax.plot(gauss_smooth(fields[:,s],1),'-',color=[0.5,0.5,0.5],linewidth=0.3,alpha=0.5)
        fields = gauss_smooth(fields, (2, 0))

        im = ax_im.imshow(
            fields, origin="lower", aspect="auto", cmap="hot", interpolation="none"
        )  # ,clim=[0,1])
        ax_im.set_xlim([-0.5, nSes - 0.5])
        ax_im.set_xlim(
            [
                -0.5 + np.where(self.status["sessions"])[0][0],
                np.where(self.status["sessions"])[0][-1] - 0.5,
            ]
        )
        ax_im.set_ylim([0, 100])

        cbaxes = plt.axes([0.44, 0.35, 0.01, 0.2])
        h_cb = plt.colorbar(im, cax=cbaxes)
        h_cb.set_label("place field \ndensity", fontsize=8)
        h_cb.set_ticks([])

        ax_im.set_xlabel("session")
        ax_im.set_ylabel("position")

        self.pl_dat.add_number(fig, ax, order=3)
        s_arr = [2, 9, 15, 30]
        s_arr2 = [0, 5, 17, 88, 97]
        # s_arr += np.where(self.status['sessions'])[0][0]
        for i in range(len(s_arr)):
            col = [0.8 - 0.2 * i, 0.8 - 0.2 * i, 0.8 - 0.2 * i]
            ax_im.annotate(
                text="",
                xy=(s_arr[i], 100),
                xytext=(s_arr[i], 110),
                fontsize=6,
                annotation_clip=False,
                arrowprops=dict(arrowstyle="->", color=col),
            )
            # ax_im.annotate(s='',xy=(s_arr[i+1]-1,100),xytext=(s_arr[i+1]-1,110),fontsize=6,annotation_clip=False,arrowprops=dict(arrowstyle='->',color=col))

            ax.plot(np.nanmean(fields[:, s_arr2[i] : s_arr2[i + 1]], 1), color=col)

        ax.set_xlim([0, 100])
        ax.set_yticks([])
        ax.set_xlabel("position")
        ax.set_ylabel("density")
        self.pl_dat.remove_frame(ax, ["top", "right"])

        # s_arr2 = np.array([1,14,34])
        # s_arr2 += np.where(self.status['sessions'])[0][0]

        props = dict(boxstyle="round", facecolor="w", alpha=0.8)
        for j, s in enumerate(s_arr):
            if s < nSes:
                ax = plt.axes([0.075 + 0.1 * j, 0.65, 0.075, 0.275])
                if j == 0:
                    self.pl_dat.add_number(fig, ax, order=1, offset=[-100, 50])
                idxes_tmp = np.where(
                    self.status["fields"][:, s, :]
                    & (self.stats["SNR_comp"][:, s] > 2)[..., np.newaxis]
                    & (self.stats["r_values"][:, s] > 0)[..., np.newaxis]
                    & (self.matching["score"][:, s, 0] > 0.5)[..., np.newaxis]
                )
                idxes = idxes_tmp[0]
                sort_idx = np.argsort(
                    self.fields["location"][idxes_tmp[0], s, idxes_tmp[1], 0]
                )
                sort_idx = idxes[sort_idx]
                nID = len(sort_idx)

                firingmap = self.stats["firingmap"][sort_idx, s, :]
                firingmap = gauss_smooth(firingmap, [0, 2])
                firingmap = firingmap - np.nanmin(firingmap, 1)[:, np.newaxis]
                # firingmap = firingmap / np.nanmax(firingmap,1)[:,np.newaxis]
                im = ax.imshow(
                    firingmap, aspect="auto", origin="upper", cmap="jet", clim=[0, 5]
                )
                ax.text(
                    5, nID * 0.95, "n = %d" % nID, bbox=props, color="k", fontsize=6
                )
                ax.text(
                    95,
                    nID / 10,
                    "Session %d" % (s + 1),
                    bbox=props,
                    color="k",
                    fontsize=6,
                    ha="right",
                )
                self.pl_dat.remove_frame(ax)
                ax.set_xticks([])
                ax.set_yticks([])

        cbaxes = plt.axes([0.46, 0.825, 0.01, 0.1])
        h_cb = plt.colorbar(im, cax=cbaxes)
        h_cb.set_label("$Ca^{2+}$", fontsize=8)
        h_cb.set_ticks([0, 5])
        h_cb.set_ticklabels(["low", "high"])

        RW_rec = np.zeros(nSes)
        slowing = np.zeros(nSes)
        sig = np.zeros((nSes, 3)) * np.NaN
        if_fr = np.zeros((nSes, 3)) * np.NaN
        oof_fr = np.zeros((nSes, 2, 3)) * np.NaN
        rel = np.zeros((nSes, 3)) * np.NaN
        for i, s in enumerate(np.where(self.status["sessions"])[0]):
            try:
                RW_rec[s] = self.behavior["performance"][s]["RW_reception"].mean()
                slowing[s] = self.behavior["performance"][s]["slowDown"].mean()
                idx_fields = np.where(self.status["fields"][:, s, :])
            except:
                # print('pass')
                continue
                # pass
            # print(idx_fields)

            try:
                oof_fr[s, 0, 0] = self.stats["oof_firingrate_adapt"][
                    self.status["activity"][:, s, 1]
                    & (~self.status["activity"][:, s, 2]),
                    s,
                ].mean()
                oof_fr[s, 0, 1:] = np.nanpercentile(
                    self.stats["oof_firingrate_adapt"][
                        self.status["activity"][:, s, 1]
                        & (~self.status["activity"][:, s, 2]),
                        s,
                    ],
                    [5, 95],
                )
                oof_fr[s, 1, 0] = self.stats["oof_firingrate_adapt"][
                    self.status["activity"][:, s, 2], s
                ].mean()
                oof_fr[s, 1, 1:] = np.nanpercentile(
                    self.stats["oof_firingrate_adapt"][
                        self.status["activity"][:, s, 2], s
                    ],
                    [5, 95],
                )
            except:
                pass

            if len(idx_fields[0]) > 0:
                sig[s, 0] = self.fields["width"][idx_fields[0], s, idx_fields[1]].mean()
                sig[s, 1:] = np.nanpercentile(
                    self.fields["width"][idx_fields[0], s, idx_fields[1]], [5, 95]
                )

                if_fr[s, 0] = self.stats["if_firingrate_adapt"][
                    idx_fields[0], s, idx_fields[1]
                ].mean()
                if_fr[s, 1:] = np.nanpercentile(
                    self.stats["if_firingrate_adapt"][idx_fields[0], s, idx_fields[1]],
                    [5, 95],
                )

                rel[s, 0] = self.fields["reliability"][
                    idx_fields[0], s, idx_fields[1]
                ].mean()
                rel[s, 1:] = np.nanpercentile(
                    self.fields["reliability"][idx_fields[0], s, idx_fields[1]], [5, 95]
                )

        ax = plt.axes([0.65, 0.85, 0.3, 0.1])
        ax.plot(
            np.where(self.status["sessions"])[0],
            self.status["activity"][:, self.status["sessions"], 2].sum(0)
            / self.behavior["time_active"][self.status["sessions"]],
            "k.",
            markersize=2,
        )
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax.set_ylabel("$t_{active}$")
        ax.set_xticklabels([])

        ax = plt.axes([0.65, 0.7, 0.3, 0.1])
        self.pl_dat.plot_with_confidence(
            ax, range(nSes), sig[:, 0], sig[:, 1:].T, col="b"
        )
        # mask_sig = np.ma.masked_array(self.fields['width'][...,0],mask=~self.status["fields"])
        # ax.plot(mask_sig.mean(2).mean(0),'b')
        ax.set_ylim([0, 20])
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax.set_ylabel("$\\sigma$")
        ax.set_xticklabels([])

        ax = plt.axes([0.65, 0.55, 0.3, 0.1])
        self.pl_dat.plot_with_confidence(
            ax, range(nSes), oof_fr[:, 0, 0], oof_fr[:, 0, 1:].T, col="k"
        )
        self.pl_dat.plot_with_confidence(
            ax, range(nSes), oof_fr[:, 1, 0], oof_fr[:, 1, 1:].T, col="b"
        )
        ax.set_xticklabels([])
        ax.set_ylim([0, 3.0])
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax.set_ylabel("$\\nu^-$")

        ax = plt.axes([0.65, 0.4, 0.3, 0.1])
        self.pl_dat.plot_with_confidence(
            ax, range(nSes), if_fr[:, 0], if_fr[:, 1:].T, col="b"
        )
        ax.set_ylim([0, 10.0])
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax.set_ylabel("$\\nu^*$")
        ax.set_xticklabels([])

        ax = plt.axes([0.65, 0.25, 0.3, 0.1])
        self.pl_dat.plot_with_confidence(
            ax, range(nSes), rel[:, 0], rel[:, 1:].T, col="b"
        )
        ax.set_ylim([0, 1])
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax.set_ylabel("$a$")
        ax.set_xticklabels([])

        ax = plt.axes([0.65, 0.1, 0.3, 0.1])
        mask_MI = np.ma.masked_array(
            self.stats["MI_value"],
            mask=~(
                self.status["activity"][..., 1] & (~self.status["activity"][..., 2])
            ),
        )
        self.pl_dat.plot_with_confidence(
            ax, range(nSes), mask_MI.mean(0), mask_MI.std(0), col="k"
        )
        # ax.plot(mask_MI.mean(),'k')
        mask_MI = np.ma.masked_array(
            self.stats["MI_value"], mask=~self.status["activity"][..., 2]
        )
        self.pl_dat.plot_with_confidence(
            ax, range(nSes), mask_MI.mean(0), mask_MI.std(0), col="b"
        )
        # ax.plot(mask_MI.mean(0),'b')
        ax.set_ylim([0, 1.5])
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax.set_ylabel("MI")

        # ax.plot(self.sessions['time_active'],self.status['activity'][...,2].sum(0),'k.',markersize=2)
        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig("timedep_parameters")

    def plot_neuron_cluster_stats(self, sv=False):

        print("plot cluster specific statistics (stability, etc)")

        nC = self.data["nC"]
        status, status_dep = get_status_arr(self)
        status_arr = ["act", "code", "stable"]

        # ds_max = 2
        # nC_good = np.sum(self.status['clusters'])

        # reprocess = True
        # if (not ('p_post_c' in self.stats.keys())) or reprocess:
        #     self.stats['p_post_c'] = {}
        #     for status_key in status_arr:
        #         self.stats['p_post_c'][status_key] = {}
        #         for status2_key in status_arr:
        #             self.stats['p_post_c'][status_key][status2_key] = np.zeros((nC,ds_max+1,2,2))*np.NaN

        #     for ds in range(1,ds_max):

        #         ### activity -> coding
        #         ## what's the general state before obtaining a place field? (active / silent?; above chance level?
        #         for c in tqdm(np.where(self.status['clusters'])[0]):

        #             counts = {}
        #             for status_key in status_arr:
        #                 counts[status_key] = {}
        #                 for status2_key in status_arr:
        #                     counts[status_key][status2_key] = np.zeros(3)

        #             for s in np.where(self.status['sessions'])[0][:-ds]:
        #                 if self.status['sessions'][s+ds]:

        #                     for status_key in status_arr:
        #                         if status[status_key][c,s]:
        #                             for status2_key in status_arr:
        #                                 if status_dep[status2_key][c,s+ds]:
        #                                     counts[status_key][status2_key][0] += 1

        #                                 if status[status2_key][c,s+ds] & status_dep[status2_key][c,s+ds]:
        #                                     counts[status_key][status2_key][1] += 1
        #                                 elif status_dep[status2_key][c,s+ds]:
        #                                     counts[status_key][status2_key][2] += 1

        #             for status_key in status_arr:
        #                 for status2_key in status_arr:
        #                     self.stats['p_post_c'][status_key][status2_key][c,ds,0,0] = counts[status_key][status2_key][1]/counts[status_key][status2_key][0] if counts[status_key][status2_key][0]>0 else np.NaN
        #                     self.stats['p_post_c'][status_key][status2_key][c,ds,0,1] = counts[status_key][status2_key][2]/counts[status_key][status2_key][0] if counts[status_key][status2_key][0]>0 else np.NaN

        # idx_c = np.where(self.status['clusters'])[0]

        subpop_lim = 0.95
        idx_c_stable = np.where(
            self.stats["p_post_c"]["stable"]["act"][:, 1, 0] > subpop_lim
        )
        idx_c_code = np.where(
            self.stats["p_post_c"]["code"]["act"][:, 1, 0] > subpop_lim
        )

        plt.figure(figsize=(7, 5), dpi=self.pl_dat.sv_opt["dpi"])

        ax = plt.subplot(441)
        ax.hist(self.stats["p_post_c"]["act"]["act"][:, 1, 0], np.linspace(0, 1, 51))
        ax.set_xlabel("$p(\\alpha_{s+1}^+|\\alpha_s^+)$")
        ax = plt.subplot(442)
        ax.hist(self.stats["p_post_c"]["code"]["act"][:, 1, 0], np.linspace(0, 1, 51))
        ax.plot([subpop_lim, subpop_lim], [0, ax.get_ylim()[1]], "k--")
        ax.set_xlabel("$p(\\alpha_{s+1}^+|\\beta_s^+)$")
        ax = plt.subplot(445)
        ax.hist(self.stats["p_post_c"]["stable"]["act"][:, 1, 0], np.linspace(0, 1, 51))
        ax.plot([subpop_lim, subpop_lim], [0, ax.get_ylim()[1]], "k--")
        ax.set_xlabel("$p(\\alpha_{s+1}^+|\\gamma_s^+)$")
        ax = plt.subplot(446)
        ax.hist(
            self.stats["p_post_c"]["stable"]["code"][:, 1, 0], np.linspace(0, 1, 51)
        )
        ax.set_xlabel("$p(\\beta_{s+1}^+|\\beta_s^+)$")

        dense = True
        ax = plt.subplot(4, 2, 6)
        _, _, patches = ax.hist(
            self.stats["p_post_c"]["act"]["act"][:, 1, 0],
            np.linspace(0, 1, 51),
            alpha=0.5,
            color="k",
            cumulative=True,
            histtype="step",
            density=dense,
            label="$\\alpha^+$",
        )
        patches[0].set_xy(patches[0].get_xy()[:-1])
        _, _, patches = ax.hist(
            self.stats["p_post_c"]["code"]["act"][:, 1, 0],
            np.linspace(0, 1, 51),
            alpha=0.5,
            color="b",
            cumulative=True,
            histtype="step",
            density=dense,
            label="$\\beta^+$",
        )
        patches[0].set_xy(patches[0].get_xy()[:-1])
        _, _, patches = ax.hist(
            self.stats["p_post_c"]["stable"]["act"][:, 1, 0],
            np.linspace(0, 1, 51),
            alpha=0.5,
            color="r",
            cumulative=True,
            histtype="step",
            density=dense,
            label="$\\gamma^+$",
        )
        patches[0].set_xy(patches[0].get_xy()[:-1])
        ax.set_xlabel("$p(\\alpha^+|X)$")
        ax.legend(fontsize=8, loc="upper left")

        ax = plt.subplot(4, 2, 8)
        _, _, patches = ax.hist(
            self.stats["p_post_c"]["act"]["code"][:, 1, 0],
            np.linspace(0, 1, 51),
            alpha=0.5,
            color="k",
            cumulative=True,
            histtype="step",
            density=dense,
            label="$\\alpha^+$",
        )
        patches[0].set_xy(patches[0].get_xy()[:-1])
        _, _, patches = ax.hist(
            self.stats["p_post_c"]["code"]["code"][:, 1, 0],
            np.linspace(0, 1, 51),
            alpha=0.5,
            color="b",
            cumulative=True,
            histtype="step",
            density=dense,
            label="$\\beta^+$",
        )
        patches[0].set_xy(patches[0].get_xy()[:-1])
        _, _, patches = ax.hist(
            self.stats["p_post_c"]["stable"]["code"][:, 1, 0],
            np.linspace(0, 1, 51),
            alpha=0.5,
            color="r",
            cumulative=True,
            histtype="step",
            density=dense,
            label="$\\gamma^+$",
        )
        patches[0].set_xy(patches[0].get_xy()[:-1])
        ax.set_xlabel("$p(\\beta^+|X)$")
        ax.legend(fontsize=8, loc="upper left")

        par_key = "MI_value"
        ax = plt.subplot(443)
        ax.hist(
            self.fields["location"][..., 0].flat,
            np.linspace(0, 100, 51),
            density=True,
            alpha=0.5,
        )
        # ax.hist(self.fields['location'][idx_c_stable,:,:,0].flat,np.linspace(0,100,51),density=True,alpha=0.5)
        idx_stable = np.where(status["stable"] == 1)
        # print(idx_stable.shape)
        # c_idx_stable = idx_c[idx_stable[0]]
        ax.hist(
            self.fields["location"][idx_stable[0], idx_stable[1], :, 0].flat,
            np.linspace(0, 100, 51),
            density=True,
            alpha=0.5,
        )
        ax.set_xlabel("$\\theta (> p(\\alpha|\\gamma))$")

        ax = plt.subplot(444)
        try:
            ax.hist(
                self.fields[par_key].flat,
                np.linspace(0, np.nanmax(self.fields[par_key]), 51),
                density=True,
                alpha=0.5,
            )
            ax.hist(
                self.fields[par_key][idx_stable[0], ...].flat,
                np.linspace(0, np.nanmax(self.fields[par_key]), 51),
                density=True,
                alpha=0.5,
            )
        except:
            ax.hist(
                self.stats[par_key].flat,
                np.linspace(0, np.nanmax(self.stats[par_key]), 51),
                density=True,
                alpha=0.5,
            )
            ax.hist(
                self.stats[par_key][idx_stable[0], ...].flat,
                np.linspace(0, np.nanmax(self.stats[par_key]), 51),
                density=True,
                alpha=0.5,
            )

        ax = plt.subplot(447)
        ax.hist(
            self.fields["location"][..., 0].flat,
            np.linspace(0, 100, 51),
            density=True,
            alpha=0.5,
        )
        ax.hist(
            self.fields["location"][idx_c_code, :, :, 0].flat,
            np.linspace(0, 100, 51),
            density=True,
            alpha=0.5,
        )
        ax.set_xlabel("$\\theta (> p(\\alpha|\\beta))$")

        ax = plt.subplot(448)
        try:
            ax.hist(
                self.fields[par_key].flat,
                np.linspace(0, np.nanmax(self.fields[par_key]), 51),
                density=True,
                alpha=0.5,
            )
            ax.hist(
                self.fields[par_key][idx_c_code, ...].flat,
                np.linspace(0, np.nanmax(self.fields[par_key]), 51),
                density=True,
                alpha=0.5,
            )
        except:
            ax.hist(
                self.stats[par_key].flat,
                np.linspace(0, np.nanmax(self.stats[par_key]), 51),
                density=True,
                alpha=0.5,
            )
            ax.hist(
                self.stats[par_key][idx_c_code, ...].flat,
                np.linspace(0, np.nanmax(self.stats[par_key]), 51),
                density=True,
                alpha=0.5,
            )
        # ax = plt.subplot(444)
        # ax.hist(self.fields['reliability'][idx_c_stable,:,:,0].flat,np.linspace(0,100,101))

        ax = plt.subplot(223)
        # ax.scatter(self.stats['p_post_c']['act'][:,1,0,0]+0.02*np.random.rand(nC_good),self.stats['p_post_c']['code'][:,1,0,1]+0.02*np.random.rand(nC_good),s=self.status['activity'][...,1].sum(1)/40,color='k',edgecolors='none')
        ax.scatter(
            self.stats["p_post_c"]["act"]["act"][:, 1, 0] + 0.02 * np.random.rand(nC),
            self.stats["p_post_c"]["code"]["code"][:, 1, 0] + 0.02 * np.random.rand(nC),
            s=self.status["activity"][..., 1].sum(1) / 40,
            color="k",
            edgecolors="none",
        )
        ax.set_xlabel("$p(\\alpha_{s+1}^+|\\alpha_{s}^+)$")
        ax.set_ylabel("$p(\\beta_{s+1}^+|\\beta_{s}^+)$")

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig("individual_neurons")

        plt.figure()
        plt.plot()
        plt.show(block=False)

    def plot_time_vs_experience(self, sv=False):

        print("### plot time dependence of dynamics ###")

        nSes = self.data["nSes"]
        ### ds > 0
        # p = {}
        SD = 1.96
        maxSes = 10
        sig_theta = self.stability["all"]["mean"][0, 2]

        self.mouse_data = {}
        self.mouse_data["t_measures"] = np.linspace(0, 200, nSes)

        trials = np.cumsum(self.behavior["trial_ct"])
        diff = {
            "t": (
                self.mouse_data["t_measures"][np.newaxis, :]
                - self.mouse_data["t_measures"][:, np.newaxis]
            ).astype("int"),
            "nights": (
                (
                    self.mouse_data["t_measures"][np.newaxis, :]
                    - self.mouse_data["t_measures"][:, np.newaxis]
                ).astype("int")
                + 10
            )
            // 24,
            "s": (
                np.arange(nSes)[np.newaxis, :] - np.arange(nSes)[:, np.newaxis]
            ).astype("int"),
            "trials": ((trials[np.newaxis, :] - trials[:, np.newaxis]) // 10).astype(
                "int"
            )
            * 10,
        }

        s_bool = np.zeros(nSes, "bool")
        s_bool[17:87] = True
        # s_bool[D.sessions[mouse]['steady'][0]:D.sessions[mouse]['steady'][1]] = True
        s_bool[~self.status["sessions"]] = False

        t_start = time.time()
        s1_shifts, s2_shifts, f1, f2 = np.unravel_index(
            self.compare["pointer"].col,
            (
                nSes,
                nSes,
                self.params["field_count_max"],
                self.params["field_count_max"],
            ),
        )
        c_shifts = self.compare["pointer"].row

        dT_shifts = (
            self.mouse_data["t_measures"][s2_shifts]
            - self.mouse_data["t_measures"][s1_shifts]
        )
        nights_shifts = (dT_shifts + 10) // 24

        # print(diff)
        arr = {}  #'s':     np.unique(np.triu(diff['s'])),
        #'t':     np.unique(np.triu(diff['t']))}
        for key in diff.keys():
            arr[key] = np.unique(np.triu(diff[key]))
        # ds_arr = np.unique(np.triu(diff['s']))
        # dt_arr = np.unique(np.triu(diff['t']))

        def get_p_rec(diff, compare, key1, key2, s_bool):

            key1_arr = np.unique(np.triu(diff[key1]))
            key2_arr = np.unique(np.triu(diff[key2]))

            p_rec = {
                "act": np.zeros((len(key1_arr), len(key2_arr), 2)) * np.NaN,
                "PC": np.zeros((len(key1_arr), len(key2_arr), 2)) * np.NaN,
                "PF": np.zeros((len(key1_arr), len(key2_arr), 2)) * np.NaN,
            }
            pval = {
                "act": np.zeros(len(key1_arr)) * np.NaN,
                "PC": np.zeros(len(key1_arr)) * np.NaN,
                "PF": np.zeros(len(key1_arr)) * np.NaN,
            }

            s1_shifts, s2_shifts, f1, f2 = np.unravel_index(
                compare["pointer"].col,
                (
                    nSes,
                    nSes,
                    self.params["field_count_max"],
                    self.params["field_count_max"],
                ),
            )
            c_shifts = compare["pointer"].row
            Ds = s2_shifts - s1_shifts

            N_ref = {}
            tmp = {}
            for dx in key1_arr:  # min(nSes,30)):
                x = np.where(key1_arr == dx)

                x_tmp = {"act": [], "PC": [], "PF": []}

                for dy in np.unique(diff[key2][diff[key1] == dx]):
                    y = np.where(key2_arr == dy)

                    for key in p_rec.keys():
                        N_ref[key] = 0
                        tmp[key] = []

                    s1_arr, s2_arr = np.where((diff[key1] == dx) & (diff[key2] == dy))

                    for s1, s2 in zip(s1_arr, s2_arr):
                        if s_bool[s1] & s_bool[s2] & (s1 != s2):
                            overlap = (
                                self.status["activity"][
                                    self.status["activity"][:, s1, 1], s2, 1
                                ]
                                .sum(0)
                                .astype("float")
                            )
                            N_ref["act"] = self.status["activity"][:, s1, 1].sum(0)
                            tmp["act"].append(overlap / N_ref["act"])

                            overlap_PC = (
                                self.status["activity"][
                                    self.status["activity"][:, s1, 2], s2, 2
                                ]
                                .sum(0)
                                .astype("float")
                            )
                            N_ref["PC"] = self.status["activity"][
                                self.status["activity"][:, s1, 2], s2, 1
                            ].sum(0)
                            tmp["PC"].append(overlap_PC / N_ref["PC"])

                            idx = np.where((s1_shifts == s1) & (s2_shifts == s2))[0]
                            N_ref["PF"] = len(idx)
                            idx_shifts = (
                                self.compare["pointer"].data[idx].astype("int") - 1
                            )
                            shifts = self.compare["shifts"][idx_shifts]
                            N_stable = (np.abs(shifts) < (SD * sig_theta)).sum()

                            tmp["PF"].append(N_stable / N_ref["PF"])

                    for key in p_rec.keys():
                        if N_ref[key] > 0:
                            p_rec[key][x, y, :] = [np.mean(tmp[key]), np.std(tmp[key])]
                            x_tmp[key].append(tmp[key])

                # print(x_tmp)
                for key in p_rec.keys():
                    if len(x_tmp[key]) > 1:
                        try:
                            # print(x_tmp[key])
                            # res = sstats.f_oneway(*x_tmp[key])
                            res = sstats.kruskal(*x_tmp[key])
                            # res = sstats.mannwhitneyu(*x_tmp[key])
                            # res = sstats.ttest_ind(*x_tmp[key])
                            # print(res)
                            pval[key][x] = res.pvalue
                        except:
                            pass
                ### now, do anova to test

            return p_rec, pval

        # print(sig_theta)
        # key1 = 't'
        # key2 = 's'
        key_arr = ["s", "nights", "trials", "t"]
        p_rec = {}
        pval = {}
        for i, key1 in enumerate(key_arr):
            for key2 in key_arr[:]:
                key_pair = "%s_%s" % (key1, key2)
                p_rec[key_pair], pval[key_pair] = get_p_rec(
                    diff, self.compare, key1, key2, s_bool
                )

        col = ["k", "tab:red", "tab:blue"]
        fig = plt.figure(figsize=(7, 5), dpi=self.pl_dat.sv_opt["dpi"])

        # print(diff)
        ax = plt.axes([0.1, 0.85, 0.125, 0.11])
        self.pl_dat.add_number(fig, ax, order=1, offset=[-175, 25])
        plt.plot(diff["s"][0, :], "k.", markersize=1.5)
        ax.set_ylabel("$\sum$ s")
        ax.yaxis.set_label_coords(-0.4, 0.5)
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax1 = plt.axes([0.325, 0.85, 0.25, 0.11])
        self.pl_dat.add_number(fig, ax1, order=2, offset=[-125, 25])

        ax = plt.axes([0.1, 0.6, 0.125, 0.11])
        self.pl_dat.add_number(fig, ax, order=3)
        plt.plot(diff["nights"][0, :], "k.", markersize=1.5)
        ax.set_ylabel("$\sum$ nights")
        ax.yaxis.set_label_coords(-0.4, 0.5)
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax2 = plt.axes([0.325, 0.6, 0.25, 0.11])
        self.pl_dat.add_number(fig, ax2, order=4, offset=[-125, 50])

        ax = plt.axes([0.1, 0.35, 0.125, 0.11])
        self.pl_dat.add_number(fig, ax, order=6)
        plt.plot(diff["trials"][0, :], "k.", markersize=1.5)
        ax.set_ylabel("$\sum$ trials")
        ax.yaxis.set_label_coords(-0.4, 0.5)
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax3 = plt.axes([0.325, 0.35, 0.25, 0.11])
        self.pl_dat.add_number(fig, ax3, order=7, offset=[-125, 50])

        ax = plt.axes([0.1, 0.1, 0.125, 0.11])
        self.pl_dat.add_number(fig, ax, order=9)
        ax.plot(diff["t"][0, :], "k.", markersize=1.5)
        ax.set_ylabel("$\sum $ t")
        ax.yaxis.set_label_coords(-0.4, 0.5)
        ax.set_xlabel("$\sum s$")
        self.pl_dat.remove_frame(ax, ["top", "right"])
        ax4 = plt.axes([0.325, 0.1, 0.25, 0.11])
        self.pl_dat.add_number(fig, ax4, order=10, offset=[-125, 50])
        key_label = ["activation", "coding", "field stability"]
        for i, key in enumerate(p_rec["s_t"].keys()):
            mask = ~np.isnan(np.nanmean(p_rec["s_t"][key][..., 0], 1))
            self.pl_dat.plot_with_confidence(
                ax1,
                arr["s"][mask],
                np.nanmean(p_rec["s_t"][key][..., 0], 1)[mask],
                np.nanstd(p_rec["s_t"][key][..., 0], 1)[mask],
                col=col[i],
                label=key_label[i],
            )
            mask = ~np.isnan(np.nanmean(p_rec["s_t"][key][..., 0], 0))
            self.pl_dat.plot_with_confidence(
                ax4,
                arr["t"][mask],
                np.nanmean(p_rec["s_t"][key][..., 0], 0)[mask],
                np.nanstd(p_rec["s_t"][key][..., 0], 0)[mask],
                col=col[i],
            )
            mask = ~np.isnan(np.nanmean(p_rec["trials_nights"][key][..., 0], 1))
            self.pl_dat.plot_with_confidence(
                ax3,
                arr["trials"][mask],
                np.nanmean(p_rec["trials_nights"][key][..., 0], 1)[mask],
                np.nanstd(p_rec["trials_nights"][key][..., 0], 1)[mask],
                col=col[i],
            )
            mask = ~np.isnan(np.nanmean(p_rec["trials_nights"][key][..., 0], 0))
            self.pl_dat.plot_with_confidence(
                ax2,
                arr["nights"][mask],
                np.nanmean(p_rec["trials_nights"][key][..., 0], 0)[mask],
                np.nanstd(p_rec["trials_nights"][key][..., 0], 0)[mask],
                col=col[i],
            )
            # pl_dat.add_number(fig,ax,order=6)
        ax1.legend(fontsize=12, loc="lower left", bbox_to_anchor=[1.1, -0.3])
        ax1.set_ylim([0, 1])
        ax2.set_ylim([0, 1])
        ax3.set_ylim([0, 1])
        ax4.set_ylim([0, 1])
        ax1.set_xlabel("$\Delta s$")
        ax2.set_xlabel("$\Delta$ nights")
        ax3.set_xlabel("$\Delta$ trials")
        ax4.set_xlabel("$\Delta t$")
        ax1.set_ylabel("$p(+|+)$", rotation="horizontal", fontsize=8)
        ax1.yaxis.set_label_coords(0.1, 1.1)
        ax2.set_ylabel("$p(+|+)$", rotation="horizontal", fontsize=8)
        ax2.yaxis.set_label_coords(0.1, 1.1)
        ax3.set_ylabel("$p(+|+)$", rotation="horizontal", fontsize=8)
        ax3.yaxis.set_label_coords(0.1, 1.1)
        ax4.set_ylabel("$p(+|+)$", rotation="horizontal", fontsize=8)
        ax4.yaxis.set_label_coords(0.1, 1.1)
        self.pl_dat.remove_frame(ax1, ["top", "right"])
        self.pl_dat.remove_frame(ax2, ["top", "right"])
        self.pl_dat.remove_frame(ax3, ["top", "right"])
        self.pl_dat.remove_frame(ax4, ["top", "right"])

        key1 = "s"
        for j, key2 in enumerate(key_arr):
            if key1 == key2:
                continue
            key_pairs = "%s_%s" % (key1, key2)
            key_pairs_rev = "%s_%s" % (key2, key1)

            ax = plt.axes([0.7, 0.85 - j * 0.25, 0.125, 0.125])
            self.pl_dat.add_number(fig, ax, order=2 + j * 3)
            ax.plot([0, arr[key1][-1]], [0.01, 0.01], "k--", linewidth=0.5)
            mask = ~np.isnan(np.nanmean(p_rec[key_pairs][key][..., 0], 1))
            for i, key in enumerate(p_rec["s_t"].keys()):
                pval[key_pairs][key][pval[key_pairs][key] < 10 ** (-6)] = 10 ** (-6)
                ax.plot(
                    arr[key1][mask],
                    pval[key_pairs][key][mask],
                    ".",
                    color=col[i],
                    markersize=2,
                )
            ax.set_yscale("log")
            ax.set_ylim([0.9 * 10 ** (-6), 1])
            ax.set_xlim([0, arr[key1][mask][-1]])
            ax.set_ylabel("p-value", fontsize=8, rotation="horizontal")
            ax.yaxis.set_label_coords(-0.2, 1.1)

            ax.set_xlabel("$\Delta $%s" % key1)
            ax = plt.axes([0.85, 0.85 - j * 0.25, 0.125, 0.125])
            ax.plot([0, arr[key2][-1]], [0.01, 0.01], "k--", linewidth=0.5)
            mask = ~np.isnan(np.nanmean(p_rec[key_pairs_rev][key][..., 0], 1))
            for i, key in enumerate(p_rec["s_t"].keys()):
                pval[key_pairs_rev][key][pval[key_pairs_rev][key] < 10 ** (-6)] = (
                    10 ** (-6)
                )
                ax.plot(
                    arr[key2][mask],
                    pval[key_pairs_rev][key][mask],
                    ".",
                    color=col[i],
                    markersize=2,
                )
            ax.set_xlabel("$\Delta $%s" % key2)
            ax.set_yscale("log")
            ax.set_yticklabels([])
            ax.set_ylim([0.9 * 10 ** (-6), 1])
            ax.set_xlim([0, arr[key2][mask][-1]])

        # ax2.set_xlabel('$\Delta$ nights')
        # ax3.set_xlabel('$\Delta t$')
        plt.show(block=False)

        if sv:
            self.pl_dat.save_fig("time_in_HC")
        return

        print(self.mouse_data["t_measures"][s2_shifts])

        # print(c_shifts.shape)
        # print(dT_shifts.shape)
        dT_arr = [4, 20, 24, 28, 44, 48, 52, 64, 68, 72, 84, 88, 92]

        # N_stable = np.zeros(nSes)*np.NaN
        # N_total = np.zeros(nSes)*np.NaN     ### number of PCs which could be stable
        # fig = plt.figure()

        p_rec = {
            "act": np.zeros((nSes, nSes)) * np.NaN,
            "PC": np.zeros((nSes, nSes)) * np.NaN,
            "PF": np.zeros((nSes, nSes)) * np.NaN,
        }

        for ds in range(1, nSes):  # min(nSes,30)):
            session_bool = np.where(
                np.pad(self.status["sessions"][ds:], (0, ds), constant_values=False)
                & np.pad(self.status["sessions"][:], (0, 0), constant_values=False)
            )[0]
            for s1 in session_bool:
                overlap = (
                    self.status["activity"][
                        self.status["activity"][:, s1, 1], s1 + ds, 1
                    ]
                    .sum(0)
                    .astype("float")
                )
                N_ref = self.status["activity"][:, s1, 1].sum(0)
                p_rec["act"][ds, s1] = overlap / N_ref

                overlap = (
                    self.status["activity"][
                        self.status["activity"][:, s1, 2], s1 + ds, 2
                    ]
                    .sum(0)
                    .astype("float")
                )
                N_ref = self.status["activity"][
                    self.status["activity"][:, s1, 2], s1 + ds, 1
                ].sum(0)
                p_rec["PC"][ds, s1] = overlap / N_ref

                Ds = s2_shifts - s1_shifts
                idx = np.where((s1_shifts == s1) & (Ds == ds))[0]

                N_data = len(idx)

                idx_shifts = self.compare["pointer"].data[idx].astype("int") - 1
                shifts = self.compare["shifts"][idx_shifts]
                N_stable = (
                    np.abs(shifts) < (SD * self.stability["all"]["mean"][ds, 2])
                ).sum()

                p_rec["PF"][ds, s1] = N_stable / N_data
                # N_total[ds] = self.status["fields"][:,session_bool,:].sum()
        # print(p_rec['PF'])
        # print(np.nanmean(p_rec['PF'],1))
        # print(recurr)
        # return recurr

        diff = {
            "t": (
                self.mouse_data["t_measures"][np.newaxis, self.status["sessions"]]
                - self.mouse_data["t_measures"][self.status["sessions"], np.newaxis]
            ).astype("int"),
            "s": np.where(self.status["sessions"])[0][np.newaxis, :]
            - np.where(self.status["sessions"])[0][:, np.newaxis],
            "trial": (
                (
                    trials[np.newaxis, self.status["sessions"]]
                    - trials[self.status["sessions"], np.newaxis]
                )
                // 10
            ).astype("int"),
        }

        # s_good = np.where(self.status['sessions'])[0]

        ### test same ds, different dt

        ds_arr = np.unique(np.triu(diff["s"]))  # [1:]
        dt_arr = np.unique(np.triu(diff["t"]))  # [1:]

        def calc_pval(diff, p_rec, key1, key1_arr, key2):

            s_good = np.where(self.status["sessions"])[0]
            nSteps = len(key1_arr)
            ## preallocate arrays
            pval = {
                "act": np.zeros(nSteps) * np.NaN,
                "PC": np.zeros(nSteps) * np.NaN,
                "PF": np.zeros(nSteps) * np.NaN,
            }

            nt = len(dt_arr)
            p = {
                "act": np.zeros((nSteps, nt)) * np.NaN,
                "PC": np.zeros((nSteps, nt)) * np.NaN,
                "PF": np.zeros((nSteps, nt)) * np.NaN,
            }

            for i, dx in enumerate(
                key1_arr
            ):  ## iterate through x-axis - values assumed to return stable statistics
                # print('ds: %d'%ds)
                dy_tmp = np.unique(
                    diff[key2][diff[key1] == dx]
                )  ## find all differences on second dimensions according to fixed dx along x-axis
                # print(' ---- dx: %d ----'%dx)
                # print(dy_tmp)
                for key in [
                    "act",
                    "PC",
                    "PF",
                ]:  ## for each of the different hierarchies do...
                    tmp = []

                    for (
                        dy
                    ) in dy_tmp:  ## iterate through different realizations of fixed dx

                        ## find all shifts with dx and dy
                        s1, s2 = np.where((diff[key1] == dx) & (diff[key2] == dy))
                        # print(s1)
                        # print(s2)
                        s1 = s_good[s1]
                        s2 = s_good[s2]
                        ds = s2[0] - s1[0]
                        # print(ds)
                        if len(s1) > 1:
                            # print(p_rec[key][ds,s1])
                            tmp.append(p_rec[key][ds, s1])
                            # print(tmp)

                        dt = np.where(
                            dt_arr
                            == (
                                self.mouse_data["t_measures"][s2[0]]
                                - self.mouse_data["t_measures"][s1[0]]
                            )
                        )[0][0]

                        p[key][ds, dt] = p_rec[key][ds, s1].mean()
                    try:
                        res = sstats.f_oneway(*tmp)
                        pval[key][i] = res.pvalue
                    except:
                        pass
            return pval, p

        print('add "nights" to parameters')
        print("add variability with $\Delta x$ as plot")
        pval_s, p_s = calc_pval(diff, p_rec, "s", ds_arr, "t")
        print("s-t done")
        pval_t, p_t = calc_pval(diff, p_rec, "t", dt_arr, "s")

        fig = plt.figure(figsize=(7, 4), dpi=pl_dat.sv_opt["dpi"])
        ax1 = plt.axes([0.12, 0.11, 0.35, 0.24])
        ax1.plot([0, ds_arr[-1]], [0.01, 0.01], "k--")
        ax1.plot(ds_arr, pval_s["act"], "ko", markersize=2, label="activation")
        ax1.plot(ds_arr, pval_s["PC"], "bo", markersize=2, label="coding")
        ax1.plot(ds_arr, pval_s["PF"], "ro", markersize=2, label="field stability")
        ax1.set_yscale("log")
        ax1.set_ylim([0.1 * 10 ** (-5), 1])
        ax1.set_xlim([0, 10.5])
        ax1.tick_params(
            axis="y",
            which="both",
            left=True,
            right=True,
            labelright=False,
            labelleft=True,
        )
        ax1.set_xlabel("session difference $\Delta s$")
        ax1.set_ylabel("p-value")
        ax1.legend(loc="lower right", fontsize=8, bbox_to_anchor=[0.9, 0])

        ax2 = plt.axes([0.525, 0.11, 0.35, 0.24])
        ax2.plot([0, dt_arr[-1]], [0.01, 0.01], "k--")
        ax2.plot(dt_arr, pval_t["act"], "ko", markersize=2)
        ax2.plot(dt_arr, pval_t["PC"], "bo", markersize=2)
        ax2.plot(dt_arr, pval_t["PF"], "ro", markersize=2)
        ax2.set_yscale("log")
        ax2.set_ylim([0.1 * 10 ** (-5), 1])
        ax2.set_xlim([0, 160])
        ax2.tick_params(
            axis="y",
            which="both",
            left=True,
            right=True,
            labelright=False,
            labelleft=False,
        )
        ax2.set_xlabel("time difference $\Delta t$ [h]")

        maxSes = 21
        w_bar = 0.05
        offset_bar = ((maxSes + 1) % 2) * w_bar / 2 + (maxSes // 2 - 1) * w_bar

        ax_act = plt.axes([0.12, 0.7, 0.35, 0.25], sharex=ax1)
        ax_PF = plt.axes([0.12, 0.4, 0.35, 0.25], sharex=ax1)
        color_t = iter(plt.cm.rainbow(np.linspace(0, 1, maxSes)))
        for i in range(1, maxSes):
            col = next(color_t)
            ax_act.bar(
                ds_arr - offset_bar + i * w_bar,
                p_s["act"][:, i],
                width=w_bar,
                facecolor=col,
            )
            ax_PF.bar(
                ds_arr - offset_bar + i * w_bar,
                p_s["PF"][:, i],
                width=w_bar,
                facecolor=col,
            )
            # plt.errorbar(ds_arr-offset_bar+i*w_bar,self.stability_dT[dT]['mean'][:maxSes,1],self.stability_dT[dT]['std'][:maxSes,1],fmt='none',ecolor='r')
        # ax.set_xlim([0,15])
        plt.setp(ax_act.get_xticklabels(), visible=False)
        plt.setp(ax_PF.get_xticklabels(), visible=False)
        ax_act.set_yticks(np.linspace(0, 1, 3))
        ax_PF.set_yticks(np.linspace(0, 1, 3))
        ax_act.tick_params(
            axis="y",
            which="both",
            left=True,
            right=True,
            labelright=True,
            labelleft=False,
        )
        ax_PF.tick_params(
            axis="y",
            which="both",
            left=True,
            right=True,
            labelright=True,
            labelleft=False,
        )
        ax_act.set_ylim([0, 1])
        ax_PF.set_ylim([0, 1])
        pl_dat.remove_frame(ax_act, ["top"])
        pl_dat.remove_frame(ax_PF, ["top"])
        ax_act.plot(0, np.NaN, label="activation recurrence $p_{\\alpha}$")
        ax_PF.plot(0, np.NaN, label="field stability $r_{stable}^*$")
        ax_act.legend(
            loc="upper right", handlelength=0, fontsize=10, bbox_to_anchor=[1, 1.1]
        )
        ax_PF.legend(
            loc="upper right", handlelength=0, fontsize=10, bbox_to_anchor=[1, 1.1]
        )

        rainbow = plt.get_cmap("rainbow")
        cNorm = colors.Normalize(vmin=dt_arr[1], vmax=dt_arr[maxSes])
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=rainbow)
        cbaxes = plt.axes([0.09, 0.4, 0.01, 0.55])
        cb = fig.colorbar(scalarMap, cax=cbaxes, orientation="vertical")
        cbaxes.yaxis.tick_left()
        cbaxes.yaxis.set_label_position("left")
        cbaxes.set_ylabel("$\Delta t$")
        # plt.legend(ncol=3)

        ax_act = plt.axes([0.525, 0.7, 0.35, 0.25], sharex=ax2)
        ax_PF = plt.axes([0.525, 0.4, 0.35, 0.25], sharex=ax2)
        maxSes = 11
        w_bar = 0.4
        offset_bar = ((maxSes + 1) % 2) * w_bar / 2 + (maxSes // 2 - 1) * w_bar
        color_s = iter(plt.cm.rainbow(np.linspace(0, 1, maxSes)))
        for i in range(1, maxSes):
            col = next(color_s)
            ax_act.bar(
                dt_arr - offset_bar + i * w_bar,
                p_s["act"][i, :],
                width=w_bar,
                facecolor=col,
            )
            ax_PF.bar(
                dt_arr - offset_bar + i * w_bar,
                p_s["PF"][i, :],
                width=w_bar,
                facecolor=col,
            )
            # plt.errorbar(np.arange(1,maxSes+1)-offset_bar+i*w_bar,self.stability_dT[dT]['mean'][:maxSes,1],self.stability_dT[dT]['std'][:maxSes,1],fmt='none',ecolor='r')
        plt.setp(ax_act.get_xticklabels(), visible=False)
        plt.setp(ax_PF.get_xticklabels(), visible=False)
        # ax.set_xlim([0,200])
        ax_act.set_ylim([0, 1])
        ax_PF.set_ylim([0, 1])
        ax_act.set_yticks(np.linspace(0, 1, 3))
        ax_PF.set_yticks(np.linspace(0, 1, 3))
        ax_act.tick_params(
            axis="y",
            which="both",
            left=True,
            right=True,
            labelright=False,
            labelleft=False,
        )
        ax_PF.tick_params(
            axis="y",
            which="both",
            left=True,
            right=True,
            labelright=False,
            labelleft=False,
        )
        pl_dat.remove_frame(ax_act, ["top"])
        pl_dat.remove_frame(ax_PF, ["top"])

        cNorm = colors.Normalize(vmin=ds_arr[1], vmax=ds_arr[maxSes])
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=rainbow)
        cbaxes = plt.axes([0.9, 0.4, 0.01, 0.55])
        cb = fig.colorbar(scalarMap, cax=cbaxes, orientation="vertical")
        cbaxes.set_ylabel("$\Delta s$")
        # plt.legend(ncol=3)

        # plt.subplot(313)
        # plt.plot(dtrial_arr,pval_dtr,'ro')
        # plt.plot(dtrial_arr,pval_rec_dtr,'ko')
        # plt.plot(dtrial_arr,pval_recPC_dtr,'bo')
        # plt.ylim([0,1])

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig("time_dependence")
        # for

        # s_good[np.where((t_diff==4)]
        # return p_stable, t_diff, s_diff

    def get_field_stability(self, SD=1.96, s_bool=None):

        nbin = self.data["nbin"]

        sig_theta = self.stability["all"]["mean"][0, 2]
        stab_thr = SD * sig_theta

        s_bool = self.status["sessions"] if s_bool is None else s_bool

        field_stability = np.zeros(self.data["nC"]) * np.NaN
        # idx_fields = np.where(self.status["fields"] & self.status['sessions'][np.newaxis,:,np.newaxis])
        idx_fields = np.where(self.status["fields"] & s_bool[np.newaxis, :, np.newaxis])

        for c in np.where(self.status["clusters"])[0]:  # [:10]

            c_fields = idx_fields[0] == c
            fields_ref = self.fields["location"][
                c, idx_fields[1][c_fields], idx_fields[2][c_fields], 0
            ]

            count_hit = 0
            # count_ref = self.status['activity'][c,:,2].sum()
            if self.status["activity"][c, s_bool, 2].sum() > 1:
                for s in np.where(self.status["activity"][c, :, 1] & s_bool)[0]:
                    if self.status["activity"][c, s, 2]:
                        fields_compare = self.fields["location"][
                            c, s, self.status["fields"][c, s, :], 0
                        ]
                        count_ref = len(fields_ref) - len(fields_compare)
                        d = np.abs(
                            np.mod(
                                fields_ref[np.newaxis, :]
                                - fields_compare[:, np.newaxis]
                                + nbin / 2,
                                nbin,
                            )
                            - nbin / 2
                        )
                        # count_hit += (np.sum(d < stab_thr)-len(fields_compare))/(count_ref-1) if count_ref > 1 else np.NaN
                        count_hit += (
                            (np.sum(d < stab_thr) - len(fields_compare)) / count_ref
                            if count_ref > 0
                            else np.NaN
                        )
            # N_norm = self.status['activity'][c,:,1].sum()
            N_norm = s_bool.sum()
            if N_norm > 0:
                field_stability[c] = (
                    count_hit / N_norm
                )  # count_ref# - count_miss / count_ref

        return field_stability

    def get_act_stability_temp(self, status_act=None, ds=3):

        act_stability = np.zeros((self.data["nC"], self.data["nSes"], 2)) * np.NaN
        # ds = ds//2

        if status_act is None:
            status_act = self.status["activity"][..., 1]

        # print(ds)
        for c in np.where(self.status["clusters"])[0]:  # [:10]

            for s in np.where(self.status["sessions"])[0][:-1]:
                s_min = max(0, s - ds)
                s_max = min(self.data["nSes"] - 1, s + ds + 1)

                count_act = status_act[c, s_min:s_max].sum()
                count_act_possible = self.status["sessions"][s_min:s_max].sum()
                count_act_recurr = 0
                count_act_recurr_possible = 0

                for s2 in range(s_min, s_max):
                    if self.status["sessions"][s2]:
                        if self.status["sessions"][s2 + 1]:
                            count_act_recurr_possible += 1
                            if status_act[c, s2]:
                                count_act_recurr += status_act[c, s2 + 1]

                # if self.status['activity'][c,s,1]:
                act_stability[c, s, 0] = count_act / count_act_possible
                act_stability[c, s, 1] = (
                    count_act_recurr / count_act_recurr_possible
                    if count_act_recurr_possible > 0
                    else np.NaN
                )
                # else:
                # act_stability[c,s,:] = 0
                # print('--- neuron %d @ s%d: ---'%(c,s))
                # print(act_stability[c,s,:])
                # print('counts: %d/%d'%(count_act,count_act_possible))
                # print(self.status['activity'][c,s_min:s_max,1])
                # print(self.status['sessions'][s_min:s_max])
        return act_stability

    def get_act_stability(self, s_bool):

        act_stability = np.zeros((self.data["nC"], 3)) * np.NaN

        for c in np.where(self.status["clusters"])[0]:  # [:10]

            count_act = self.status["activity"][c, s_bool, 1].sum()
            count_act_possible = s_bool.sum()
            count_act_recurr = 0
            count_act_recurr_possible = 0

            for s in np.where(s_bool)[0][:-1]:

                if self.status["sessions"][s + 1]:
                    count_act_recurr_possible += 1
                    if self.status["activity"][c, s, 1]:
                        count_act_recurr += self.status["activity"][c, s + 1, 1]

            act_stability[c, 0] = count_act / count_act_possible
            act_stability[c, 1] = (
                count_act_recurr / count_act_recurr_possible
                if count_act_recurr_possible > 0
                else np.NaN
            )
            act_stability[c, 2] = act_stability[c, 1] - (count_act) / count_act_possible

            # print('--- neuron %d : ---'%c)
            # print(act_stability[c,:])
            # print('counts: %d/%d'%(count_act,count_act_possible))
            # print('counts (recurr): %d/%d'%(count_act_recurr,count_act_recurr_possible))
            # print(self.status['activity'][c,s_bool,1])
            # print(self.status['sessions'][s_min:s_max])
        return act_stability

    def get_field_stability_temp(self, SD=1.96, ds=3):

        # nC = self.data['nC']
        # nSes = self.data['nSes']
        nbin = self.data["nbin"]
        # nC,nSes = self.status['activity'].shape[:2]
        sig_theta = self.stability["all"]["mean"][0, 2]
        stab_thr = SD * sig_theta
        # nbin = 100
        # ds = ds//2
        print(ds)
        field_stability = np.zeros((self.data["nC"], self.data["nSes"])) * np.NaN
        # act_stability = np.zeros((nC,nSes))*np.NaN
        idx_fields = np.where(
            self.status["fields"] & self.status["sessions"][np.newaxis, :, np.newaxis]
        )

        for c in np.where(self.status["clusters"])[0]:

            c_fields = idx_fields[0] == c

            for s in np.where(self.status["sessions"])[0][:-1]:

                field_stability[c, s] = 0

                if self.status["activity"][c, s, 2]:
                    s_min = max(0, s - ds)
                    s_max = min(self.data["nSes"] - 1, s + ds + 1)
                    if self.status["activity"][c, s_min:s_max, 2].sum() > 1:
                        s_fields = (idx_fields[1] >= s_min) & (idx_fields[1] < s_max)
                        fields_ref = self.fields["location"][
                            c,
                            idx_fields[1][c_fields & s_fields],
                            idx_fields[2][c_fields & s_fields],
                            0,
                        ]

                        fields_compare = self.fields["location"][
                            c, s, self.status["fields"][c, s, :], 0
                        ]
                        count_ref = len(fields_ref) - len(fields_compare)
                        d = np.abs(
                            np.mod(
                                fields_ref[np.newaxis, :]
                                - fields_compare[:, np.newaxis]
                                + nbin / 2,
                                nbin,
                            )
                            - nbin / 2
                        )

                        field_stability[c, s] += (
                            np.sum(d < stab_thr) - len(fields_compare)
                        ) / count_ref  # if count_ref > 0 else np.NaN
                        # count_hit = 0

                # count_ref = self.status['activity'][c,s_min:s_max,2].sum()
                # act_stability[c,s] = self.status['activity'][c,s_min:s_max,1].sum()/self.status['sessions'][s_min:s_max].sum()

                # if self.status['activity'][c,s_min:s_max,2].sum()>1:
                #     for s2 in range(s_min,s_max):#np.where(self.status['activity'][c,:,1])[0]:
                #         if self.status['activity'][c,s2,2]:
                #             fields_compare = self.fields['location'][c,s2,self.status["fields"][c,s2,:],0]
                #             count_ref = len(fields_ref)-len(fields_compare)
                #             d = np.abs(np.mod(fields_ref[np.newaxis,:]-fields_compare[:,np.newaxis]+nbin/2,nbin)-nbin/2)
                #             # count_hit += (np.sum(d < stab_thr)-len(fields_compare))/(count_ref-1)
                #             count_hit += (np.sum(d < stab_thr)-len(fields_compare))/count_ref if count_ref > 0 else np.NaN

                # N_norm = self.status['activity'][c,s_min:s_max,1].sum()
                # N_norm = self.status['sessions'][s_min:s_max].sum()
                # if N_norm > 0:
                # field_stability[c,s] = count_hit / N_norm#count_ref# - count_miss / count_ref
                # print(field_stability[c,s])

        return field_stability


class plot_dat:

    def __init__(
        self, mouse, pathFigures, nSes, para, sv_suffix="", sv_ext="png", sv_dpi=300
    ):
        self.pathFigures = pathFigures
        if not os.path.exists(self.pathFigures):
            os.mkdir(self.pathFigures)
        self.mouse = mouse

        # L_track =

        # nbin = para['nbin']
        nbin = 40
        L_track = nbin

        self.sv_opt = {"suffix": sv_suffix, "ext": sv_ext, "dpi": sv_dpi}

        self.plt_presi = True
        self.plot_pop = False

        self.plot_arr = ["NRNG", "GT", "RW"]
        self.col = ["b", "g", "r"]
        self.col_fill = [[0.5, 0.5, 1], [0.5, 1, 0.5], [1, 0.5, 0.5]]

        self.h_edges = np.linspace(-0.5, nSes + 0.5, nSes + 2)
        self.n_edges = np.linspace(1, nSes, nSes)
        self.bin_edges = np.linspace(1, L_track, nbin)

        # self.bars = {}
        # self.bars['PC'] = np.zeros(nbin)
        # self.bars['PC'][para['zone_mask']['others']] = 1

        # self.bars['GT'] = np.zeros(nbin);

        # if np.count_nonzero(para['zone_mask']['gate'])>1:
        #   self.bars['GT'][para['zone_mask']['gate']] = 1

        # self.bars['RW'] = np.zeros(nbin);
        # self.bars['RW'][para['zone_mask']['reward']] = 1

        ### build blue-red colormap
        # n = 51;   ## must be an even number
        # cm = ones(3,n);
        # cm(1,1:ceil(n/2)) = linspace(0,1,ceil(n/2));      ## red
        # cm(2,1:ceil(n/2)) = linspace(0,1,ceil(n/2));      ## green
        # cm(2,ceil(n/2):n) = linspace(1,0,floor(n/2)+1);   ## green
        # cm(3,ceil(n/2):n) = linspace(1,0,floor(n/2)+1);   ## blue

    def add_number(self, fig, ax, order=1, offset=None):

        # offset = [-175,50] if offset is None else offset
        offset = [-150, 50] if offset is None else offset
        offset = np.multiply(offset, self.sv_opt["dpi"] / 300)
        pos = fig.transFigure.transform(plt.get(ax, "position"))
        x = pos[0, 0] + offset[0]
        y = pos[1, 1] + offset[1]
        ax.text(
            x=x,
            y=y,
            s="%s)" % chr(96 + order),
            ha="center",
            va="center",
            transform=None,
            weight="bold",
            fontsize=14,
        )

    def remove_frame(self, ax, positions=None):

        if positions is None:
            positions = ["left", "right", "top", "bottom"]

        for p in positions:
            ax.spines[p].set_visible(False)

        # if 'left' in positions:
        # ax.set_yticks([])

        # if 'bottom' in positions:
        # ax.set_xticks([])

    def plot_with_confidence(
        self, ax, x_data, y_data, CI, col="k", ls="-", lw=1, label=None
    ):

        col_fill = np.minimum(np.array(colors.to_rgb(col)) + np.ones(3) * 0.3, 1)
        if len(CI.shape) > 1:
            ax.fill_between(x_data, CI[0, :], CI[1, :], color=col_fill, alpha=0.2)
        else:
            ax.fill_between(x_data, y_data - CI, y_data + CI, color=col_fill, alpha=0.2)
        ax.plot(x_data, y_data, color=col, linestyle=ls, linewidth=lw, label=label)

    def save_fig(self, fig_name, fig_pos=None):
        path = os.path.join(
            self.pathFigures,
            "m%s_%s%s.%s"
            % (self.mouse, fig_name, self.sv_opt["suffix"], self.sv_opt["ext"]),
        )
        plt.savefig(path, format=self.sv_opt["ext"], dpi=self.sv_opt["dpi"])
        print("Figure saved as %s" % path)


def bootstrap_shifts(fun, shifts, N_bs, nbin):

    N_data = len(shifts)
    if N_data == 0:
        return (
            np.zeros(4) * np.NaN,
            np.zeros((2, 4)) * np.NaN,
            np.zeros(4) * np.NaN,
            np.zeros((2, nbin)) * np.NaN,
        )

    samples = np.random.randint(0, N_data, (N_bs, N_data))
    # sample_randval = np.random.rand(N_bs,N_data)
    shift_distr_bs = np.zeros((N_bs, nbin))
    par = np.zeros((N_bs, 4)) * np.NaN
    for i in range(N_bs):
        shift_distr_bs[i, :] = shifts[samples[i, :], :].sum(0)
        shift_distr_bs[i, :] /= shift_distr_bs[i, :].sum()
        par[i, :], p_cov = fun(shift_distr_bs[i, :])
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
        return np.zeros(4) * np.NaN, np.NaN


# def get_shift_distr(ds,compare,para):

#     nSes,nbin,N_bs,idx_celltype = para
#     L_track=100
#     p = {'all':{},
#         'cont':{},
#         'mix':{},
#         'discont':{},
#         'silent_mix':{},
#         'silent':{}}

#     s1_shifts,s2_shifts,f1,f2 = np.unravel_index(compare['pointer'].col,(nSes,nSes,5,5))
#     #print(idx_celltype)
#     Ds = s2_shifts-s1_shifts
#     idx_ds = np.where((Ds==ds) & idx_celltype)[0]
#     N_data = len(idx_ds)

#     idx_shifts = compare['pointer'].data[idx_ds].astype('int')-1
#     shifts = compare['shifts'][idx_shifts]
#     shifts_distr = compare['shifts_distr'][idx_shifts,:].toarray()

#     for pop in p.keys():
#         if pop == 'all':
#             idxes = np.ones(N_data,'bool')
#         elif pop=='cont':
#             idxes = compare['inter_coding'][idx_ds,1]==1
#         elif pop=='mix':
#             idxes = ((compare['inter_coding'][idx_ds,1]>0) & (compare['inter_coding'][idx_ds,1]<1)) & (compare['inter_active'][idx_ds,1]==1)
#         elif pop=='discont':
#             idxes = (compare['inter_coding'][idx_ds,1]==0) & (compare['inter_active'][idx_ds,1]==1)
#         elif pop=='silent_mix':
#             idxes =(compare['inter_active'][idx_ds,1]>0) & (compare['inter_active'][idx_ds,1]<1)
#         elif pop=='silent':
#             idxes = compare['inter_active'][idx_ds,1]==0

#         # p[pop]['mean'], p[pop]['std'], _ = bootstrap_shifts(fit_shift_model,cdf_shifts_ds[idxes,:],N_bs,nbin)
#         p[pop]['mean'], p[pop]['CI'], p[pop]['std'], _ = bootstrap_shifts(fit_shift_model,shifts_distr[idxes,:],N_bs,nbin)
#     return p
