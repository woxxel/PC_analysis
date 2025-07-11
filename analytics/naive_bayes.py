import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tqdm import *

from utils import get_firingrate, gauss_smooth
from utils_analysis import define_active


class naive_Bayes:

    def __init__(self,cluster):

        self.para = {'f':        15,                 ## frequency of measurement
                'runthr':   10,            ## minimum distance travelled, to be considered "moving"
                'sm':       np.ones(10+1),      ## size of dilation filter to smooth mouse-speed
                'lr_min':   30,            ## minimum number of frames spent running, to be counted as "longrun"
                'pad':      0,                ## number of border bins to be removed from data (not working properly, 0 = off)
                'nbins':    100,
                'T':        8989,       ## number of time steps
                'nbin':     100,
                'd':        1}      ## dimensionality of data
        self.cluster = cluster

    def training(self,s,T_cut=300,sig_smooth=5):

        ## ### INPUT VARIABLES ###
        ## trainName :    (string) path to csvfile to read data from for training bayes-classifier
        ##                  columns: 1) time; 2)&3) x-/y-position 4) binposition 5) - end) neuron activity data
        ## testName :     (string) path to csvfile to read data from for testing bayes classifier
        ## cell_array:    (bool) Nx1-vector, containing true for all neurons to be used for decoding and false, else. If empty, all are used
        ## dT       :     (integer) number of previous timesteps to include in estimation
        ## sig_smooth:    (double) width of smoothing kernel for probability distributions (0 = no smoothing)
        ## plt      :     (boolean) choose, whether to plot while calculating (true) or not (false)
        ##                  plot provides insight into probabilities used by naive-bayes-estimator
        ## plt_vid  :     (boolean) choose, whether dynamic plotting is saved as video

        ### build bayesian probabilities to prepare reverse inference:
        ##    bayes rule:   p(s|x(t)) = p(x|s) * p(s) / p(x)
        ##          here:             = prod_dt(p(x(t-dt)|s)/p(x(t-dt))) * p(s)
        ##      with dt in [0,dT] ()
        ##      and p(x(t)|s) = p(x_1(t)|s)*p(x_2(t)|s)*...*p(x_n(t)|s) (naive bayes assumption: independence of neural activity)
        ##
        ##    x_n  = activity of cell n (1=active, 0=silent)
        ##    s    = position of the mouse
        ##    p_x  = p(x_n)   - probability of cell n being active
        ##    p_s  = p(s)     - probability of the mouse being at position s
        ##    p_xs = p(x_n|s) - (prior) probability of cell n being active, GIVEN THAT the mouse is at position s
        ##    p_sx = p(s|x_n) - (posterior) probability of the mouse being at position s, GIVEN THAT cell n is active

        ### define parameters
        T = self.para['T']
        nbin = self.para['nbin']
        nC = self.cluster.meta['nC']

        pathSession = os.path.join(self.cluster.para['pathMouse'],'Session%02d'%(s+1))
        data = define_active(pathSession)

        pathLoad = os.path.join(pathSession,'results_redetect.mat')
        ld = loadmat(pathLoad,variable_names=['S'],squeeze_me=True)

        n_arr = self.cluster.IDs['neuronID'][self.cluster.status[:,s,1],s,1].astype('int')
        c_arr = np.where(self.cluster.status[:,s,1])[0]
        data['S'] = np.zeros((nC,T),'int')
        qtl_steps = 2
        for c,n in zip(c_arr,n_arr):
            # print(n)
            _,fr_thr,_ = get_firingrate(ld['S'][n,:],sd_r=1)
            data['S'][c,:] = self.get_quantiled_fr(ld['S'][n,:],fr_thr,qtl_steps=qtl_steps)
            # data['S'][c,:] = ld['S'][n,:]>fr_thr

        # nC = self.cluster.stats['cluster_bool'].sum()

    #    T_cut = 600;

        T_cut = T_cut * self.para['f']
        T_training = np.zeros(T,'bool')
        T_training[:T_cut] = True
        # T_training = np.cumsum(data['active'])<T_cut

        print('training until:')
        # print(T_training)
        print(T_cut)
        coarse = 1
        # print(data['time'][np.where(T_training)[0][1]])
        nbin_coarse = int(nbin/coarse)
        loc = data['position'].astype('int')
        loc_coarse =  (data['position']/coarse).astype('int')
        self.p = {'xs':     np.zeros((nC,nbin_coarse,qtl_steps)),
                  's':      np.ones(nbin_coarse)/nbin_coarse}
        # self.p['s'] = np.histogram(loc_coarse[data['active']],np.linspace(0,nbin_coarse,nbin_coarse+1))[0]
        # self.p['s'] = self.p['s']/self.p['s'].sum()
        ## build prior probability
        S = data['S'][c_arr,:]
        # for x in range(self.para['pad'],nbin-self.para['pad']):
        for x in range(nbin_coarse):
            t_idx = ((loc_coarse==x) & data['active'] & T_training)
            nt = t_idx.sum()
            if nt:
                # print(S[:,t_idx])
                for i in range(qtl_steps):
                    self.p['xs'][c_arr,x,i] = (S[:,t_idx]==i).sum(1)/nt
            else:
                self.p['xs'][:,x] = 0

        if sig_smooth:
            self.p['s'] = gauss_smooth(self.p['s'],sig_smooth)
            self.p['xs'] = gauss_smooth(self.p['xs'],(0,sig_smooth,0))
        self.p['xs'][self.p['xs']==0] = 0.001

        print('------ training done ---------')


    def get_quantiled_fr(self,S,S_thr,smooth=5,qtl_steps=5):

        S_smooth = gauss_smooth(np.floor(S / S_thr).astype('float')*self.para['f'],smooth)#(S>S_thr).astype('float')#
        # S_qtl = S_qtl[self.dataBH['active']]
        qtls = np.quantile(S_smooth[S_smooth>0],np.linspace(0,1,qtl_steps))
        # S_qtl = np.count_nonzero(S_smooth[:,np.newaxis]>=qtls[np.newaxis,1:-1],1)
        S_qtl = np.count_nonzero(S_smooth[:,np.newaxis]>=qtls[np.newaxis,:-1],1)
        return S_qtl


    def run_estimation(self,s,cells=None,dT=0,sig_smooth=5):

        #    change to different input/test-files
        #    get smoothing option in here
        #    get evaluation of results: integral over absolute estimation error?

        ### function calculating naive bayes estimation of mouse location from behavior and (binary) activity data provided in a spreadsheet
        qtl_steps = 2

        nbin = self.para['nbin']
        T = self.para['T']
        nC = self.cluster.meta['nC']

        n_arr = self.cluster.IDs['neuronID'][self.cluster.status[:,s,1],s,1].astype('int')
        c_arr = np.where(self.cluster.status[:,s,1])[0]

        print(n_arr.shape)
        print(c_arr.shape)
        pathSession = os.path.join(self.cluster.para['pathMouse'],'Session%02d'%(s+1))
        data = define_active(pathSession)

        pathLoad = os.path.join(pathSession,'results_redetect.mat')
        ld = loadmat(pathLoad,variable_names=['S'],squeeze_me=True)
        data['S'] = np.zeros((nC,T),'int')
        for c,n in zip(c_arr,n_arr):
            # print(n)
            _,fr_thr,_ = get_firingrate(ld['S'][n,:],sd_r=1)
            data['S'][c,:] = self.get_quantiled_fr(ld['S'][n,:],fr_thr,qtl_steps=qtl_steps)
            # data['S'][c,:] = ld['S'][n,:]>fr_thr

        d = 1       ## dimensionality of environment (> 1D not yet supported)

        coarse = 1
        nbin_coarse = int(nbin/coarse)
        ## calculate estimate and display
        loc_est = np.zeros(T)*np.NaN;
        x_err = np.zeros(T)*np.NaN;

        p_sx = np.zeros((nbin_coarse,T))

        p_infer = np.ones((nbin_coarse,T))

        p_x = np.zeros(T)
        p_xs = np.zeros((T,nbin_coarse))

        print('get activity in quantiles')

        if cells is None:
            # cells = self.cluster.status[:,s,1]      ## choose place cells, only
            cells = self.cluster.status[:,50,2]      ## choose place cells, only
        # print(self.p['s'].shape)
        # S = data['S'][cells,:]
        c_arr = np.where(cells)[0]

        for t in tqdm(range(T)):
            # act_now = data['S'][:,t]
            # print(data['S'][c_arr,t])
            p_xs_now = self.p['xs'][c_arr,:,data['S'][c_arr,t]]

            # print(p_xs_now[:,5])
            # print(p_xs_now[:,5])
            # print(p_xs_now.shape)
            # p_xs_now = np.copy(self.p['xs'])
            # p_xs_now[~act_now,:] = 1-p_xs_now[~act_now,:]
            # print(p_xs_now)
            # p_xs[t,:] = np.prod(np.float128(p_xs_now),0)
            p_xs[t,:] = np.sum(np.log(p_xs_now),0)
            ###somethings not right - check formulas
            # print(p_xs[t,:])
            # p_xs_loc = p_xs[t,:]+np.log(self.p['s'])
            # p_xs_loc = p_xs[t,:]*self.p['s']
            # print(p_xs_loc)
            # p_x[t] = np.sum(p_xs_loc)       ## probability of obtaining this activity in general
            # print(p_x[t])
            # print(p_xs[t,:])
            # self.get_p_xs(cells,data['S'][n_arr,t]) ## probability of obtaining observed activity for each position
            # p_x = np.dot(p_xs_now,self.p['s'])       ## probability of obtaining this activity in general
            # print(p_x)
            # p_sx[:,t] = (p_xs_now/p_x1).T
            # p_infer[:,t] = (p_xs_loc/p_x[t]).T

            # print(p_sx[...,t])
            # p_infer[:,t] = np.prod(p_sx[:,t],1)*self.p['s'].T         ## bayesian estimate of position
            # p_infer[:,t] = np.prod(p_sx[:,cells,t],1)*self.p['s'].T         ## bayesian estimate of position
            # print(p_infer[:,t])
            # print(p_xs[t-dT:t,:]*self.p['s']/p_x[t-dT:t,np.newaxis])
            # p_infer[:,t] = np.prod(p_xs[t-dT:t,:]*self.p['s']/p_x[t-dT:t,np.newaxis],0)
            p_infer[:,t] = np.sum(p_xs[t-dT:t,:]+np.log(self.p['s'])[np.newaxis,:],0)
            # print(p_infer[:,t])
            # for dt in range(min(t-1,dT)):
#
                # p_infer[:,t] = p_infer[:,t]*p_infer[:,t-dt]#np.prod(p_sx[:,cells,t-dt],1);

            max_pos = np.argmax(p_infer[:,t])
        #      if log(max_amp) > 0
            loc_est[t] = max_pos

        plt.figure()
        plt.subplot(211)
        plt.plot(data['time'],data['position']/coarse,'k.',markersize=1)
        plt.plot(data['time'],loc_est,'r.',markersize=1)
        plt.xlim([0,600])
        plt.subplot(212)
        plt.plot(data['time'],data['position']/coarse-loc_est,'k.',markersize=1)
        plt.xlim([0,600])
        plt.show(block=False)

        # data = read_data(pathSession);

    #    if ~(ses(1)==ses(2))
        # pathMatch = dir(pathcat(pathMouse,'matching/results*_std=0_thr=70_w=33_OnACID.mat'));
        # pathMatch = pathcat(pathMouse,'matching',pathMatch.name);
        # data_ld = load(pathMatch);
        # assign = data_ld.assignments+1;
        # c_idx = find(~isnan(assign(:,ses(1))));
        #
        # n_idx(:,1) = assign(c_idx,ses(1));
        # n_idx(:,2) = assign(c_idx,ses(2));
        #
        # n_idx = sortrows(n_idx);
    #    end



        # data.N
    ## uncomment to remove silent neurons from data - does not effect the estimator
    #    p_x_active = sum(data.act,2)/T;  ## calculating firing rate of each neuron
    #    rm_idx = p_x_active < 10/T; ## and removing silent neurons from data (
    #    p_x_active(rm_idx) = [];
    #    data.act(rm_idx,:) = [];

    #    p_s(1,1+para.pad:para.nbins-para.pad) = histcounts(data.loc(data.run_bool),linspace(para.pad,para.nbins-para.pad,para.nbins-2*para.pad+1)+0.5,'Normalization','probability');


        #    if ses(1)==ses(2)
        #      cell_array = data.p_vals < 0.1;
        #    else
        # cell_array = find(data.p_vals < 0.1 & ~isnan(n_idx(:,2)));
        # cell_assign = n_idx(cell_array,2);



          ## get matches of neurons
        #    cell_array = n_idx(:,2);
        #    if isempty(cell_array)
        #      cell_array = sum(data.act,2)/T >= 10/T;#true(N,1);
        #    end
        #    [sum(cell_array),data.N]



        # if plt:
        #
        #     bar_loc = zeros(para.nbins,1);
        #     bar_sm = ones(5+1,1);     ## width of sliding location bar
        #
        #     figure('Position',[0.1 0.1 1800 1500])
        #
        #     ## left column
        #
        #     # dwelltime distribution
        #     subplot(4,2,1)
        #     bar(p_s)
        #     ylabel('p(s)')
        #
        #     # inferred position of whole network
        #     subplot(4,2,3)
        #     hold on
        #     h_bar = bar(log(p_infer(:,1)));
        #     h_bin = plot(0,0,'xr','MarkerSize',10);
        #     ylim([-40,40])
        #     ylabel('log(p(s|r))')
        #     xlabel('bin #')
        #
        #     # inferred position per neuron
        #     subplot(2,2,3)
        #     hold on
        #     h_im = imagesc(p_sx(:,:,1)'.*p_s);
        #     h_bar_loc = bar(bar_loc,1,'FaceAlpha',0.5,'FaceColor','g','EdgeColor','None');
        #     ylabel('neuron')
        #     xlabel('location')
        #     ylim([0,sum(cell_array)])
        #     set(gca,'clim',[0,5])
        #     colormap('jet')
        #     colorbar
        #
        #     ## right column
        #
        #     # position of mouse in real space
        #     subplot(2,2,2)
        #     h_loc = scatter(0,0,'ko','filled');
        # #      h_loc_err = scatter(0,0,'k.');
        #     xlim([min(data.loc)-20,max(data.loc)+20])
        # #      ylim([min(data.speed(:,2))-20,max(data.speed(:,2))+20])
        #     ylim([-1,1])
        #     h_title = suptitle('t=0');
        #     xlabel('x position')
        #     ylabel('y position')
        #
        #     # position of mouse in bin-space vs estimated position
        #     subplot(4,2,6)
        #     hold on
        #     bar(data.time,(~data.run_bool)*para.nbins,1,'FaceColor',[0.9,0.9,0.9])
        #     scatter(data.time,data.loc,'.k')
        #     h_infer = scatter(data.time,loc_est,'.r');
        #
        #     xlim([0,data.time(end)])
        #     ylim([0,para.nbins])
        #     ylabel('bin s')
        #
        #     # error of estimation
        #     subplot(4,2,8)
        #     hold on
        #     bar(data.time,(~data.run_bool)*para.nbins,1,'FaceColor',[0.9,0.9,0.9])
        #     bar(data.time,-(~data.run_bool)*para.nbins,1,'FaceColor',[0.9,0.9,0.9])
        #     plot([0,data.time(end)],[0,0],'k--')
        #     h_err = scatter(data.time,x_err,'.r');
        #     xlim([0,data.time(end)])
        #     ylim([-para.nbins/2,para.nbins/2])
        #     ylabel('\Delta s')
        #     xlabel('t [s]')
        #
        #     if plt_vid        ## preparing video recording
        #       video = struct;
        #       video.path = pathcat(pwd, 'BayesTest.avi');
        #
        #       video.obj = VideoWriter(video.path);
        #       video.obj.FrameRate = 15;
        #       video.obj.Quality = 70;
        #
        #       open(video.obj);
        #     end
        #   end




        #      else
        #        loc_est(t) = NaN;
        #      end


        #     if plt
        #       set(h_bar,'YData',log(p_infer(:,t)))
        #       set(h_bin,'XData',data.loc(t))
        #
        #       t_min = max(1,t-100);
        #       x_data = data.loc(t_min:t);
        # #        y_data = data(t_min:t,3);
        #       nt = t-t_min+1;
        #       y_data = zeros(nt,1);
        #       set(h_loc,'XData',x_data,'YData',y_data,'SizeData',linspace(5,100,nt))
        #
        #
        #       set(h_im,'CData',p_sx(:,cell_array,t)')
        #       bar_loc = zeros(para.nbins,1);
        #       bar_loc(data.loc(t)) = 1;
        #       bar_loc = imdilate(bar_loc,bar_sm);
        #       set(h_bar_loc,'YData',bar_loc*data.N)
        #
        #       set(h_infer,'YData',loc_est)
        #
        #       x_err(t) = mod(data.loc(t)-loc_est(t)+para.nbins/2,para.nbins)-para.nbins/2;
        #       set(h_err,'YData',x_err)
        #
        #       set(h_title,'String',sprintf('t=%d',t))
        #
        #       if plt_vid
        #         drawnow
        #         frame = getframe(gcf);
        #         writeVideo(video.obj,frame);
        #       else
        #         pause(1/para.f)
        #       end
        #     end
        # if plt && plt_vid
        # close(h.video.obj)

        # figure
        # subplot(2,1,1)
        # hold on
        #
        # bar(data.time,(~data.run_bool)*para.nbins,1,'FaceColor',[0.8,0.8,0.8])
        # scatter(data.time,data.loc,'.k')
        # size(loc_est)
        # scatter(data.time,loc_est,'.r')
        # ylabel('bin position s')
        # xlim([0,data.time(end)])
        # ylim([0,para.nbins])
        #
        # #### check: coding errors mostly during standing still?
        # subplot(2,1,2)
        # hold on
        # bar(data.time,(~data.run_bool)*para.nbins/2,1,'FaceColor',[0.8,0.8,0.8])
        # bar(data.time,-(~data.run_bool)*para.nbins/2,1,'FaceColor',[0.8,0.8,0.8])
        # x_err = mod(data.loc'-loc_est+para.nbins/2,para.nbins)-para.nbins/2;
        # scatter(data.time,x_err,'.r')
        # xlim([0,data.time(end)])
        # ylim([-para.nbins/2,para.nbins/2])
        # xlabel('t [s]')
        # ylabel('estimation error \Delta s')
        #
        # out = struct;
        # out.N_err_lr = sum(x_err(data.run_bool) > 5);
        # out.N_err_nlr = sum(x_err(~data.run_bool) > 5);
        # disp(sprintf('Errors during running: %d',out.N_err_lr))
        # out.N_err_lr/sum(data.run_bool)
        # disp(sprintf('Errors during resting: %d',out.N_err_nlr))
        # out.N_err_lr/sum(~data.run_bool)
        # out.error_rate = mean(abs(x_err));
        # disp(sprintf('error per timebin: %5.3g',out.error_rate))
        # print(pathcat(pwd,'bayes_estimation.png'),'-dpng','-r300')


    def get_p_xs(self,cells,act):
        # calculates probability of obtaining observed activity
        # N = cells.sum()
        for n,c in enumerate(np.where(cells)[0]):
            if ~act[n]:
                self.p['xs'][c,:] = 1-self.p['xs'][c,:]
        # return p_xs




# function data = read_data(fileName)
#   ### read data in from file "fileName"
#   data = struct;
# #    [folder,~,ext] = fileparts(fileName);
#   folder = fileName;
# #    ext
# #    switch ext
# #      case '.csv'
# #
# #        data_ld = csvread(trainName);
# #        data.time = data_ld(:,1)*1/para.f;    ## time = Tx1 array containing timestamps
# #        data.loc = data_ld(:,4);              ## loc = Txd array containing location data
# #        data.act = data_ld(:,5:end)';         ## act = NxT array containing activity data
# #        data.speed = data_ld(:,2:3);
# #      case {'.mat',''}
#       pathAct = dir(pathcat(folder,'*OnACID.mat'));
#       pathAct = pathcat(folder,pathAct.name);
#
# #        Ca could be used by scaling prob from 0 to 1(=threshold from baseline firing rate)
#       data_ld = load(pathAct,'S');
#
#       data.N = size(data_ld.S,1);       ## number of neurons
#       ## test if data is ordered properly
#       if data.N > 4000
#         data_ld.S = transpose(data_ld.S);
#         data.N = size(data_ld.S,1);
#       end
#
#       for n = 1:data.N
#         [data.act(n,:),~,~] = get_spikes(data_ld.S(n,:));
# #          [~,md,sd_r] = get_spikes(data_ld.S(n,:));
# #          sp_thr = md + 2*sd_r;
# #          data.act(n,:) = min(1,data_ld.S(n,:)/sp_thr);
#       end
#       data.act = data.act > 0;
#
#       pathBH = dir(pathcat(folder,'*aligned.mat'));
#       pathBH = pathcat(folder,pathBH.name);
#
#       data_ld = load(pathBH);
#       data.time = data_ld.alignedData.resampled.time;
#       data.loc = data_ld.alignedData.resampled.binpos;
#       data.speed = data_ld.alignedData.resampled.speed;
#       data.run_bool = data_ld.alignedData.resampled.longrunperiod;
#
#       pathPC = dir(pathcat(folder,'*PC_fields.mat'));
#       pathPC = pathcat(folder,pathPC.name);
#
#       data_ld = load(pathPC);
#       PC_fields = data_ld.PC_fields;
#       data.p_vals = zeros(data.N,1);
#       for n=1:length(PC_fields)
#         data.p_vals(n) = PC_fields(n).MI.p_value;
#       end
#
# #    end
# end