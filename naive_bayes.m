
function [out] = naive_bayes(pathMouse,ses,cell_array,dT,sig_smooth,T_cut,plt,plt_vid)
  
%    change to different input/test-files
%    get smoothing option in here
%    get evaluation of results: integral over absolute estimation error?
  
  %{
  %%% function calculating naive bayes estimation of mouse location from behavior and (binary) activity data provided in a spreadsheet
  
  %% ### INPUT VARIABLES ###
  %% trainName :    (string) path to csvfile to read data from for training bayes-classifier
  %%                  columns: 1) time; 2)&3) x-/y-position 4) binposition 5) - end) neuron activity data
  %% testName :     (string) path to csvfile to read data from for testing bayes classifier 
  %% cell_array:    (bool) Nx1-vector, containing true for all neurons to be used for decoding and false, else. If empty, all are used
  %% dT       :     (integer) number of previous timesteps to include in estimation
  %% sig_smooth:    (double) width of smoothing kernel for probability distributions (0 = no smoothing)
  %% plt      :     (boolean) choose, whether to plot while calculating (true) or not (false)
  %%                  plot provides insight into probabilities used by naive-bayes-estimator
  %% plt_vid  :     (boolean) choose, whether dynamic plotting is saved as video
  
  %%% build bayesian probabilities to prepare reverse inference:
  %%    bayes rule:   p(s|x(t)) = p(x|s) * p(s) / p(x)
  %%          here:             = prod_dt(p(x(t-dt)|s)/p(x(t-dt))) * p(s)
  %%      with dt in [0,dT] () 
  %%      and p(x(t)|s) = p(x_1(t)|s)*p(x_2(t)|s)*...*p(x_n(t)|s) (naive bayes assumption: independence of neural activity)
  %%    
  %%    x_n  = activity of cell n (1=active, 0=silent)
  %%    s    = position of the mouse
  %%    p_x  = p(x_n)   - probability of cell n being active
  %%    p_s  = p(s)     - probability of the mouse being at position s
  %%    p_xs = p(x_n|s) - (prior) probability of cell n being active, GIVEN THAT the mouse is at position s
  %%    p_sx = p(s|x_n) - (posterior) probability of the mouse being at position s, GIVEN THAT cell n is active
  %}
  
%%% define parameters
  para = struct('f',15,...                 %% frequency of measurement
                'runthr',10,...            %% minimum distance travelled, to be considered "moving"
                'sm',ones(1,10+1),...      %% size of dilation filter to smooth mouse-speed
                'lr_min',30,...            %% minimum number of frames spent running, to be counted as "longrun"
                'pad',0,...                %% number of border bins to be removed from data (not working properly, 0 = off)
                'nbins',80);
  
  pathSession = pathcat(pathMouse,sprintf('Session%02d',ses(1)));
  data = read_data(pathSession);
  
%    if ~(ses(1)==ses(2))
    pathMatch = dir(pathcat(pathMouse,'matching/results*_std=0_thr=70_w=33_OnACID.mat'));
    pathMatch = pathcat(pathMouse,'matching',pathMatch.name);
    data_ld = load(pathMatch);
    assign = data_ld.assignments+1;
    c_idx = find(~isnan(assign(:,ses(1))));
    
    n_idx(:,1) = assign(c_idx,ses(1));
    n_idx(:,2) = assign(c_idx,ses(2));
    
    n_idx = sortrows(n_idx);
%    end
  
  T = size(data.loc,2)       %% number of time steps
  d = size(data.loc,1);       %% dimensionality of environment (> 1D not yet supported)
  
  data.N
%% uncomment to remove silent neurons from data - does not effect the estimator
%    p_x_active = sum(data.act,2)/T;  %% calculating firing rate of each neuron
%    rm_idx = p_x_active < 10/T; %% and removing silent neurons from data (
%    p_x_active(rm_idx) = [];
%    data.act(rm_idx,:) = [];
  
  p_s = zeros(1,para.nbins);
%    p_s(1,1+para.pad:para.nbins-para.pad) = histcounts(data.loc(data.run_bool),linspace(para.pad,para.nbins-para.pad,para.nbins-2*para.pad+1)+0.5,'Normalization','probability');
  p_s = ones(1,para.nbins)/para.nbins;
  
%    T_cut = 600;
  T_cut = T_cut * para.f
  T_training = cumsum(data.run_bool)<T_cut;
  
  disp('training until:')
  data.time(find(T_training,1,'last'))
  
  %% build prior probability
  p_xs = zeros(data.N,para.nbins);
  for s = 1+para.pad:para.nbins-para.pad
    t_idx = (data.loc == s & data.run_bool & T_training);
    nt = sum(t_idx);
    if nt
      for n = 1:data.N
        p_xs(n,s) = sum(data.act(n,t_idx)>0)/nt;
      end
    else
      p_xs(:,s) = 0;
    end
  end
  
  if sig_smooth
    p_s = imgaussfilt(p_s,sig_smooth,'Padding','circular');
    for n = 1:data.N           %% should also work without looping - but how?
      p_xs(n,:) = imgaussfilt(p_xs(n,:),sig_smooth,'Padding','circular');
    end
  end
  
  disp('------ training done ---------')
  
%    if ses(1)==ses(2)
%      cell_array = data.p_vals < 0.1;
%    else
  cell_array = find(data.p_vals < 0.1 & ~isnan(n_idx(:,2)));
  cell_assign = n_idx(cell_array,2);
  
  pathSession = pathcat(pathMouse,sprintf('Session%02d',ses(2)));
  data = read_data(pathSession);
  
  %% get matches of neurons
  
%    cell_array = n_idx(:,2);
%    if isempty(cell_array)
%      cell_array = sum(data.act,2)/T >= 10/T;%true(N,1);
%    end
%    [sum(cell_array),data.N]
  
  %% calculating speed to obtain longrun-periods
  
  T = size(data.loc,2)                            %% number of time steps
  d = size(data.loc,1);                           %% dimensionality of environment (> 1D not yet supported)
  para.nbins = max(para.nbins,max(data.loc));     %% number of bins
  
  %% calculate estimate and display
  loc_est = zeros(T,1)*NaN;
  x_err = zeros(T,1)*NaN;
  
  p_sx = zeros(para.nbins,data.N,T);
  p_infer = ones(para.nbins,T);
  
  if plt
    
    bar_loc = zeros(para.nbins,1);
    bar_sm = ones(5+1,1);     %% width of sliding location bar
    
    figure('Position',[0.1 0.1 1800 1500])
    
    %% left column
    
    % dwelltime distribution
    subplot(4,2,1)
    bar(p_s)
    ylabel('p(s)')
    
    % inferred position of whole network
    subplot(4,2,3)
    hold on
    h_bar = bar(log(p_infer(:,1)));
    h_bin = plot(0,0,'xr','MarkerSize',10);
    ylim([-40,40])
    ylabel('log(p(s|r))')
    xlabel('bin #')
    
    % inferred position per neuron
    subplot(2,2,3)
    hold on
    h_im = imagesc(p_sx(:,:,1)'.*p_s);
    h_bar_loc = bar(bar_loc,1,'FaceAlpha',0.5,'FaceColor','g','EdgeColor','None');
    ylabel('neuron')
    xlabel('location')
    ylim([0,sum(cell_array)])
    set(gca,'clim',[0,5])
    colormap('jet')
    colorbar
    
    %% right column
    
    % position of mouse in real space
    subplot(2,2,2)
    h_loc = scatter(0,0,'ko','filled');
%      h_loc_err = scatter(0,0,'k.');
    xlim([min(data.loc)-20,max(data.loc)+20])
%      ylim([min(data.speed(:,2))-20,max(data.speed(:,2))+20])
    ylim([-1,1])
    h_title = suptitle('t=0');
    xlabel('x position')
    ylabel('y position')
    
    % position of mouse in bin-space vs estimated position
    subplot(4,2,6)
    hold on
    bar(data.time,(~data.run_bool)*para.nbins,1,'FaceColor',[0.9,0.9,0.9])
    scatter(data.time,data.loc,'.k')
    h_infer = scatter(data.time,loc_est,'.r');
    
    xlim([0,data.time(end)])
    ylim([0,para.nbins])
    ylabel('bin s')
    
    % error of estimation
    subplot(4,2,8)
    hold on
    bar(data.time,(~data.run_bool)*para.nbins,1,'FaceColor',[0.9,0.9,0.9])
    bar(data.time,-(~data.run_bool)*para.nbins,1,'FaceColor',[0.9,0.9,0.9])
    plot([0,data.time(end)],[0,0],'k--')
    h_err = scatter(data.time,x_err,'.r');
    xlim([0,data.time(end)])
    ylim([-para.nbins/2,para.nbins/2])
    ylabel('\Delta s')
    xlabel('t [s]')
    
    if plt_vid        %% preparing video recording
      video = struct;
      video.path = pathcat(pwd, 'BayesTest.avi');
      
      video.obj = VideoWriter(video.path);
      video.obj.FrameRate = 15;
      video.obj.Quality = 70;
      
      open(video.obj);
    end
  end
  
  
  for t = 1:T
    
    p_xs_now = get_p_xs(p_xs(cell_array,:),data.act(cell_assign,t));   %% probability of obtaining observed activity for each position
    p_x = p_xs_now * p_s';                %% probability of obtaining this activity in general
    p_sx(:,cell_array,t) = (p_xs_now./p_x)';       
    
    p_infer(:,t) = prod(p_sx(:,cell_array,t),2).*p_s';   %% bayesian estimate of position
    for dt = 1:min(t-1,dT)
      p_infer(:,t) = p_infer(:,t).*prod(p_sx(:,cell_array,t-dt),2);
    end
    
    [max_amp,max_pos] = max(p_infer(:,t));
%      if log(max_amp) > 0
    loc_est(t) = max_pos;
%      else
%        loc_est(t) = NaN;
%      end
    
    
    if plt
      set(h_bar,'YData',log(p_infer(:,t)))
      set(h_bin,'XData',data.loc(t))
      
      t_min = max(1,t-100);
      x_data = data.loc(t_min:t);
%        y_data = data(t_min:t,3);
      nt = t-t_min+1;
      y_data = zeros(nt,1);
      set(h_loc,'XData',x_data,'YData',y_data,'SizeData',linspace(5,100,nt))
      
      
      set(h_im,'CData',p_sx(:,cell_array,t)')
      bar_loc = zeros(para.nbins,1);
      bar_loc(data.loc(t)) = 1;
      bar_loc = imdilate(bar_loc,bar_sm);
      set(h_bar_loc,'YData',bar_loc*data.N)
      
      set(h_infer,'YData',loc_est)
      
      x_err(t) = mod(data.loc(t)-loc_est(t)+para.nbins/2,para.nbins)-para.nbins/2;
      set(h_err,'YData',x_err)
      
      set(h_title,'String',sprintf('t=%d',t))
      
      if plt_vid
        drawnow
        frame = getframe(gcf);
        writeVideo(video.obj,frame);
      else
        pause(1/para.f)
      end
    end
  end
  if plt && plt_vid
    close(h.video.obj)
  end
  figure
  subplot(2,1,1)
  hold on
  
  bar(data.time,(~data.run_bool)*para.nbins,1,'FaceColor',[0.8,0.8,0.8])
  scatter(data.time,data.loc,'.k')
  size(loc_est)
  scatter(data.time,loc_est,'.r')
  ylabel('bin position s')
  xlim([0,data.time(end)])
  ylim([0,para.nbins])
  
  %%%% check: coding errors mostly during standing still?
  subplot(2,1,2)
  hold on
  bar(data.time,(~data.run_bool)*para.nbins/2,1,'FaceColor',[0.8,0.8,0.8])
  bar(data.time,-(~data.run_bool)*para.nbins/2,1,'FaceColor',[0.8,0.8,0.8])
  x_err = mod(data.loc'-loc_est+para.nbins/2,para.nbins)-para.nbins/2;
  scatter(data.time,x_err,'.r')
  xlim([0,data.time(end)])
  ylim([-para.nbins/2,para.nbins/2])
  xlabel('t [s]')
  ylabel('estimation error \Delta s')
  
  out = struct;
  out.N_err_lr = sum(x_err(data.run_bool) > 5);
  out.N_err_nlr = sum(x_err(~data.run_bool) > 5);
  disp(sprintf('Errors during running: %d',out.N_err_lr))
  out.N_err_lr/sum(data.run_bool)
  disp(sprintf('Errors during resting: %d',out.N_err_nlr))
  out.N_err_lr/sum(~data.run_bool)
  out.error_rate = mean(abs(x_err));
  disp(sprintf('error per timebin: %5.3g',out.error_rate))
  print(pathcat(pwd,'bayes_estimation.png'),'-dpng','-r300')
  
end


function [p_xs] = get_p_xs(p_xs,act)
  % calculates probability of obtaining observed activity
  
  N = length(act);
  for n = 1:N
    if ~act(n)
      p_xs(n,:) = 1-p_xs(n,:);
    end
  end
end



function data = read_data(fileName)
  %%% read data in from file "fileName"
  data = struct;
%    [folder,~,ext] = fileparts(fileName);
  folder = fileName;
%    ext
%    switch ext
%      case '.csv'
%        
%        data_ld = csvread(trainName);
%        data.time = data_ld(:,1)*1/para.f;    %% time = Tx1 array containing timestamps
%        data.loc = data_ld(:,4);              %% loc = Txd array containing location data
%        data.act = data_ld(:,5:end)';         %% act = NxT array containing activity data
%        data.speed = data_ld(:,2:3);
%      case {'.mat',''}
      pathAct = dir(pathcat(folder,'*OnACID.mat'));
      pathAct = pathcat(folder,pathAct.name);
      
%        Ca could be used by scaling prob from 0 to 1(=threshold from baseline firing rate)
      data_ld = load(pathAct,'S');
      
      data.N = size(data_ld.S,1);       %% number of neurons
      %% test if data is ordered properly
      if data.N > 4000
        data_ld.S = transpose(data_ld.S);
        data.N = size(data_ld.S,1);
      end
      
      for n = 1:data.N
        [data.act(n,:),~,~] = get_spikes(data_ld.S(n,:));
%          [~,md,sd_r] = get_spikes(data_ld.S(n,:));
%          sp_thr = md + 2*sd_r;
%          data.act(n,:) = min(1,data_ld.S(n,:)/sp_thr);
      end
      data.act = data.act > 0;
      
      pathBH = dir(pathcat(folder,'*aligned.mat'));
      pathBH = pathcat(folder,pathBH.name);
      
      data_ld = load(pathBH);
      data.time = data_ld.alignedData.resampled.time;
      data.loc = data_ld.alignedData.resampled.binpos;
      data.speed = data_ld.alignedData.resampled.speed;
      data.run_bool = data_ld.alignedData.resampled.longrunperiod;
      
      pathPC = dir(pathcat(folder,'*PC_fields.mat'));
      pathPC = pathcat(folder,pathPC.name);
      
      data_ld = load(pathPC);
      PC_fields = data_ld.PC_fields;
      data.p_vals = zeros(data.N,1);
      for n=1:length(PC_fields)
        data.p_vals(n) = PC_fields(n).MI.p_value;
      end
      
%    end
end


function data_out = hsm(data)
  %%% adapted from python version of caiman
  %%% Robust estimator of the mode of a data set using the half-sample mode.
  %%% versionadded: 1.0.3
    
  %%% Create the function that we can use for the half-sample mode
  %%% needs input of sorted data
  
  Ndat = length(data);
  if Ndat == 1
      data_out = data(1);
  elseif Ndat == 2
      data_out = mean(data);
  elseif Ndat == 3
      i1 = data(2) - data(1);
      i2 = data(3) - data(2);
      if i1 < i2
          data_out = mean(data(1:2));
      elseif i2 > i1
          data_out = mean(data(2:end));
      else
          data_out = data(2);
      end
  else
      wMin = inf;
      N = floor(Ndat/2) + mod(Ndat,2);
      for i = 1:N
          w = data(i+N-1) - data(i);
          if w < wMin
              wMin = w;
              j = i;
          end
      end
      data_out = hsm(data(j:j+N-1));
  end
end


function [spikes,md,sd_r] = get_spikes(data)
  
  data_hsm = data(data>0);
  md = hsm(sort(data_hsm));       % Find the mode
  
  % only consider values under the mode to determine the noise standard deviation
  ff1 = data_hsm - md;
  ff1 = -ff1 .* (ff1 < 0);
  
  % compute 25 percentile
  ff1 = sort(ff1);
  ff1(ff1==0) = NaN;
  Ns = round(sum(ff1>0) * .5);
  
  % approximate standard deviation as iqr/1.349
  iqr_h = ff1(end-Ns);
  sd_r = 2 * iqr_h / 1.349;
  data_thr = md+2*sd_r;
  spikes = floor(data/data_thr);
%    spikeNr = sum(floor(data/data_thr));
  
end