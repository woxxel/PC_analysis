
def plot_fieldshift_theory(A_offset,tau_s,sigma)

  nbin = 100;
  nSes = 20;
  x_arr = linspace(-nbin/2,nbin/2,nbin+1);
  s_arr = linspace(1,nSes,nSes);

  F_est = @(x,data)...
        x(4)/(sqrt(2*pi)*x(2))*exp(-((data-x(1)).^2./(2*x(2)^2))) + x(3)/length(data);     %% gaussian + linear offset

  exp_dist = @(a,x_data) a(2)*exp(-x_data/a(1));

  paras_cont = exp_dist([tau_s(1),A_offset(1)],s_arr);
  paras_disc = exp_dist([tau_s(2),A_offset(2)],s_arr);

  f1 = figure('position',[100 100 600 200]);
  ax21 = subplot(1,3,1);
  hold on
  sig_arr = false(1,nbin+1);
  
  sig_arr(nbin/2+1-2*sigma:nbin/2+1+2*sigma) = true;
%    round(2*sigma)
  bar(ax21,x_arr,sig_arr*0.08,1,'FaceColor',[0.8,0.8,0.8],'EdgeColor','None')

  ax22 = subplot(1,3,2);
  hold on
  sig_arr_cdf = false(1,nbin/2+1);
  sig_arr_cdf(1:2*sigma) = true;
  bar(ax22,x_arr(nbin/2+1:end),0.16*sig_arr_cdf,1,'FaceColor',[0.8,0.8,0.8],'EdgeColor','None')

  ax23 = subplot(1,3,3);
  hold on
  bar(ax23,x_arr(nbin/2+1:end),sig_arr_cdf,1,'FaceColor',[0.8,0.8,0.8],'EdgeColor','None')



  f2 = figure('position',[500 500 600 200]);
  ax11 = subplot(1,2,1);
  hold on
  plot(ax11,1.96*[sigma,sigma],[0,1],'k:')
%    ax1 = subplot(1,2,1);
%    hold(ax1,'on')
%    ax2 = subplot(1,2,2);
%    hold(ax2,'on')

  p_theory_tmp = F_est([0,sigma,1-paras_cont(1),paras_cont(1)],x_arr);
  p_theory_1 = p_theory_tmp;
  p_theory_tmp(nbin/2+2:end) = p_theory_tmp(nbin/2+2:end) + fliplr(p_theory_tmp(1:nbin/2));   %% take absolute
  p_theory_tmp(1:nbin/2) = [];
  bar(ax22,x_arr(nbin/2+1:end),p_theory_tmp,'FaceColor','k')
  ylim(ax22,[0,0.16])
  ylabel(ax21,'p(\Delta x)')
  xlabel(ax21,'\Delta x')
  xlabel(ax22,'|\Delta x|')


  cdf_theory_1 = cumsum(p_theory_tmp);

%    bar(ax21,x_arr,p_theory_1,'FaceColor','k')
%    ylim(ax21,[0,0.08])
%    stairs(ax23,x_arr(nbin/2+1:end),cdf_theory_1,'b')
%    ylim(ax23,[0,1])
%    ylabel(ax23,'cdf(|\Delta x|)')
%    xlabel(ax23,'|\Delta x|')

  cdf_uniform = linspace(0,1,nbin/2+1);
  D_max = max(cdf_theory_1 - cdf_uniform);

%    length(cdf_uniform)
%    x_arr(nbin/2+1:end)
%    length(x_arr(nbin/2+1:end))

  plot(ax11,x_arr(nbin/2+1:end),cdf_uniform,'r--','LineWidth',2)

%    nSes = 20;
  D_KL_cont = zeros(1,nSes);
  D_KL_disc = zeros(1,nSes);

  for i = 1:nSes
    col_tmp = (i-1)/(nSes+1-10);
    col_cont = [col_tmp,col_tmp,1];
    col_disc = [1,col_tmp,col_tmp];

    p_theory_cont = F_est([0,sigma,1-paras_cont(i),paras_cont(i)],x_arr);
%      plot(ax1,x_arr,p_theory_cont,'Color',col_cont,'LineWidth',2)

    p_theory_disc = F_est([0,sigma,1-paras_disc(i),paras_disc(i)],x_arr);
%      plot(ax2,x_arr,p_theory_disc,'Color',col_disc,'LineWidth',2)

    p_theory_cont(nbin/2+2:end) = p_theory_cont(nbin/2+2:end) + fliplr(p_theory_cont(1:nbin/2));   %% take absolute
    p_theory_cont(1:nbin/2) = [];                                                        %% remove negative side of distribution
    cdf_theory_cont = cumsum(p_theory_cont);

    p_theory_disc(nbin/2+2:end) = p_theory_disc(nbin/2+2:end) + fliplr(p_theory_disc(1:nbin/2));   %% take absolute
    p_theory_disc(1:nbin/2) = [];           %% remove negative side of distribution
    cdf_theory_disc = cumsum(p_theory_disc);

%      if ismember(i,[2,4,6])
%        stairs(ax11,x_arr(nbin/2+1:end),cdf_theory_cont,'Color',col_cont,'LineWidth',2,'DisplayName','Model')
%        stairs(ax11,x_arr(nbin/2+1:end),cdf_theory_disc,'Color',col_disc,'LineWidth',2,'DisplayName','Model')
%      end
    [~,max_pos] = max(abs(cdf_theory_1 - cdf_theory_cont));
    D_KL_cont(i) = cdf_theory_1(max_pos) - cdf_theory_cont(max_pos);

    [~,max_pos] = max(abs(cdf_theory_1 - cdf_theory_disc));
    D_KL_disc(i) = cdf_theory_1(max_pos) - cdf_theory_disc(max_pos);

  end
       %% remove negative side of distribution
  plot(ax11,x_arr(nbin/2+1:end),cdf_theory_1,'Color','b','LineWidth',2,'DisplayName','Model')

  xlabel(ax11,'|\Delta x| [cm]')
  ylabel(ax11,'cdf(|\Delta x|)')
  xlim(ax11,[0,nbin/2])
  ylim(ax11,[0,1])

  set(ax11,'Position',[0.18 0.22, 0.75 0.75])
  set(gcf,'Position',[100 100 300 200])
%    ax12 = axes('position',[0.37,0.35,0.1,0.2]);
%    hold on
%    plot(ax12,[0,nSes],[0,0],'k:')
%    plot(ax12,[0,nSes],[D_max,D_max],'k--')
%    plot(ax12,s_arr,D_KL_cont,'b')
%    plot(ax12,s_arr,D_KL_disc,'r')
%    xlabel(ax12,'\Delta s')
%    ylabel(ax12,'D_{KL}')
%    ylim([-0.2,0.6])

%    ax13 = subplot(1,2,2);
%    hold on
%    plot(ax13,s_arr,paras_cont,'b')
%    plot(ax13,s_arr,paras_disc,'r')
%    xlim(ax13,[0,nSes])
%    ylim(ax13,[0,1])
%    set(ax13,'yscale','log')
%    ylabel('r_{stable}')
%    xlabel('\Delta s')
  close(f1)
  sv_ext = 'png';
%    figure(f1)
%    path = pathcat('/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Data/Figures',sprintf('shiftdistr_theory_%4.2f.%s',A_offset,sv_ext));
%    print(path,sprintf('-d%s',sv_ext),'-r300')
%    disp(sprintf('Figure saved as %s',path))

  figure(f2)
  path = pathcat('/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Data/Figures',sprintf('shiftdistr_sketch_%3.1f.%s',A_offset(1),sv_ext));
  print(path,sprintf('-d%s',sv_ext),'-r300')
  disp(sprintf('Figure saved as %s',path))
end
