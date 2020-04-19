
function [PCs] = find_PCs(PCs,p_value,Bayes_thr,A_ratio_thr,p_mass_thr)
  tic
  nSes = size(PCs.status,2);
  
  nROI = sum(PCs.status(:,:,2),1);
  ROI_ct = sum(PCs.status(:,:,2),2);
  
  nC = size(PCs.status,1);
  
  PCs.MI.p_value(PCs.MI.p_value==0.001) = 10^(-10);    %% need this - can't get any lower than 0.001 with 1000 shuffles...
  PC_idx = {};
  for s = 1:nSes
    
    N = nROI(s);
    FDR_control = linspace(1,N,N)'/N*p_value;
    
    idxes = find(PCs.status(:,s,2));
    [sort_PC,n_idx] = sort(PCs.MI.p_value(PCs.status(:,s,2),s));
    
    PC_idx{s} = n_idx(sort_PC<FDR_control);
    PC_idx{s} = idxes(PC_idx{s});
    
%      figure()
%      hold on
%      plot(FDR_control,'r--')
%      plot(sort_PC,'k')
%      waitforbuttonpress
  end
  
  A_ratio = PCs.fields.parameter(:,:,:,2,1)./PCs.fields.parameter(:,:,:,1,1);
  Bayes = PCs.Bayes.factor(:,:,1) - PCs.Bayes.factor(:,:,2);
  for c = 1:nC
    for s = 1:nSes
      
      if PCs.status(c,s,2)
        if ismember(c,PC_idx{s}) && Bayes(c,s) > Bayes_thr
          idx_change = squeeze(A_ratio(c,s,:) < A_ratio_thr | PCs.fields.posterior_mass(c,s,:) < p_mass_thr);
        else
          idx_change = true(3,1);
        end
        
        
        if any(idx_change)
          PCs.fields.status(c,s,idx_change) = NaN;
          
          PCs.status(c,s,:) = false;
          PCs.status(c,s,2) = true;
          PCs.status(c,s,PCs.fields.status(c,s,~isnan(PCs.fields.status(c,s,:)))) = true;
        end
      end
    end
  end
  toc
end