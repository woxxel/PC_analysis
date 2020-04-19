
function [PCs,frac_active] = find_PCs(PCs,p_value,perc_thr,width_thr)
  
  nSes = size(PCs.status,2);
  
  nROI = sum(PCs.status(:,:,2),1);
  ROI_ct = sum(PCs.status(:,:,2),2);
  
  nC = size(PCs.status,1);
  
  PC_idx = {};
  for s = 1:nSes
    
    N = nROI(s);
    FDR_control = linspace(1,N,N)'/N*p_value;
    
    [sort_PC,n_idx] = sort(PCs.MI.p_value(PCs.status(:,s,2),s));
    
    idxes = find(PCs.status(:,s,2));
    
    PC_idx{s} = n_idx(sort_PC<FDR_control);
    PC_idx{s} = idxes(PC_idx{s});
  end
  
  
  frac_active = zeros(nC,nSes,size(PCs.fields.status,3))*NaN;
  for c = 1:nC
    for s = 1:nSes
      
      if PCs.status(c,s,2)
        if ismember(c,PC_idx{s})%PCs.MI.p_value(c,s) < p_value%prctile_arr(c,s)
          
%            frac_active(c,s,~isnan(PCs.fields.status(c,s,:))) = nansum(PCs.trials(c,s).field_rate > 1.5*PCs.firingrate(c,s),2)./sum(~isnan(PCs.trials(c,s).field_rate),2);
          
          frac_active(c,s,~isnan(PCs.fields.status(c,s,:))) = nansum(PCs.trials(c,s).field_rate > 1.1*PCs.trials(c,s).rate,2)./sum(~isnan(PCs.trials(c,s).field_rate),2);
%            squeeze(frac_active(c,s,:))
          
          idx_change = squeeze(frac_active(c,s,:) < perc_thr | PCs.fields.width(c,s,:) < width_thr);
          
        else
          idx_change = true(5,1);
        end
        
        
        if any(idx_change)
          PCs.fields.center(c,s,idx_change) = NaN;
          PCs.fields.status(c,s,idx_change) = NaN;
          PCs.fields.width(c,s,idx_change) = NaN;
          
          PCs.status(c,s,:) = false;
          PCs.status(c,s,2) = true;
          
          PCs.status(c,s,PCs.fields.status(c,s,~isnan(PCs.fields.status(c,s,:)))) = true;
        end
      end
    end
  end

end