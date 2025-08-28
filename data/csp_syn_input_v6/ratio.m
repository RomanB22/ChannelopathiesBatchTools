% calculate ratio map as map_exp/map_uniform
%clear all;
close all;
load('m1l23-to-m1-csp.mat');
save = 1;
rescaled = 1;
plotsigma = 1;
cellnum = 18;

map_uni_raw = load('map_uni_BS0284_memb_BS0477_morph.dat');
if rescaled, map_uni_rescaled_raw = load(['map_scaled_scracm' num2str(cellnum) '_BS0284_memb_BS0477_morph.dat']); end
map_uni=zeros(10,30);
counter = 0;
for ii=1:10 
    for jj=1:30 
        counter = counter + 1;
        map_uni(ii,jj) = map_uni_raw(counter)*1000; % nA -> pA
        if map_uni(ii,jj) ==  1, map_uni(ii,jj) = NaN; end % only include pixels where there was dendrite
        if rescaled, 
              map_uni_rescaled(ii,jj) = map_uni_rescaled_raw(counter)*1000; 
             if map_uni_rescaled(ii,jj) ==  1, map_uni_rescaled(ii,jj) = NaN; end
        end % nA -> pA
    end
end

% Calculate ratio and density maps
minRatio=0;
maxRatio=15;
toDensity = 1.5 /8; % spines/um  1/8 L2/3->L5B conns
map_exp = squeeze(maps(:,:,cellnum))';


ratio_map = map_exp ./ map_uni;  % calculate ratio
ratio_map(isnan(ratio_map)) = 0;  % set nans to 0 (no dendrite)
ratio_map = max(min(ratio_map, maxRatio), minRatio); % set min max ratio values

density_map = ratio_map .* toDensity;
density_map_nan = density_map;

if rescaled, 
    ratio_rescaled_map = map_exp ./ map_uni_rescaled; 
    ratio_rescaled_map(isnan(ratio_rescaled_map)) = 0;  % set nans to 0 (no dendrite)
    ratio_rescaled_map = max(min(ratio_rescaled_map, maxRatio), minRatio); % set min max ratio values
end


if save,
    ftemp = fopen(['ratio_scracm' num2str(cellnum) '_BS0284_memb_BS0477_morph.dat'],'w' );
    ftemp2 = fopen(['density_scracm' num2str(cellnum) '_BS0284_memb_BS0477_morph.dat'],'w' );
    ftemp3 = fopen(['radial_scracm' num2str(cellnum) '_BS0284_memb_BS0477_morph.dat'],'w' );
    ftemp4 = fopen(['exp_radial_scracm' num2str(cellnum) '_BS0284_memb_BS0477_morph.dat'],'w' );
    for ii=1:10 
        for jj=1:30 
            fprintf(ftemp, '%f\r\n', ratio_map(ii,jj));
            fprintf(ftemp2, '%f\r\n', density_map(ii,jj));
            if density_map_nan(ii,jj) == 0,
                density_map_nan(ii,jj) = nan;
            end
%             if map_exp(ii,jj) == 0,
%                 map_exp(ii,jj) = nan;
%             end
        end
    end
    
   density_radial = nanmean(density_map_nan);
   density_radial(isnan(density_radial)) = 0;
   
map_exp_radial = mean(map_exp);
%    map_exp_radial(isnan(map_exp_radial)) = 0;
   
     for jj=1:30 
            fprintf(ftemp3, '%f\r\n', density_radial(jj));
            fprintf(ftemp4, '%f\r\n', -map_exp_radial(jj));
      end
        fclose(ftemp);
        fclose(ftemp2);
        fclose(ftemp3);
        fclose(ftemp4);
end

h1=figure('Position', [100, 100, 1500, 800]);
%colormap(flipud(jet));
maxC = 15;
colormap(jet);
subplot(171); imagesc(-map_exp'); title(['Experimental cell' num2str(cellnum)]); c=colorbar(); ylabel(c, '-pA'); axis equal; axis tight;
subplot(172); imagesc(-map_uni'); title('Uniform sim'); colorbar(); c=colorbar(); ylabel(c, '-pA'); axis equal; axis tight;
subplot(173); imagesc(ratio_map', [0, maxC]); title('Ratio = Exp/Sim'); colorbar(); c=colorbar(); ylabel(c, 'ratio'); axis equal; axis tight;
subplot(174); imagesc(density_map', [0, maxC*toDensity]); title('Density map (syns/um)'); colorbar(); c=colorbar(); ylabel(c, 'syns/um');axis equal; axis tight; 
subplot(175); imagesc(density_radial'); title('Density radial (syns/um)'); colorbar(); c=colorbar(); ylabel(c, 'syns/um'); axis equal; axis tight;
if rescaled, 
    subplot(176); imagesc(-map_uni_rescaled'); title('Rescaled uniform sim'); colorbar(); c=colorbar(); ylabel(c, '-pA'); axis equal; axis tight;
    subplot(177); imagesc(ratio_rescaled_map', [0, maxC]); title('Ratio 2 = Exp/Sim rescaled'); colorbar(); c=colorbar(); ylabel(c, 'ratio'); axis equal; axis tight;
end

%set(h1,'PaperUnits','inches','PaperPosition',[0 0 7 3.5])
%print(h1,'-dpng','-r800',['scracm' num2str(cellnum) '_BS0248_memb_BS0477_morph_max15_5.png'])

