%% Convert TWL time series from .mat to .csv

% Pre-computed TWL from on of the six of Alfredo's time series
% Not sure where this one is. Think these values were calculated by Ian
infile = 'D:\crs\proj\2023_NCB_recovery\NCB_TWL_R2_6.mat'
load(infile)
n = length(TT)

ofile = 'D:\crs\proj\2023_NCB_recovery\annual_max.csv'

%% Convert TT from datenum to Datetime object
t = datetime(TT,'ConvertFrom','datenum');
%% Find max valuse for each year
yr = unique(t.Year);
ny = length(yr)
maxZZ = nan*ones(ny,1);
maxTWL = nan*ones(ny,1);
maxR2 = nan*ones(ny,1);
maxR2_95up  = nan*ones(ny,1);
naxR2_95low  = nan*ones(ny,1);

for i = 1:ny
   idx = t.Year==yr(i);
   maxZZ(i) = max( ZZ(idx) );
   maxTWL(i) = max( TWL(idx) )
   maxR2(i) = max( R2(idx) )
   maxR2_95up(i) = max( R2_95up(idx) )
   maxR2_95low(i) = max( R2_95low(idx) )
end

%% write to csv
fid = fopen( ofile, 'w')
fprintf(fid,'Year, ZZ, TWL, R2, R2_95low, R2_95up\n' )
for i = 1:ny
   fprintf(fid,'%d, %.3f, %.3f, %.3f, %.3f, %.3f\n', ...
      yr(i), maxZZ(i), maxTWL(i), maxR2(i), maxR2_95low(i), maxR2_95up(i) );
end
fclose(fid)
