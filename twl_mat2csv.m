%% Convert TWL time series from .mat to .csv

% Pre-computed TWL from on of the six of Alfredo's time series
% Not sure where this one is. Think these values were calculated by Ian
infile = 'D:\crs\proj\2023_NCB_recovery\NCB_TWL_R2_6.mat'
load(infile)
n = length(TT)

ofile = 'D:\crs\proj\2023_NCB_recovery\NCB_TWL_R2_6.csv'
%%
fid = fopen( ofile, 'w')
fprintf(fid,'Time, TT, ZZ, TWL, R2, R2_95low, R2_95up\n' )
for i = 1:n
   fprintf(fid,'''%s'', %f, %.3f, %.3f, %.3f, %.3f, %.3f\n', ...
      datestr(TT(i), 'YYYY-mm-dd HH:MM'), TT(i), ZZ(i), TWL(i), R2(i), R2_95low(i), R2_95up(i) );
end
fclose(fid)