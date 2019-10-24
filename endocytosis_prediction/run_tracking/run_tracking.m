% matlab code from: https://github.com/DanuserLab/cmeAnalysis

addpath(genpath('cmeAnalysis'));
data = loadConditionData('auxilin_data/A7D2', {'TagRFP', 'EGFP'}, {'rfp', 'gfp'}, 'Parameters', [1.49 200 16]);
cmeDataViewer(data(1))

%{
cmeAnalysis

OR

data = loadConditionData
cmeAnalysis(data)

N.A.: 1.49
magnification: 200
camera pixel size: 16
Root directory: folder w/ all cells
number of channels: 2
    (pick RFP then GFP)
fluorescent marker 1: rfp
fluorescent marker 2: gfp
channel 1 name: "TagRFP"
channel 2 name: "EGFP"


%}


% Saves into Detection, Tracking, Analysis folders
% Visualizing: cmeDataViewer(data)
% If there are multiple folders, % Visualizing: cmeDataViewer(data(folder_num))
% Saves tracks for RFP (clathrin) and tracks at RFP locations for GFP (auxilin)


%{
FINAL OUTPUT

Remove outliers? (y/n) n


Lifetime analysis - processing: 100%
Max. intensity threshold on first 4 frames: 1314.53
95th percentile of 1st frame distribution: 1453.73
Lifetime distribution percentiles (5th, 25th, 50th, 75th, 95th):
  CSs:  [NaN, 5.4, 7.7, 13.4, 37.5] s
  CCPs: [5.7, 22.9, 45.8, 86.9, 198.3] s
Error using griddedInterpolant
The coordinates of the input points must be finite values; Inf and NaN
are not permitted.

Error in interp1 (line 149)
        F = griddedInterpolant(X,V,method);

Error in runLifetimeAnalysis (line 426)
        lftPctCS = interp1(lftCDF(uidx), lftRes.t(uidx), [0.05 0.25 0.5
        0.75 0.95]);

Error in cmeAnalysis (line 151)
    res.lftRes = runLifetimeAnalysis(data, lopts{:},...
 

%}