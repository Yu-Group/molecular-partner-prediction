% matlab code from: https://github.com/DanuserLab/cmeAnalysis

addpath(genpath(cmeAnalysis));


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
% Saves tracks for RFP (clathrin) and tracks at RFP locations for GFP (auxilin)