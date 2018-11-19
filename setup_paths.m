function setup_paths()

% Add the neccesary paths

[pathstr, name, ext] = fileparts(mfilename('fullpath'));

addpath(genpath([pathstr '/Processing/']));

% Tracker implementation
addpath([pathstr '/implementation/']);

% Utilities
addpath([pathstr '/utils/']);

%networks
addpath([pathstr '/networks/'])

% The feature extraction
addpath(genpath([pathstr '/feature_extraction/']));

% Matconvnet
addpath([pathstr '/external_libs/matconvnet/matlab/mex/']);
addpath([pathstr '/external_libs/matconvnet/matlab']);
addpath([pathstr '/external_libs/matconvnet/matlab/simplenn']);
vl_setupnn;

% PDollar toolbox
addpath(genpath([pathstr '/external_libs/pdollar_toolbox/channels']));

% Mtimesx
addpath([pathstr '/external_libs/mtimesx/']);