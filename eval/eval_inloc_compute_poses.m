% Evaluate Sparse-NCNet matches on top of densePE shortlist

% adjust path and experiment name
inloc_demo_path = '/your_path_to/InLoc_demo_old/';
experiment = 'sparsencnet_3200_hard_soft';

if exist('ncnet_path')==0
	ncnet_path=fullfile(pwd,'..');
end
matches_path = fullfile(ncnet_path,'datasets','inloc',matches');

sorted_list_fn = 'densePE_top100_shortlist_cvpr18.mat';
sorted_list = load(fullfile(ncnet_path,'datasets','inloc','shortlists',sorted_list_fn));

addpath(fullfile(ncnet_path,'lib_matlab'));

% init paths
cd(inloc_demo_path)
startup;
[ params ] = setup_project_ht_WUSTL;
% add extra parameters
params.output.dir = experiment;
params.output.gv_nc4d.dir = fullfile(params.output.dir, 'gv_nc4d'); % dense matching results path
params.output.gv_nc4d.matformat = '.gv_nc4d.mat'; % dense matching results 
params.output.pnp_nc4d.matformat = '.pnp_nc4d_inlier.mat'; % PnP results 
% redefine gt poses path
params.gt.dir = fullfile(ncnet_path,'lib_matlab')

Nq=length(sorted_list.ImgList);

pnp_topN=10;
% set parameters
% params.ncnet.thr = 0.75;
params.ncnet.thr = 0;
params.ncnet.pnp_thr = 0.2;
params.output.pnp_nc4d_inlier.dir = fullfile(params.output.dir, ...
  sprintf('top_%i_PnP_thr%03d_rthr%03d',pnp_topN,params.ncnet.thr*100,params.ncnet.pnp_thr*100));
NC4D_matname = fullfile(params.output.dir, 'shortlist_densePE.mat');

% compute poses from matches
ir_top100_NC4D_localization_pnponly;

do_densePV=true

if do_densePV
  params.output.synth.dir = fullfile(params.output.dir, ...
  sprintf('top_%i_thr%03d_rthr%03d_densePV'));

  nc4dPV_matname = fullfile(params.output.dir, 'shortlist_densePV.mat');
  % run pose verification by rendering sythetic views
  ht_top10_NC4D_PV_localization
end
