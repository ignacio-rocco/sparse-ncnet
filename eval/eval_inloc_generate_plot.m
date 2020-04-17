% adjust path and experiment name
inloc_demo_path = '/your_path_to/InLoc_demo_old/';
addpath(fullfile(pwd,'..','lib_matlab'));
run(fullfile(inloc_demo_path,'startup.m'))
addpath(inloc_demo_path)
[ params ] = setup_project_ht_WUSTL

params.gt.dir='lib_matlab'
params.output.dir=''

densePE = load('../datasets/inloc/shortlists/densePE_top100_shortlist_cvpr18')
inloc = load('../datasets/inloc/shortlists/densePV_top10_shortlist_cvpr18')
dense_ncnet=load('../datasets/inloc/shortlists/ncnet_shortlist_neurips18.mat');
sparse_ncnet_1600_hard=load('../datasets/inloc/shortlists/sparsencnet_shortlist_1600_hard.mat');
sparse_ncnet_3200_hard_soft=load('../datasets/inloc/shortlists/sparsencnet_shortlist_3200_hard_soft.mat');

% define custom palette (kind of color-blind friendly)
cb1=[0,0,0]/255
cb2=[0,73,73]/255
cb3=[0,146,146]/255
cb4=[255,109,182]/255
cb5=[255,182,119]/255
cb6=[73,0,146]/255
cb7=[0,109,219]/255
cb8=[182,109,255]/255
cb9=[109,182,255]/255
cb10=[182,219,255]/255
cb11=[146,0,0]/255
cb12=[146,73,0]/255
cb13=[219,209,0]/255
cb14=[36,255,36]/255
cb15=[255,255,109]/255

% plot
method = struct();
i=1
method(i).ImgList = sparse_ncnet_3200_hard_soft.ImgList;
method(i).description = 'InLoc + Sparse-NCNet (H+S, 200\times150)';
method(i).marker = '-';
method(i).color = 'black';
method(i).ms = 8
i=i+1
method(i).ImgList = sparse_ncnet_1600_hard.ImgList;
method(i).description = 'InLoc + Sparse-NCNet (H, 100\times75)';
method(i).marker = '+-.';
method(i).color = cb7;
method(i).ms = 8
i=i+1
method(i).ImgList = dense_ncnet.ImgList;
method(i).description = 'InLoc + NCNet (H)';
method(i).marker = 's-.';
method(i).color = cb10;
method(i).ms = 8
i=i+1
method(i).ImgList = inloc.ImgList;
method(i).description = 'InLoc';
method(i).marker = 'x--';
method(i).color = cb4;
method(i).ms = 8
i=i+1
method(i).ImgList = densePE.ImgList;
method(i).description = 'DensePE';
method(i).marker = 'o--';
method(i).color = cb14;
method(i).ms = 8

ht_plotcurve_WUSTL

