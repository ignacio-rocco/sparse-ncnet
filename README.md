# Sparse Neighbourhood Consensus Networks

## About
This is the implementation of the paper "Efficient Neighbourhood Consensus Networks via Submanifold Sparse Convolutions" by Ignacio Rocco, Relja Arandjelović and Josef Sivic [[arXiv](https://arxiv.org/abs/2004.10566)].

## Installation
For installation instructions, please see [INSTALL.md](INSTALL.md).

## Quickstart
For a demo of the method, see the Jupyter notebook [`demo/demo.ipynb`](demo/demo.ipynb).

## Training
To train a model with the default parameters run `python train.py`.

## Evaluation on HPatches Sequences
1. Browse to `eval/`. 
2. Run `python eval_hpatches_extract.py` adjusting the checkpoint and experiment name.
3. Use `eval_hpatches_generate_plot.ipynb` with the appropriate experiment name to generate the plot.

## Evaluation on InLoc
In order to run the InLoc evaluation, you first need to clone the [InLoc demo repo](https://github.com/HajimeTaira/InLoc_demo), and download and compile all the required depedencies. Then:

1. Browse to `eval/`. 
2. Run `python eval_inloc_extract.py` adjusting the checkpoint and experiment name.
This will generate a series of matches files in the `datasets/inloc/matches/` folder that then need to be fed to the InLoc evaluation Matlab code. 
3. Modify the `eval/eval_inloc_compute_poses.m` file provided in this repo to indicate the path of the InLoc demo repo, and the name of the experiment (the particular folder name inside `datasets/inloc/matches/`), and run it using Matlab.
4. Use the `eval/eval_inloc_generate_plot.m` file to plot the results from shortlist file generated in the previous stage: `/your_path_to/InLoc_demo_old/experiment_name/shortlist_densePV.mat`. Precomputed shortlist files are provided in `datasets/inloc/shortlist`.

## Evaluation on Aachen Day-Night
In order to run the Aachen Day-Night evaluation, you first need to clone the [Visualization benchmark repo](https://github.com/tsattler/visuallocalizationbenchmark), and download and compile [all the required depedencies](https://github.com/tsattler/visuallocalizationbenchmark/tree/master/local_feature_evaluation) (in particular, you'll need to compile Colmap if you have not done so yet). Then:

1. Browse to `eval/`. 
2. Run `python eval_aachen_extract.py` adjusting the checkpoint and experiment name.
3. Copy the `eval_aachen_reconstruct.py` file to `visuallocalizationbenchmark/local_feature_evaluation` and run it in the following way:

```
python eval_aachen_reconstruct.py 
	--dataset_path /path_to_aachen/aachen 
	--colmap_path /local/colmap/build/src/exe
	--method_name experiment_name
```
4. Upload the file `/path_to_aachen/aachen/Aachen_eval_[experiment_name].txt` to `https://www.visuallocalization.net/` to get the results on this benchmark.

## BibTeX 

If you use this code in your project, please cite our paper:

````
@article{Rocco20,
        author       = "Rocco, I. and Arandjelovi\'c, R. and Sivic, J."
        title        = "Efficient Neighbourhood Consensus Networks via Submanifold Sparse Convolutions",
        journal      = "arXiv preprint arXiv:2004.10566",
        year         = "2020",
        }
````

