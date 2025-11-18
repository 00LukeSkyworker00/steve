
# STEVE: Slot-Transformer for Videos 
*NeurIPS 2022*

#### [[arXiv](https://arxiv.org/abs/2205.14065)] [[project](https://sites.google.com/view/slot-transformer-for-videos)]

This is the **official PyTorch implementation** of the STEVE model. 

In this branch, we provide the scripts to evaluate and compute the video-level FG-ARI.

### Authors
Gautam Singh and Yi-Fu Wu and Sungjin Ahn

### Dataset
Any of the MOVi-A/B/C/D/E datasets with masks can be downloaded using the script `download_movi_with_masks.py`. The remaining datasets used in the paper i.e., CATER-Tex, MOVi-Tex, MOVi-Solid, Youtube-Traffic and Youtube-Aquarium, shall be released soon.

### Evaluation
Following is an example command to run the evaluation script:
```bash
python eval_fgari_video.py --data_path "eval_data/*" --trained_model_paths "saved_model_seed_0.pt" "saved_model_seed_1.pt" "saved_model_seed_2.pt" 
```

### Outputs
The script produces the standard output on the terminal screen showing the FG-ARI mean and standard deviation.

### Evaluation
See the branch named `evaluate` for the evaluation scripts.

### Citation
```
@inproceedings{
  singh2022simple,
  title={Simple Unsupervised Object-Centric Learning for Complex and Naturalistic Videos},
  author={Gautam Singh and Yi-Fu Wu and Sungjin Ahn},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=eYfIM88MTUE}
}
```
