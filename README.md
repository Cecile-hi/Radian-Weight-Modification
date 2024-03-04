# RWM
This is the official project of the Radian Weight Modification (RWM) method proposed by our paper titled "What to remember: Self-adaptive continual learning for audio deepfake detection", published on the the 38th Annual AAAI Conference on Artificial Intelligence (AAAI 2024) (https://arxiv.org/abs/2312.09651)

## Previous work
The RWM is based on our previous work 'Do You Remember? Overcoming Catastrophic Forgetting for Fake Audio Detection' published on the 40th International Conference on Machine Learning (ICML 2023) (https://arxiv.org/abs/2308.03300).

## Citation

If you find this toolkit useful, please consider citing following papers.
```
@InProceedings{pmlr-v202-zhang23au,
  title = 	 {Do You Remember? {O}vercoming Catastrophic Forgetting for Fake Audio Detection},
  author =       {Zhang, Xiaohui and Yi, Jiangyan and Tao, Jianhua and Wang, Chenglong and Zhang, Chu Yuan},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {41819--41831},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/zhang23au/zhang23au.pdf},
  url = 	 {https://proceedings.mlr.press/v202/zhang23au.html}
}

```
```
@article{zhang2023remember,
  title={What to remember: Self-adaptive continual learning for audio deepfake detection},
  author={Zhang, Xiaohui and Yi, Jiangyan and Wang, Chenglong and Zhang, Chuyuan and Zeng, Siding and Tao, Jianhua},
  journal={arXiv preprint arXiv:2312.09651},
  year={2023}
}
```
### Introduction 
We propose a continual learning approach called Radian Weight Modification (RWM) for audio deepfake detection. The fundamental concept underlying RWM involves categorizing all classes into two groups: those with compact feature distributions across tasks, such as genuine audio, and those with more spread-out distributions, like various types of fake audio. These distinctions are quantified by means of the in-class cosine distance, which subsequently serves as the basis for RWM to introduce a trainable gradient modification direction for distinct data types. Experimental evaluations against mainstream continual learning methods reveal the superiority of RWM in terms of knowledge acquisition and mitigating forgetting in audio deepfake detection. Furthermore, RWM's applicability extends beyond audio deepfake detection, demonstrating its potential significance in diverse machine learning domains such as image recognition.

### For training experiment, run 
```
  python train.sh --yaml
```
An example would be: 

```
  python train.sh clear10/clear10_feature_resnet50_moco.yaml
```
### For parsing metric, run
```
python parse_log_to_result.py --split --verbose[to print out the result matrix as well] --move[move to main server to plot] 
```
An example would be: 

```
python parse_log_to_result.py --split clear100_feature_resnet50_moco --verbose 1 --move 1
```
### For plotting the result matrix, like one in our paper, first need to specify --move 1 in running parse_log_to_result.py, and then run
```
python get_metric_all.py --plot 1
```
The code is based on the official code of the CLEAR benchmark. This project include the RAWM mentioned in our paper that (re)produced by us.


