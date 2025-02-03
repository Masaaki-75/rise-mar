This is the official repository for the paper "Radiologist-in-the-Loop Self-Training for Generalizable CT Metal Artifact Reduction". [[arxiv](https://arxiv.org/abs/2501.15610), [tmi](https://ieeexplore.ieee.org/document/10857416)]

![](./figs/overview.png)

# Highlights
- A simple yet effective semi-supervised learning framework (termed RISE-MAR) that ensures high-quality pseudo groundtruths for clinical CT metal artifact reduction (MAR).
- A pretrained clinical quality assessor (termed CQA) model that scores the quality of a CT image with potential metal artifacts.

# TODO
- [x] Preprint version
- [x] Model implementation code
- [x] Training code
- [ ] CQA model weights


# Walkthrough
Implementing the full version of our method (RISE-MAR) involves several steps:
## 1. Data preparation
Download related CT datasets. Simulation may be required for metal artifact-affected data generation. 

For each paired dataset (i.e., each artifact-affected image is paired with an artifact-free groundtruth), generate the corresponding JSON file that stores the metadata (please check `./data/meta/example_paired_deepl.json` for an example). The JSON file should look like:
```jsonc
{
    "unique_name_for_case_1": {
        "metal_img": "path/to/the/artifact/affected/image",
        "li_img": "path/to/the/LI/corrected/image",
        "gt_img": "path/to/the/artifact/free/ground/truth",
        "metal_mask": "path/to/the/metal/mask",
        "root_dir": "path/prefix/for/the/above/paths/(optional)",
    },
    // ...
    "unique_name_for_case_n": {
        "metal_img": "path/to/the/artifact/affected/image",
        "li_img": "path/to/the/LI/corrected/image",
        "gt_img": "path/to/the/artifact/free/ground/truth",
        "metal_mask": "path/to/the/metal/mask",
        "root_dir": "path/prefix/for/the/above/paths/(optional)",
    },
}
```

For unpaired datasets (i.e., the artifact-affected images and artifact-free ones are unpaired), you will have to provide the quality scores for each image if you want to train a model for image quality assessment like our CQA model. 

Otherwise, simply gather the file paths for artifact-affected images and artifact-free ones separately to generate the JSON file as follows (please check `./data/meta/example_unpaired_deepl.json` for an example):
```jsonc
{
    "ma": {  // For unpaired metal artifact-affected images
        "unique_name_for_case_1": {
            "img": "/path/to/the/artifact/affected/image",
            "quality": "an integer for the quality score", // optional
            "root_dir": "path/prefix/(optional)"
        },
        // ...
        "unique_name_for_case_n": {
            "img": "/path/to/the/artifact/affected/image",
            "quality": "an integer for the quality score",
            "root_dir": "path/prefix/(optional)"
        },
    "mf": {  // For unpaired metal artifact-free images
        "unique_name_for_case_1": {
            "img": "/path/to/the/artifact/free/image",
            "quality": "an integer for the quality score", // optional
            "root_dir": "path/prefix/(optional)"
        },
        // ...
        "unique_name_for_case_n": {
            "img": "/path/to/the/artifact/free/image",
            "quality": "an integer for the quality score",
            "root_dir": "path/prefix/(optional)"
        },
  }
```

After finishing the data preparation, browse `./configs/__init__.py` and modify the paths for the metadata.


## 2. Baseline models training (recommended)
We refer to "baseline models" as models that produce imperfect prediction for clinical MAR. These models can be obtained via:
- Some conventional methods (e.g., LI, NMAR, ...)
- Insufficient training (e.g., early stop before model convergence)
- Models with limited domain transferability (e.g., can handle simulated metal artifacts very well but perform poor MAR on real data)
- ...

These models are actually still useful! In our work, they:
- serve as undertrained MAR models that provide CT images with low-to-moderate quality, <u>greatly enhancing the diversity of our clinical quality assessment dataset</u>.
- provide a better way of network parameters initialization. We observe that, for unsupervised or semisupervised learning, such initialization often leads to <u>a more stable training process</u>.
- serve as prior models, i.e., teacher network in the paper. A stronger prior model certainly perform better MAR, but with our RISE-MAR framework, we can still employ an imperfect one while expecting satisfactory MAR results ;) 
- ...

As an example of training script, please run:
```sh
bash scripts/train_supervised_mar.sh
```
After obtaining these models, specify the related paths (`UNDETRAINED_WEIGHTS` and `PRETRAINED_*`) in `./configs/__init__.py`.


## 3. CQA model training
Modify the arguments in `scripts/train_cqa.sh` and run that script:
```sh
bash scripts/train_cqa.sh
```

After the training, specify the path to the CQA model weights (`PRETRAINED_CQA_PATH`) in `./configs/__init__.py`.

We will also release pretrained weights for our CQA model soon. Please stay tuned!


## 4. MAR model training
Modify the arguments in `scripts/train_risemar.sh` and run that script:
```sh
bash scripts/train_risemar.sh
```


# Citation
If you find our work and code helpful, please kindly cite our paper :blue_heart:

```bibtex
@ARTICLE{clma2024risemar,
  author={Ma, Chenglong and Li, Zilong and Li, Yuanlin and Han, Jing and Zhang, Junping and Zhang, Yi and Liu, Jiannan and Shan, Hongming},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Radiologist-in-the-Loop Self-Training for Generalizable {CT} Metal Artifact Reduction}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2025.3535906}
}
```