<h1>STAG4D: Spatial-Temporal Anchored Generative 4D Gaussians</h1>

<div>
    <a href='https://github.com/zeng-yifei?tab=repositories/' target='_blank'>Yifei Zeng</a><sup>1</sup>&emsp;
    <a href="https://github.com/yanqinJiang" target='_blank'>Yanqin Jiang*</a><sup>2</sup>&emsp;
    <a href="https://sites.google.com/site/zhusiyucs/home/" target='_blank'>Siyu Zhu</a><sup>3</sup>&emsp;
    <a href='https://github.com/YuanxunLu' target='_blank'>Yuanxun Lu</a><sup>1</sup>&emsp;
    <a href="https://linyou.github.io/">Youtian Lin</a><sup>1</sup>&emsp;
    <a href='https://zhuhao-nju.github.io/home/' target='_blank'>Hao Zhu</a><sup>1</sup>&emsp;
    <a href="https://people.ucas.ac.cn/~huweiming">Weiming Hu</a><sup>2</sup>&emsp;
    <a href='https://cite.nju.edu.cn/People/Faculty/20190621/i5054.html' target='_blank'>Xun Cao</a><sup>1</sup>&emsp;
    <a href='https://yoyo000.github.io/' target='_blank'>Yao Yao</a><sup>1+</sup>&emsp;
</div>
<div>
    <sup>1</sup>Nanjing University
    <sup>2</sup>CASIA
    <sup>3</sup>Fudan University
</div>
<div>
    <sup>*</sup>equal contribution
    <sup>+</sup>corresponding author
</div>

<h4 align="center">
  <a href="https://nju-3dv.github.io/projects/STAG4D/" target='_blank'>[Project Page]</a> •
</h4>

# Update

6.20: IMPORTANT. Fix the bug caused by new version of diff_gauss. Newest version of diff_gauss use `color, depth, norm, alpha, radii, extra` as an output. However, previous version use `color, depth, alpha, radii` as an output. Using older version of this code will cause mismatch error and may misuse normal for the alpha loss, resulting in bad results.

5.26: Update Text/Image to 4D data below.

5.21: Fix RGB loss into the batch loop. Add visualize code.


# ⚙️ Installation

```bash
pip install -r requirements.txt

git clone --recursive https://github.com/slothfulxtx/diff-gaussian-rasterization.git
pip install ./diff-gaussian-rasterization

pip install ./simple-knn
```

# Video-to-4D
To generate the examples in the project page, you can download the dataset from [google drive](https://drive.google.com/file/d/1YDvhBv6z5SByF_WaTQVzzL9qz3TyEm6a/view?usp=sharing). Place them in the dataset folder, and run:
```bash
python main.py --config configs/stag4d.yaml path=dataset/minions save_path=minions

#use --gui=True to turn on the visualizer (recommend)
python main.py --config configs/stag4d.yaml path=dataset/minions save_path=minions gui=True

```

To generate the spatial-temporal consistent data from stratch, your should place your rgba data in the form of 

```
├── dataset
│   | your_data 
│     ├── 0_rgba.png
│     ├── 1_rgba.png
│     ├── 2_rgba.png
│     ├── ...

```

and then run 
```bash
python scripts/gen_mv.py --path dataset/your_data --pipeline_path xxx/guidance/zero123pp

python main.py --config configs/stag4d.yaml path=data_path save_path=saving_path gui=True
```

To visualize the result, use you can replace the main.py with visualize.py, and the result will be saved to the valid/xxx path, e.g.:
```bash
python visualize.py --config configs/stag4d.yaml path=dataset/minions save_path=minions
```

<img src='assets/videoto4d.gif' height='60%'>

# Text-to-4D
For Text to 4D generation, we recommend using SDXL and SVD to generate a reasonable video. Then, after matting the video, use
the command above to generate a good 4D result. (This pipeline contains many independent parts and is kind of complex, so we may upload the whole workflow after integration if possible.)

If you want generate the examples in the paper, I also updated the corresponding data here in [google drive](https://drive.google.com/file/d/1EDNL7EBMR1vlfMOABdXjHzcKY7IXdcnj/view?usp=sharing). Remember to set size to 26 in config or use `size=26` in the command:

```bash
python main.py --config configs/stag4d.yaml path=dataset/xxx save_path=xxx size=26
```

<img src='assets/textto4d3.gif' height='60%'>

# Tips for better quality
If you want sacrifice time for better quality, here is some tips you can try to further improve the generated quality.

1, Use larger batch size.

2, Run for more steps.

## Citation
If you find our work useful for your research, please consider citing our paper as well as Consistent4D:
```
@article{zeng2024stag4d,
      title={STAG4D: Spatial-Temporal Anchored Generative 4D Gaussians}, 
      author={Yifei Zeng and Yanqin Jiang and Siyu Zhu and Yuanxun Lu and Youtian Lin and Hao Zhu and Weiming Hu and Xun Cao and Yao Yao},
      year={2024},
      eprint={2403.14939},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{jiang2023consistent4d,
      title={Consistent4D: Consistent 360{\deg} Dynamic Object Generation from Monocular Video}, 
      author={Yanqin Jiang and Li Zhang and Jin Gao and Weimin Hu and Yao Yao},
      year={2023},
      eprint={2311.02848},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgment
This repo is built on [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian) and [Zero123plus](https://github.com/SUDO-AI-3D/zero123plus). Thank all the authors for their great work.