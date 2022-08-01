# MaskSurf

## Masked Surfel Prediction for Self-Supervised Point Cloud Learning, [arxiv](https://arxiv.org/pdf/2207.03111.pdf)

[comment]: <> ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/masked-autoencoders-for-point-cloud-self/3d-point-cloud-classification-on-scanobjectnn&#41;]&#40;https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=masked-autoencoders-for-point-cloud-self&#41;)
[comment]: <> ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/masked-autoencoders-for-point-cloud-self/3d-point-cloud-classification-on-modelnet40&#41;]&#40;https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=masked-autoencoders-for-point-cloud-self&#41;)

Masked auto-encoding is a popular and effective self-supervised learning approach to point cloud learning. However, most of the existing methods reconstruct only the masked points and overlook the local geometry information, which is also important to understand the point cloud data. 
In this work, we make the first attempt, to the best of our knowledge, to consider the local geometry information explicitly into the masked auto-encoding, and propose a novel Masked Surfel Prediction (MaskSurf) method. Specifically, given the input point cloud masked at a high ratio, we learn a transformer-based encoder-decoder network to estimate the underlying masked surfels by simultaneously predicting the surfel positions (i.e., points) and per-surfel orientations (i.e., normals). The predictions of points and normals are supervised by the Chamfer Distance and a newly introduced Position-Indexed Normal Distance in a set-to-set manner. Our MaskSurf is validated on six downstream tasks under three fine-tuning strategies. In particular, MaskSurf outperforms its closest competitor, Point-MAE, by 1.2\% on the real-world dataset of ScanObjectNN under the OBJ-BG setting, justifying the advantages of masked surfel prediction over masked point cloud reconstruction. 


| ![./figure/net.png](./figure/net.png) |
|:-------------:|
| Fig.1: The overall framework of MaskSurf. |

## 1. Requirements
PyTorch >= 1.7.0;
python >= 3.7;
CUDA >= 9.0;
GCC >= 4.9;
torchvision;

```
pip install -r requirements.txt
```

```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## 2. Datasets

We use ShapeNet, ScanObjectNN, ModelNet40 and ShapeNetPart in this work. See [DATASET.md](./DATASET.md) for details.

## 3. MaskSurf Models

Pre-trained models will be provided here. 

[comment]: <> (|  Task | Dataset | Config | Acc.| Download|      )

[comment]: <> (|  ----- | ----- |-----|  -----| -----|)

[comment]: <> (|  Pre-training | ShapeNet |[pretrain.yaml]&#40;./cfgs/pretrain.yaml&#41;| N.A. | To add |)

[comment]: <> (|  Classification | ScanObjectNN |[finetune_scan_hardest.yaml]&#40;./cfgs/finetune_scan_hardest.yaml&#41;| 85.18%| &#41;  |)

[comment]: <> (|  Classification | ScanObjectNN |[finetune_scan_objbg.yaml]&#40;./cfgs/finetune_scan_objbg.yaml&#41;|90.02% | [here]&#40;https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_objbg.pth&#41; |)

[comment]: <> (|  Classification | ScanObjectNN |[finetune_scan_objonly.yaml]&#40;./cfgs/finetune_scan_objonly.yaml&#41;| 88.29%| [here]&#40;https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_objonly.pth&#41; |)

[comment]: <> (|  Classification | ModelNet40&#40;1k&#41; |[finetune_modelnet.yaml]&#40;./cfgs/finetune_modelnet.yaml&#41;| 93.80%| [here]&#40;https://github.com/Pang-Yatian/Point-MAE/releases/download/main/modelnet_1k.pth&#41; |)

[comment]: <> (|  Classification | ModelNet40&#40;8k&#41; |[finetune_modelnet_8k.yaml]&#40;./cfgs/finetune_modelnet_8k.yaml&#41;| 94.04%| [here]&#40;https://github.com/Pang-Yatian/Point-MAE/releases/download/main/modelnet_8k.pth&#41; |)

[comment]: <> (| Part segmentation| ShapeNetPart| [segmentation]&#40;./segmentation&#41;| 86.1% mIoU| [here]&#40;https://github.com/Pang-Yatian/Point-MAE/releases/download/main/part_seg.pth&#41; |)

[comment]: <> (|  Task | Dataset | Config | 5w10s Acc. &#40;%&#41;| 5w20s Acc. &#40;%&#41;| 10w10s Acc. &#40;%&#41;| 10w20s Acc. &#40;%&#41;|     )

[comment]: <> (|  ----- | ----- |-----|  -----| -----|-----|-----|)

[comment]: <> (|  Few-shot learning | ModelNet40 |[fewshot.yaml]&#40;./cfgs/fewshot.yaml&#41;| 96.3 ± 2.5| 97.8 ± 1.8| 92.6 ± 4.1| 95.0 ± 3.0| )

## 4. Running
We provide all the scripts for pre-training and fine-tuning in the [run.sh](./run.sh). 
Additionally, we provide a simple tool to collect the mean and standard deviation of results, for example: ```python parse_test_res.py ./experiments/{experiments_settting}/cfgs/ --multi-exp```

### MaskSurf Pre-training
To pretrain MaskSurf on ShapeNet training set, run the following command. If you want to try different models or masking ratios etc., first create a new config file, and pass its path to --config.

```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/pretrain_MaskSurf.yaml --exp_name <output_file_name>
```
### MaskSurf Fine-tuning

Fine-tuning on ScanObjectNN, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_scan_hardest_{protocol}.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Fine-tuning on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_modelnet_{protocol}.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Voting on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --test --config cfgs/finetune_modelnet_{protocol}.yaml \
--exp_name <output_file_name> --ckpts <path/to/best/fine-tuned/model>
```
Few-shot learning on ModelNet40 or ScanObjectNN, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/fewshot_{dataset}_{protocol}.yaml --finetune_model \
--ckpts <path/to/pre-trained/model> --exp_name <output_file_name> --way <5 or 10> --shot <10 or 20> --fold <0-9>
```
Domain generalization, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/dg_{source}_{protocol}.yaml --finetune_model --exp_name <output_file_name> --ckpts <path/to/best/fine-tuned/model>
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/dg_{source}2scannet_{protocol}.yaml --test --finetune_model --exp_name <output_file_name> --ckpts <./experiments/dg_{source}_{protocol}.yaml/cfgs/<path/to/best/fine-tuned/model>
```
Part segmentation on ShapeNetPart, run:
```
cd segmentation
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --ckpts <path/to/pre-trained/model> --root path/to/data --learning_rate 0.0002 --epoch 300
```
Semantic segmentation on S3DIS, run:
```
cd segmentation
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --optimizer_part all --ckpts <path/to/pre-trained/model> --root path/to/data --learning_rate 0.0002 
CUDA_VISIBLE_DEVICES=<GPUs> python main_test.py  --root path/to/data --visual  --ckpts <path/to/best/fine-tuned/model>
```

## 5. Visualization

Guidelines about visualization will be available here. 

[comment]: <> (Visulization of pre-trained model on ShapeNet validation set, run:)

[comment]: <> (```)

[comment]: <> (python main_vis.py --test --ckpts <path/to/pre-trained/model> --config cfgs/pretrain.yaml --exp_name <name>)

[comment]: <> (```)

[comment]: <> (<div  align="center">    )

[comment]: <> ( <img src="./figure/vvv.jpg" width = "900"  align=center />)

[comment]: <> (</div>)

## Acknowledgements

Our codes are built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE)

## Reference

```
@article{zhang2022masked,
  title={Masked Surfel Prediction for Self-Supervised Point Cloud Learning},
  author={Zhang, Yabin and Lin, Jiehong and He, Chenhang and Chen, Yongwei and Jia, Kui and Zhang, Lei},
  journal={arXiv preprint arXiv:2207.03111},
  year={2022}
}
```
