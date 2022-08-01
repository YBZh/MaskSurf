#!/bin/bash

YAML=pretrain_MaskSurf
# pre-training
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/${YAML}.yaml --exp_name log
# fine-tuning on the ScanObjectNN dataset.
for random in $(seq 1 3)
do
    # Transfering features protocol
    CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_hardest_transferring_features.yaml --finetune_model --exp_name ${YAML} --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth
    CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_objbg_transferring_features.yaml --finetune_model --exp_name ${YAML} --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth
    CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_objonly_transferring_features.yaml --finetune_model --exp_name ${YAML} --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth
    # Linear classification protocol
    CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_hardest_linear_classification.yaml --finetune_model --exp_name ${YAML} --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth
    CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_objbg_linear_classification.yaml --finetune_model --exp_name ${YAML} --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth
    CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_objonly_linear_classification.yaml --finetune_model --exp_name ${YAML} --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth
    # Non-linear classification protocol
    CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_hardest_non_linear_classification.yaml --finetune_model --exp_name ${YAML} --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth
    CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_objbg_non_linear_classification.yaml --finetune_model --exp_name ${YAML} --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth
    CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_objonly_non_linear_classification.yaml --finetune_model --exp_name ${YAML} --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth
done
# collect results on the 'finetune_scan_objonly_transferring_features' setting. NOTE that the parse_test_res.py is based on the `dassl' library.
python parse_test_res.py ./experiments/finetune_scan_objonly_transferring_features/cfgs/ --multi-exp



# fine-tuning on the ModelNet40 dataset.
for random in $(seq 1 3)
do
    # Transfering features protocol
    CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_modelnet_transferring_features.yaml --finetune_model --exp_name ${YAML} --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth
    CUDA_VISIBLE_DEVICES=0 python main.py --test --config cfgs/finetune_modelnet_transferring_features.yaml --exp_name ${YAML} --ckpts ./experiments/finetune_modelnet_transferring_features/cfgs/${YAML}/ckpt-best.pth
    # Linear classification protocol
    CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_modelnet_linear_classification.yaml --finetune_model --exp_name ${YAML} --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth
    # Non-linear classification protocol
    CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_modelnet_non_linear_classification.yaml --finetune_model --exp_name ${YAML} --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth
done
python parse_test_res.py ./experiments/finetune_modelnet_non_linear_classification/cfgs/ --multi-exp


# Few-shot on ScanobjectNN
for WAY in 5 10
do
  for SHOT in 10 20
  do
    for FOLD in $(seq 0 9)
    do
      CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fewshot_scanobjectnn_transferring_features.yaml --finetune_model \
      --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth --exp_name ${YAML} --way ${WAY} --shot ${SHOT} --fold ${FOLD}
      CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fewshot_scanobjectnn_linear_classification.yaml --finetune_model \
      --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth --exp_name ${YAML} --way ${WAY} --shot ${SHOT} --fold ${FOLD}
      CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fewshot_scanobjectnn_non_linear_classification.yaml --finetune_model \
      --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth --exp_name ${YAML} --way ${WAY} --shot ${SHOT} --fold ${FOLD}
    done
  done
done
python parse_test_res.py ./experiments/fewshot_scanobjectnn_transferring_features/cfgs/ --multi-exp --few-shot

# Domain generalization.
for random in $(seq 1 3)
do
     # modelnet 2 scannet
     CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/dg_modelnet_transferring_features.yaml --finetune_model --exp_name ${YAML} --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth
     CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/dg_modelnet2scannet_transferring_features.yaml --test --finetune_model --exp_name ${YAML} --ckpts ./experiments/dg_modelnet_transferring_features/cfgs/${YAML}/ckpt-best.pth
     # shapenet 2 scannet
     CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/dg_shapenet_transferring_features.yaml --finetune_model --exp_name ${YAML} --ckpts ./experiments/${YAML}/cfgs/log/ckpt-last.pth
     CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/dg_shapenet2scannet_transferring_features.yaml --test --finetune_model --exp_name ${YAML} --ckpts ./experiments/dg_shapenet_transferring_features/cfgs/${YAML}/ckpt-best.pth
     # results of other protocols could be similarly achieved.
done
python parse_test_res.py ./experiments/dg_modelnet2scannet_transferring_features/cfgs/ --multi-exp --few-shot


# part segmentation
cd segmentation
for random in $(seq 1 1) # results of multiple runs are similar.
do
    CUDA_VISIBLE_DEVICES=0 python main.py --optimizer_part only_new --log_dir ${YAML}_only_new --ckpts ../experiments/${YAML}/cfgs/log/ckpt-last.pth --root ../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/ --learning_rate 0.0002 --epoch 300
    CUDA_VISIBLE_DEVICES=0 python main.py --optimizer_part all --log_dir ${YAML}_all --ckpts ../experiments/${YAML}/cfgs/log/ckpt-last.pth --root ../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/ --learning_rate 0.0002 --epoch 300
done


# semantic segmentation
cd semantic_segmentation
for random in $(seq 1 1)
do
    CUDA_VISIBLE_DEVICES=0 python main.py --optimizer_part only_new --log_dir ${YAML}_only_new --ckpts ../experiments/${YAML}/cfgs/log/ckpt-last.pth --root ../data/stanford_indoor3d/ --learning_rate 0.0002
    CUDA_VISIBLE_DEVICES=0 python main_test.py --gpu 0 --log_dir ${YAML}_only_new --root ../data/stanford_indoor3d/  --ckpts ./log/semantic_seg/${YAML}_only_new/checkpoints/best_model.pth
    # Transfer features protocol
    CUDA_VISIBLE_DEVICES=0 python main.py --optimizer_part all --log_dir ${YAML}_all --ckpts ../experiments/${YAML}/cfgs/log/ckpt-last.pth --root ../data/stanford_indoor3d/ --learning_rate 0.0002
    CUDA_VISIBLE_DEVICES=0 python main_test.py --gpu 0 --log_dir ${YAML}_all --root ../data/stanford_indoor3d/  --ckpts ./log/semantic_seg/${YAML}_all/checkpoints/best_model.pth
done

