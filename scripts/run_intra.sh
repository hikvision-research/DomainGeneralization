#!/bin/bash
pip install yacs
pip install gdown

cd ..


DATASET_ROOT=./datasets
DATASET=$1
NET=resnet50_ms_L23
TRAINER=IntraADRNet

ND=2
BATCH=128

if [ ${DATASET} == pacs ]; then
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch

    DATASET_NAME='PACS'
    
elif [ ${DATASET} == office_home_dg ]; then
    D1=art
    D2=clipart
    D3=product
    D4=real_world
    
    DATASET_NAME='OfficeHomeDG'
    

fi

for SEED in $(seq 1 1)
do
    for SETUP in $(seq 1 1)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            T=${D4}
        fi
        python train.py \
        --root ${DATASET_ROOT} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --source-domains ${S1} ${S2} ${S3} \
        --target-domains ${T} \
        --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
        --config-file configs/trainers/dg/mixstyle/${DATASET}.yaml \
        --output-dir output/${DATASET}/${TRAINER}_${NET}_ndomain${ND}_batch${BATCH}/${T}/seed${SEED} \
        MODEL.BACKBONE.NAME ${NET} \
        DATALOADER.TRAIN_X.SAMPLER RandomSampler \
        DATALOADER.TRAIN_X.N_DOMAIN ${ND} \
        DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH} \
        OPTIM.MAX_EPOCH 150 \
        OPTIM.LR 0.008 \
        MODEL.INIT_WEIGHTS ./pretrain/resnet50.pth \
        DATASET.NAME ${DATASET_NAME} \
        DATASET.ROOT ${DATASET_ROOT}
    done
done