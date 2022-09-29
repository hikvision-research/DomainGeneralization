#!/bin/bash
pip install yacs
pip install gdown

cd ..


DATASET_ROOT=./datasets
DATASET=$1
NET=resnet50_ms_L23
NET_INIT=pretrain/resnet50.pth

ND=2
BATCH=128
INTRA_EPOCHS=150
I2_EPOCHS=20
DS_EPOCHS=30

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

for SEED in $(seq 1 5)
do
    # Domain-Specific Model Training
    for SOURCE in ${D1} ${D2} ${D3} ${D4}
    do
        python train.py \
            --root ${DATASET_ROOT} \
            --seed ${SEED} \
            --trainer IntraADRNet \
            --source-domains ${SOURCE} \
            --target-domains ${SOURCE} \
            --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
            --config-file configs/trainers/dg/mixstyle/${DATASET}.yaml \
            --output-dir output/${DATASET}/${NET}_batch${BATCH}_DomainSpec/${SOURCE}/seed${SEED} \
            MODEL.BACKBONE.NAME ${NET} \
            DATALOADER.TRAIN_X.SAMPLER RandomSampler \
            DATALOADER.TRAIN_X.N_DOMAIN ${ND} \
            DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH} \
            OPTIM.MAX_EPOCH ${DS_EPOCHS} \
            OPTIM.LR 0.008 \
            MODEL.INIT_WEIGHTS ${NET_INIT} \
            DATASET.NAME ${DATASET_NAME} \
            DATASET.ROOT ${DATASET_ROOT}
    done


    for SETUP in $(seq 1 4)
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

        # Intra-ADR Warm-uping
        python train.py \
        --root ${DATASET_ROOT} \
        --seed ${SEED} \
        --trainer IntraADRNet \
        --source-domains ${S1} ${S2} ${S3} \
        --target-domains ${T} \
        --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
        --config-file configs/trainers/dg/mixstyle/${DATASET}.yaml \
        --output-dir output/${DATASET}/IntraADRNet_${NET}_ndomain${ND}_batch${BATCH}/${T}/seed${SEED} \
        MODEL.BACKBONE.NAME ${NET} \
        DATALOADER.TRAIN_X.SAMPLER RandomSampler \
        DATALOADER.TRAIN_X.N_DOMAIN ${ND} \
        DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH} \
        OPTIM.MAX_EPOCH ${INTRA_EPOCHS} \
        OPTIM.LR 0.008 \
        MODEL.INIT_WEIGHTS ${NET_INIT} \
        DATASET.NAME ${DATASET_NAME} \
        DATASET.ROOT ${DATASET_ROOT}

        DS_WEIGHT1=output/${DATASET}/${NET}_batch${BATCH}_DomainSpec/${S1}/seed${SEED}/model/model.pth.tar-${DS_EPOCHS}
        DS_WEIGHT2=output/${DATASET}/${NET}_batch${BATCH}_DomainSpec/${S2}/seed${SEED}/model/model.pth.tar-${DS_EPOCHS}
        DS_WEIGHT3=output/${DATASET}/${NET}_batch${BATCH}_DomainSpec/${S3}/seed${SEED}/model/model.pth.tar-${DS_EPOCHS}
        DA_WEIGHT=output/${DATASET}/IntraADRNet_${NET}_ndomain${ND}_batch${BATCH}/${T}/seed${SEED}/model/model.pth.tar-${INTRA_EPOCHS}

        # I2-ADR Training
        python train.py \
        --root ${DATASET_ROOT} \
        --seed ${SEED} \
        --trainer I2ADRNet \
        --source-domains ${S1} ${S2} ${S3} \
        --target-domains ${T} \
        --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
        --config-file configs/trainers/dg/mixstyle/${DATASET}.yaml \
        --output-dir output/${DATASET}/I2ADRNet_${NET}_ndomain${ND}_batch${BATCH}/${T}/seed${SEED} \
        MODEL.BACKBONE.NAME ${NET} \
        DATALOADER.TRAIN_X.SAMPLER RandomSampler \
        DATALOADER.TRAIN_X.N_DOMAIN ${ND} \
        DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH} \
        OPTIM.MAX_EPOCH ${I2_EPOCHS} \
        OPTIM.LR 0.0005 \
        MODEL.INIT_WEIGHTS ${DA_WEIGHT} \
        MODEL.TEACHER_WEIGHTS ${DS_WEIGHT1},${DS_WEIGHT2},${DS_WEIGHT3} \
        DATASET.NAME ${DATASET_NAME} \
        DATASET.ROOT ${DATASET_ROOT}
    done
done
