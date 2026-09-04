#!/bin/bash

# Custom configurations
DATA=${1:-"D:/Fewshot-Fruit/archive/images/images"}
SPLIT=${2:-"D:/Fewshot-Fruit/test_split.json"}
SEED=${3:-1}
SHOTS=${4:-16}

TRAINER=MaPLe
DATASET=fruit
CFG=vit_b16_c2_ep5_batch4_2ctx
LOADEP=5

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
DIR_TRAIN=output/base2new/train_base/${COMMON_DIR}
DIR_TEST_BASE=output/base2new/test_base/${COMMON_DIR}
DIR_TEST_NEW=output/base2new/test_new/${COMMON_DIR}

echo "=========================================================="
echo "  MaPLe Fruit Benchmark - Base-to-Novel Generalization"
echo "  Dataset path : ${DATA}"
echo "  Split path   : ${SPLIT}"
echo "  Seed         : ${SEED} | Shots: ${SHOTS}"
echo "=========================================================="

# 1. Train on Base classes (14 classes)
echo ""
echo "[Step 1/3] Training MaPLe on Base classes..."
python train.py \
    --root "${DATA}" \
    --split-path "${SPLIT}" \
    --seed "${SEED}" \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir "${DIR_TRAIN}" \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

# 2. Evaluate on Base classes
echo ""
echo "[Step 2/3] Evaluating on Base classes..."
python train.py \
    --root "${DATA}" \
    --split-path "${SPLIT}" \
    --seed "${SEED}" \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir "${DIR_TEST_BASE}" \
    --model-dir "${DIR_TRAIN}" \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

# 3. Evaluate on Novel classes (5 classes)
echo ""
echo "[Step 3/3] Evaluating zero-shot transfer on Novel classes..."
python train.py \
    --root "${DATA}" \
    --split-path "${SPLIT}" \
    --seed "${SEED}" \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir "${DIR_TEST_NEW}" \
    --model-dir "${DIR_TRAIN}" \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES new

echo ""
echo "=========================================================="
echo "  Training & Base-to-Novel evaluation completed!"
echo "  Base results : ${DIR_TEST_BASE}"
echo "  Novel results: ${DIR_TEST_NEW}"
echo "=========================================================="
