@echo off
setlocal enabledelayedexpansion

set DATA=%~1
if "%DATA%"=="" set DATA=D:/Fewshot-Fruit/archive/images/images

set SPLIT=%~2
if "%SPLIT%"=="" set SPLIT=D:/Fewshot-Fruit/test_split.json

set SEED=%~3
if "%SEED%"=="" set SEED=1

set SHOTS=%~4
if "%SHOTS%"=="" set SHOTS=16

set TRAINER=MaPLe
set DATASET=fruit
set CFG=vit_b16_c2_ep5_batch4_2ctx
set LOADEP=5

set COMMON_DIR=%DATASET%/shots_%SHOTS%/%TRAINER%/%CFG%/seed%SEED%
set DIR_TRAIN=output/base2new/train_base/%COMMON_DIR%
set DIR_TEST_BASE=output/base2new/test_base/%COMMON_DIR%
set DIR_TEST_NEW=output/base2new/test_new/%COMMON_DIR%

echo ==========================================================
echo   MaPLe Fruit Benchmark - Base-to-Novel Generalization
echo   Dataset path : %DATA%
echo   Split path   : %SPLIT%
echo   Seed         : %SEED% ^| Shots: %SHOTS%
echo ==========================================================

echo.
echo [Step 1/3] Training MaPLe on Base classes...
python train.py ^
    --root "%DATA%" ^
    --split-path "%SPLIT%" ^
    --seed %SEED% ^
    --trainer %TRAINER% ^
    --dataset-config-file configs/datasets/%DATASET%.yaml ^
    --config-file configs/trainers/%TRAINER%/%CFG%.yaml ^
    --output-dir "%DIR_TRAIN%" ^
    DATASET.NUM_SHOTS %SHOTS% ^
    DATASET.SUBSAMPLE_CLASSES base

echo.
echo [Step 2/3] Evaluating on Base classes...
python train.py ^
    --root "%DATA%" ^
    --split-path "%SPLIT%" ^
    --seed %SEED% ^
    --trainer %TRAINER% ^
    --dataset-config-file configs/datasets/%DATASET%.yaml ^
    --config-file configs/trainers/%TRAINER%/%CFG%.yaml ^
    --output-dir "%DIR_TEST_BASE%" ^
    --model-dir "%DIR_TRAIN%" ^
    --load-epoch %LOADEP% ^
    --eval-only ^
    DATASET.NUM_SHOTS %SHOTS% ^
    DATASET.SUBSAMPLE_CLASSES base

echo.
echo [Step 3/3] Evaluating zero-shot transfer on Novel classes...
python train.py ^
    --root "%DATA%" ^
    --split-path "%SPLIT%" ^
    --seed %SEED% ^
    --trainer %TRAINER% ^
    --dataset-config-file configs/datasets/%DATASET%.yaml ^
    --config-file configs/trainers/%TRAINER%/%CFG%.yaml ^
    --output-dir "%DIR_TEST_NEW%" ^
    --model-dir "%DIR_TRAIN%" ^
    --load-epoch %LOADEP% ^
    --eval-only ^
    DATASET.NUM_SHOTS %SHOTS% ^
    DATASET.SUBSAMPLE_CLASSES new

echo.
echo ==========================================================
echo   Completed!
echo ==========================================================
