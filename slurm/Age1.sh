#!/usr/bin/env bash

JOBNAME="main3D_baseline"
OUTFN="/pylon5/ac5616p/yukeyi/main3D.log"
ERRFN="/pylon5/ac5616p/yukeyi/main3Derror.log"
sbatch -o $OUTFN -e $ERRFN --job-name $JOBNAME /pylon5/ac5616p/Data/HeartSegmentationProject/CAP_challenge/CAP_challenge_training_set/test2/slurmLauncherGPU_AI.sh python main3D.py