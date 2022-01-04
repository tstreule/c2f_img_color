# Ensure that file exists / you're in the correct directory
FILE=main.py
[ -f /etc/resolv.conf ] || echo "$FILE does not exist."; return
# Activate environment
conda deactivate
conda activate dl-env
# Submit job
bsub -o "$(pwd)/logs.txt" -n 4 -W 4:00 \
-R "rusage[mem=6000,scratch=10000,ngpus_excl_p=8]" \
-R "select[gpu_model0==GeForceGTX1080Ti,gpu_mtotal0>=15000]" \
python -u "$FILE" --model pretrain --max_epochs 40 --gpus -1 --strategy ddp \
--data_dir ~/.fastai/data/coco_sample/train_sample --batch_size 32 --dataset_size -1
# Review job
clear; bbjobs; echo ""; ls; echo ""