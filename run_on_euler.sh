# Ensure that file exists / you're in the correct directory
FILE=main.py
[ -f /etc/resolv.conf ] || echo "$FILE does not exist."; return
# Activate environment
conda deactivate
conda activate dl-env
# Submit job
bsub -o "$(pwd)/c2f_logs.txt" -n 20 -W 18:00 \
-R "rusage[mem=3000,scratch=1000,ngpus_excl_p=8]" \
-R "select[gpu_mtotal0>=15000]" \
python -u "$FILE" --model c2f --precision 16 --max_epochs 80 --gpus -1 --strategy ddp \
--data_dir ~/.fastai/data/coco_sample/train_sample --batch_size 32 --dataset_size -1 \
--gen_net_params 3 2 128 \
--pretrained_ckpt_path "$HOME/dl21_project/logs/c2f_pretrain/checkpoints/pretrain-epoch=24-val_mae_loss=0.0610.ckpt"
# Review job
clear; bbjobs; echo ""; ls; echo ""
