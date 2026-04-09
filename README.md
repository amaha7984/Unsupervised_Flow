# Unsupervised_Flow
Flow-based Model For Unsupervised Domain Translation
- This project explored how we can use self-supervision to learn unpaired translation.
- We explored learning an embedding space using DINOv2 and Two-Layer MLP

## Training OT-CFM with DDP in Pixel Space
```
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
     train_ddp.py --model "otcfm" --lr 2e-4 --ema_decay 0.9999 \
    --pixel \
    --batch_size 64 --total_steps 160001 --save_step 40000 \
    --data_root "/path/to/dataset" \
    --output_dir ./path/to/saving/weight \
    --parallel True --master_port "xxxxx"
```

### Acknowledgments
Our code is developed based on [conditional-flow-matching](https://github.com/atong01/conditional-flow-matching) and reproduced under the MIT License. We would like to thanks the authors for the great work. 