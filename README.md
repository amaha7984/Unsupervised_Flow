# Self-Flow
Self-Flow: Learned Semantic Couplings for Cross-Domain Flow Matching Without Paired Supervision 

- The problem we are considering is translation of images from one domain to another domain with flow matching.
- The domain could be semantic relation (road structures), fidelity (realistic), and color relation (color grading).
- This project explores how we can use self-supervision to learn unpaired translation.
- The main idea is that a learned embedding can be used as task specific input-output pair selection for training flow matching model in unpaired setting.
- By unpaired setting meaning, we mean a scenario where there is no exact corresponding input and output images.
- We explored learning an embedding space using DINOv2 and Two-Layer MLP.

## Training OT-CFM ([conditional-flow-matching](https://github.com/atong01/conditional-flow-matching)) with DDP in Pixel Space
```
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
     train_ddp.py --model "otcfm" --lr 2e-4 --ema_decay 0.9999 \
    --pixel \
    --batch_size 64 --total_steps 160001 --save_step 40000 \
    --data_root "/path/to/dataset" \
    --output_dir ./path/to/saving/weight \
    --parallel True --master_port "xxxxx"
```

**Note:**
- To train independent coupling conditional flow matching, change --model "icfm"
- To train the proposed method, change --model "selfcfm"

## Datasets
### UIEB Dataset
- Paper: "An Underwater Image Enhancement Benchmark Dataset and Beyond"
- Link: https://li-chongyi.github.io/proj_benchmark.html
- Preparation of UIEB Dataset for Image-to-Image Translation
  - Download Images from the Link Above
  - Run: 
         ``` python3.11 preparing_UIEB_dataset.py
         ```
  - Divide dataset, so that train set have 800 images and test set have 90 images

### Cityscapes Dataset
- Paper: "The Cityscapes Dataset for Semantic Urban Scene Understanding"
- Preparation of Cityscapes dataset
   - Follow the information provided in "Unpaired image-to-image translation using cycle-consistent adversarial networks" at the experimental setups.

### Google Map (Aerial -> Photograph)
- Paper: "Unpaired image-to-image translation using cycle-consistent adversarial networks"
- Data can be obtained from the github: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix



## Acknowledgments
Our code is developed based on [conditional-flow-matching](https://github.com/atong01/conditional-flow-matching) and reproduced under the MIT License. We would like to thanks the authors for the great work. 