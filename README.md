Catalogue:
1. Running demo
   1.1  Environment Installation
   Install the required packages:
   pip install -r requirements.txt
   1.2 prepare ImageBind,Vicuna, and PandaGPT Checkpoint:
   You can download the pre-trained ImageBind model using . 
2. train Model
   2.1 Data preparation
   You can download our dataset using .
   2.2 training
   To train this Model on our dataset, please run the following commands:
   cd ./code
   bash ./scripts/train_plant.sh
   If you want to directly use our trained weights you can unzip the ckpt file.
   
