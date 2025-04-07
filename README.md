# 1.Preparation:

## 1.1. Environment Installation:
* Install the required packages:
```yaml
pip install -r requirements.txt
```

## 1.2. Prepare ImageBind,Vicuna, and PandaGPT Checkpoint:
* You can download the pre-trained ImageBind model using [this form](https://drive.google.com/drive/folders/1Rg8uvWHxqzSseNGC68R4J42B9tEqdKR3?usp=drive_link).


# 2.Train Model:
## 2.1. Data preparation:
* You can download our dataset using [this form](https://drive.google.com/drive/folders/1lFr5x5wgomKf2dmQxDE2ylIe0Xc_Fc4t?usp=drive_link).
## 2.2 training
* To train this Model on our dataset, please run the following commands:
```yaml
cd ./code
bash ./scripts/train_plant.sh
```
If you want to directly use our trained weights you can unzip the ckpt file.
