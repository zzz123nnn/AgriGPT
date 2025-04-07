import os
from model.openllama import OpenLLAMAPEFTModel
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
# paths
parser.add_argument("--few_shot", type=bool, default=True)
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--round", type=int, default=3)

# parser.add_argument("--save_dir", type=str, default="./pixel_outputs")  # Add this line
command_args = parser.parse_args()

describles = {}
describles[
    'apple'] = "This is a photo of a apple for anomaly detection, which should be round, without any damage, flaw, defect, scratch, hole or broken part."
# describles['cable'] = "This is a photo of three cables for anomaly detection, they are green, blue and grey, which cannot be missed or swapped and should be without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'corn'] = "This is a photo of a corn for anomaly detection, which should be black and orange, with print '500', without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'tomato'] = "This is a photo of tomato for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
# describles['grid'] = "This is a photo of grid for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'grape'] = "This is a photo of a grape for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
# describles['leather'] = "This is a photo of leather for anomaly detection, which should be brown and without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'palm'] = "This is a photo of a palm for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part, and shouldn't be fliped."
describles[
    'peach'] = "This is a photo of a peach for anomaly detection, which should be white, with print 'FF' and red patterns, without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'pepper'] = "This is a photo of a pepper for anomaly detection, which tail should be sharp, and without any damage, flaw, defect, scratch, hole or broken part."
# describles['tile'] = "This is a photo of tile for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
# describles['toothbrush'] = "This is a photo of a toothbrush for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'potato'] = "This is a photo of a potato for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
# describles['wood'] = "This is a photo of wood for anomaly detection, which should be brown with patterns, without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'strawberry'] = "This is a photo of a strawberry for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."

FEW_SHOT = command_args.few_shot

# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': './ckpt/train_supervised/pytorch_model.pt',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}

model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()

print(f'[!] init the 7b model over ...')

# # 从这开始
# # 加载 centroids.pkl
# centroids_path = os.path.join(os.path.dirname(args['anomalygpt_ckpt_path']), 'centroids.pkl')
# if hasattr(model, 'feature_cluster') and model.feature_cluster is not None:
#     model.feature_cluster.load_clusters(centroids_path, device=torch.device('cuda'), dtype=torch.half)
# else:
#     print(f'[!] Model does not have feature_cluster or feature_cluster is None.')
#
# # 从这结束

# **加载训练时保存的聚类中心**
feature_cluster_path = './ckpt/train_supervised/feature_clusters.pth'
model.feature_cluster.save_path = feature_cluster_path  # 设置正确的 save_path
model.feature_cluster.load_centroids(device="cuda")  #
print(f'[INFO] 成功加载测试用聚类中心: {feature_cluster_path}')

"""Override Chatbot.postprocess"""
p_auc_list = []
i_auc_list = []


def predict(
        input,
        image_path,
        normal_img_path,
        max_length,
        top_p,
        temperature,
        history,
        modality_cache,
):
    prompt_text = ''
    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}\n### Assistant: {a}\n###'
        else:
            prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
    if len(history) == 0:
        prompt_text += f'{input}'
    else:
        prompt_text += f' Human: {input}'

    response, pixel_output = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [],
        'video_paths': [],
        'thermal_paths': [],
        'normal_img_paths': normal_img_path if normal_img_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    })

    return response, pixel_output


input = "Is there any anomaly in the image?"
root_dir = '../data/mvtec_anomaly_detection'

mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

CLASS_NAMES = ['apple', 'corn', 'tomato', 'grape', 'palm', 'peach', 'pepper', 'potato', 'strawberry']


precision = []

#
# # Create save directory if it doesn't exist
# os.makedirs(command_args.save_dir, exist_ok=True)

for c_name in CLASS_NAMES:
    #
    # # Create class-specific directory
    # class_save_dir = os.path.join(command_args.save_dir, c_name)
    # os.makedirs(class_save_dir, exist_ok=True)

    normal_img_paths = [
        "../data/mvtec_anomaly_detection/" + c_name + "/train/good/" + str(command_args.round * 4).zfill(3) + ".png",
        "../data/mvtec_anomaly_detection/" + c_name + "/train/good/" + str(command_args.round * 4 + 1).zfill(
            3) + ".png",
        "../data/mvtec_anomaly_detection/" + c_name + "/train/good/" + str(command_args.round * 4 + 2).zfill(
            3) + ".png",
        "../data/mvtec_anomaly_detection/" + c_name + "/train/good/" + str(command_args.round * 4 + 3).zfill(
            3) + ".png"]
    normal_img_paths = normal_img_paths[:command_args.k_shot]
    right = 0
    wrong = 0
    p_pred = []
    p_label = []
    i_pred = []
    i_label = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if "test" in file_path and 'png' in file and c_name in file_path:
                if FEW_SHOT:
                    resp, anomaly_map = predict(describles[c_name] + ' ' + input, file_path, normal_img_paths, 512, 0.1,
                                                1.0, [], [])
                else:
                    resp, anomaly_map = predict(describles[c_name] + ' ' + input, file_path, [], 512, 0.1, 1.0, [], [])
                is_normal = 'good' in file_path.split('/')[-2]

                if is_normal:
                    img_mask = Image.fromarray(np.zeros((224, 224)), mode='L')
                else:
                    mask_path = file_path.replace('test', 'ground_truth')
                    mask_path = mask_path.replace('.png', '_mask.png')
                    img_mask = Image.open(mask_path).convert('L')

                img_mask = mask_transform(img_mask)
                img_mask[img_mask > 0.1], img_mask[img_mask <= 0.1] = 1, 0
                img_mask = img_mask.squeeze().reshape(224, 224).cpu().numpy()

                anomaly_map = anomaly_map.reshape(224, 224).detach().cpu().numpy()

                p_label.append(img_mask)
                p_pred.append(anomaly_map)

                i_label.append(1 if not is_normal else 0)
                i_pred.append(anomaly_map.max())

                position = []



                if 'good' not in file_path and 'Yes' in resp:
                    right += 1
                elif 'good' in file_path and 'No' in resp:
                    right += 1
                else:
                    wrong += 1

                # anomaly_map_np = anomaly_map
                # # Create visualization - only anomaly map
                # plt.figure(figsize=(5, 5))
                # plt.imshow(anomaly_map_np, cmap='jet')
                # plt.axis('off')
                #
                # # Save the figure
                # save_name = os.path.splitext(os.path.basename(file_path))[0] + '_anomaly.png'
                # save_path = os.path.join(class_save_dir, save_name)
                # plt.savefig(save_path, bbox_inches='tight', dpi=300,
                #                 facecolor='navy')  # Set background to navy blue
                # plt.close()
                # # Save pixel output
                # anomaly_map_np = anomaly_map
                # # Save pixel output as grayscale image
                # anomaly_img = Image.fromarray((anomaly_map_np * 255).astype(np.uint8))  # Convert to 8-bit grayscale
                # save_name = os.path.splitext(os.path.basename(file_path))[0] + '_pixel_output.png'
                # save_path = os.path.join(class_save_dir, save_name)
                # anomaly_img.save(save_path)
                #
                # # Continue with existing evaluation...
                # is_normal = 'good' in file_path.split('/')[-2]
                # img_mask_k = img_mask
                # if is_normal:
                #     img_mask_k = Image.fromarray(np.zeros((224, 224)), mode='L')
                # else:
                #     mask_path = file_path.replace('test', 'ground_truth')
                #     mask_path = mask_path.replace('.png', '_mask.png')
                #     img_mask_k = Image.open(mask_path).convert('L')
                # Continue with your existing evaluation code...
                # Rest of your code remains the same...

    p_pred = np.array(p_pred)
    p_label = np.array(p_label)

    i_pred = np.array(i_pred)
    i_label = np.array(i_label)

    p_auroc = round(roc_auc_score(p_label.ravel(), p_pred.ravel()) * 100, 2)
    i_auroc = round(roc_auc_score(i_label.ravel(), i_pred.ravel()) * 100, 2)

    p_auc_list.append(p_auroc)
    i_auc_list.append(i_auroc)
    precision.append(100 * right / (right + wrong))

    print(c_name, 'right:', right, 'wrong:', wrong)
    print(c_name, "i_AUROC:", i_auroc)
    print(c_name, "p_AUROC:", p_auroc)

print("i_AUROC:", torch.tensor(i_auc_list).mean())
print("p_AUROC:", torch.tensor(p_auc_list).mean())
print("precision:", torch.tensor(precision).mean())