import torch
from model.openllama import OpenLLAMAPEFTModel
from PIL import Image as PILImage
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import argparse
# 初始化模型参数
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': './ckpt/train_supervised/pytorch_model.pt',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 512,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1
}

# 加载模型
model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()

# **加载训练时保存的聚类中心**
feature_cluster_path = './ckpt/train_supervised/feature_clusters.pth'
model.feature_cluster.save_path = feature_cluster_path
model.feature_cluster.load_centroids(device="cuda")
print(f'[INFO] 成功加载测试用聚类中心: {feature_cluster_path}')


def predict_cli(image_path, class_name, max_length=512, top_p=0.01, temperature=1.0):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"查询图片 {image_path} 不存在！")

    print(f'[INFO] 处理图像: {image_path}')
    print(f'[INFO] 使用类别 {class_name} 的聚类中心进行异常检测')

    # **获取该类别的聚类中心**
    centroids = model.feature_cluster.get_centroids([class_name], layer_idx=0)[0]

    if centroids is None:
        raise ValueError(f"[ERROR] {class_name} 没有聚类中心，请检查训练时是否保存！")

    # **提取查询图片的 patch tokens**
    query_patch_tokens = model.encode_image_for_one_shot([image_path])

    # **计算相似度**
    sims = []
    for layer_idx in range(len(query_patch_tokens)):
        class_centroids = model.feature_cluster.get_centroids([class_name], layer_idx=layer_idx)[0]

        if class_centroids is not None:
            query_patch_tokens_reshaped = query_patch_tokens[layer_idx].view(1, 256, 1, 1280)
            centroids_reshaped = class_centroids.unsqueeze(0).unsqueeze(1)

            cosine_similarity_matrix = torch.nn.functional.cosine_similarity(query_patch_tokens_reshaped,
                                                                             centroids_reshaped, dim=-1)
            sim_max, _ = torch.max(cosine_similarity_matrix, dim=2)
            sims.append(sim_max)

    if sims:
        sim = torch.mean(torch.stack(sims, dim=0), dim=0).reshape(1, 1, 16, 16)
        sim = torch.nn.functional.interpolate(sim, size=224, mode='bilinear', align_corners=True)
        anomaly_map = 1 - sim

    anomaly_map_prompts = model.prompt_learner(anomaly_map)

    return anomaly_map, anomaly_map_prompts


def chat_loop(image_path, class_name, max_length=512, top_p=0.01, temperature=1.0):
    history = []
    anomaly_map, anomaly_map_prompts = predict_cli(image_path, class_name, max_length, top_p, temperature)

    while True:
        user_input = input("\n### Human: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("[INFO] 退出对话。")
            break

        prompt_text = "\n".join([f"### Human: {q}\n### Assistant: {a}" for q, a in history])
        prompt_text += f"\n### Human: {user_input}"

        batch_size = 1
        p_before = '### Human: <Img>'
        p_before_tokens = model.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(model.device)
        p_before_embeds = model.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)

        p_middle = '</Img> '
        p_middle_tokens = model.llama_tokenizer(p_middle, return_tensors="pt", add_special_tokens=False).to(model.device)
        p_middle_embeds = model.llama_model.model.model.embed_tokens(p_middle_tokens.input_ids).expand(batch_size, -1, -1)

        p_after_tokens = model.llama_tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').to(model.device)
        p_after_embeds = model.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

        bos = torch.ones([batch_size, 1], dtype=p_before_tokens.input_ids.dtype, device=p_before_tokens.input_ids.device)
        bos = bos * model.llama_tokenizer.bos_token_id
        bos_embeds = model.llama_model.model.model.embed_tokens(bos)

        inputs_embeds = torch.cat(
            [bos_embeds, p_before_embeds, p_middle_embeds, anomaly_map_prompts, p_after_embeds],
            dim=1)

        outputs = model.llama_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_length,
            top_p=top_p,
            temperature=temperature,
            do_sample=True,
            use_cache=True,
        )

        output_text = model.llama_tokenizer.decode(outputs[0][:-2], skip_special_tokens=True)
        print(f"### Assistant: {output_text}")

        history.append((user_input, output_text))


if __name__ == '__main__':
    # chat_loop("/path/to/image.png", "palm")

    parser = argparse.ArgumentParser(description="AnomalyGPT 命令行多轮对话模式")
    parser.add_argument("--image", type=str, required=True, help="查询图像路径")
    parser.add_argument("--class_name", type=str, required=True, help="类别名称")
    parser.add_argument("--max_length", type=int, default=512, help="最大输出长度")
    parser.add_argument("--top_p", type=float, default=0.01, help="Top P 采样参数")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度系数")

    args = parser.parse_args()
    chat_loop(args.image, args.class_name, args.max_length, args.top_p, args.temperature)