import torch
from model.openllama import OpenLLAMAPEFTModel
import cv2
import numpy as np
from PIL import Image as PILImage
from matplotlib import pyplot as plt


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
    'lora_dropout': 0.1
}

model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()



feature_cluster_path = './ckpt/train_supervised/feature_clusters.pth'
model.feature_cluster.save_path = feature_cluster_path  # 设置正确的 save_path
model.feature_cluster.load_centroids(device="cuda")  #
print(f'[INFO] 成功加载测试用聚类中心: {feature_cluster_path}')

def process_image_output(pixel_output, image_path):

    plt.imshow(pixel_output.to(torch.float16).reshape(224, 224).detach().cpu(), cmap='binary_r')
    plt.axis('off')
    plt.savefig('output.png', bbox_inches='tight', pad_inches=0)

    target_size = 224
    original_width, original_height = PILImage.open(image_path).size
    if original_width > original_height:
        new_width = target_size
        new_height = int(target_size * (original_height / original_width))
    else:
        new_height = target_size
        new_width = int(target_size * (original_width / original_height))

    new_image = PILImage.new('L', (target_size, target_size), 255)
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2

    pixel_output = PILImage.open('output.png').resize((new_width, new_height), PILImage.LANCZOS)
    new_image.paste(pixel_output, (paste_x, paste_y))
    new_image.save('output.png')

    image = cv2.imread('output.png', cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    cv2.imwrite('output.png', eroded_image)

    return 'output.png'

def predict(input_text, image_path, history, modality_cache):

    if not image_path:
        return "There is no image path provided! Please provide an image path.", None, history, modality_cache


    inputs = {
        'prompt': input_text,
        'image_paths': [image_path],
        'normal_img_paths': [],
        'audio_paths': [],
        'video_paths': [],
        'thermal_paths': [],
        'top_p': 0.01,
        'temperature': 1.0,
        'max_tgt_len': 512,
        'modality_embeds': modality_cache
    }

    response, pixel_output = model.generate(inputs, web_demo=True)

    output_image_path = None
    if pixel_output is not None:
        output_image_path = process_image_output(pixel_output, image_path)


    history.append((input_text, response))
    return response, output_image_path, history, modality_cache

def main():
    print("Welcome to AnomalyGPT Command Line Interface")
    print("Enter 'quit' to exit the program")
    print("Enter 'clear' to clear history")
    print("===========================================")

    history = []
    modality_cache = []


    image_path = input("Please enter the path to your query image: ").strip()

    while True:
        print("\nEnter your question or command:")
        user_input = input("> ").strip()

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        if user_input.lower() == 'clear':
            history = []
            modality_cache = []
            print("History cleared!")
            continue


        response, output_image_path, history, modality_cache = predict(
            user_input, image_path, history, modality_cache
        )


        print("\nResponse:")
        print(response)

        if output_image_path:
            print(f"Image output saved to: {output_image_path}")

if __name__ == "__main__":
    main()
