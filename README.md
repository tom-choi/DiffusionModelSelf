# DiffusionModelSelf

1. I removed cached_download from the import line in the file /easy-diffusion/installer_files/env/lib/python3.8/site-packages/diffusers/utils/dynamic_modules_utils.py
2. self_impletement_1.ipynb could be a good starting point to test the code.

# Colab
- Example: https://colab.research.google.com/drive/1bVv6ugaBUHSRr-HY24cxkHc2b4t-QoBR?usp=sharing

# Job 
- DDPM 部分:
  - self_impletement_1.ipynb: 從mnist數據集(手寫數字)到採樣DDPM模型訓練，訓練到生成新的手寫數字。
  - self_impletement_2.ipynb: 從fashion數據集(衣服圖片)到採樣DDPM模型訓練，訓練到生成新的衣服圖片。
  - self_impletement_3.ipynb: 從自己的數據集(圖片)到採樣DDPM模型訓練，訓練到生成新的圖片。 (Not Done)

- SD 部分:
  - self_impletement_SD_1.ipynb: 拆整個Stable Diffusion的流程，試跑一段路程。
    - 如果本地跑不了，跑這個：https://colab.research.google.com/drive/1bVv6ugaBUHSRr-HY24cxkHc2b4t-QoBR?usp=sharing

# Reference
- https://github.com/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb