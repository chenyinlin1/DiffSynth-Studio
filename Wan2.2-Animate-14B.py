import os
# 将模型缓存目录指向公共根目录，避免重复嵌套子目录导致找不到文件
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/mnt/shared-storage-gpfs2/solution-gpfs02/liugaofeng/Data/model_related"

import torch
from PIL import Image
from diffsynth.core import load_state_dict
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download, snapshot_download

examples_dir = "data/examples/wan/animate"
# 如果示例数据已存在则跳过下载，避免重复耗时
if not os.path.exists(examples_dir):
    dataset_snapshot_download(
        dataset_id="DiffSynth-Studio/examples_in_diffsynth",
        local_dir="./",
        allow_file_pattern=f"{examples_dir}/*",
    )
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    redirect_common_files=False,  # 使用本地原始路径，避免重定向到不存在的目录
)



# Animate
input_image = Image.open("data/examples/wan/animate/animate_input_image.png")
animate_pose_video = VideoData("data/examples/wan/animate/animate_pose_video.mp4").raw_data()[:81-4]
animate_face_video = VideoData("data/examples/wan/animate/animate_face_video.mp4").raw_data()[:81-4]
video = pipe(
    prompt="视频中的人在做动作",
    seed=0, tiled=True,
    input_image=input_image,
    animate_pose_video=animate_pose_video,
    animate_face_video=animate_face_video,
    num_frames=81, height=720, width=1280,
    num_inference_steps=20, cfg_scale=1,
)
save_video(video, "video_1_Wan2.2-Animate-14B.mp4", fps=15, quality=5)

# Replace
# snapshot_download("Wan-AI/Wan2.2-Animate-14B", allow_file_pattern="relighting_lora.ckpt", local_dir="models/Wan-AI/Wan2.2-Animate-14B")
# lora_state_dict = load_state_dict(os.path.join(os.environ["DIFFSYNTH_MODEL_BASE_PATH"], "Wan-AI/Wan2.2-Animate-14B/relighting_lora.ckpt"), torch_dtype=torch.bfloat16, device="cuda")["state_dict"]
# lora_state_dict = {i: lora_state_dict[i].to(torch.bfloat16) for i in lora_state_dict}
# pipe.load_lora(pipe.dit, state_dict=lora_state_dict)
# input_image = Image.open("data/examples/wan/animate/replace_input_image.png")
# animate_pose_video = VideoData("data/examples/wan/animate/replace_pose_video.mp4").raw_data()[:81-4]
# animate_face_video = VideoData("data/examples/wan/animate/replace_face_video.mp4").raw_data()[:81-4]
# animate_inpaint_video = VideoData("data/examples/wan/animate/replace_inpaint_video.mp4").raw_data()[:81-4]
# animate_mask_video = VideoData("data/examples/wan/animate/replace_mask_video.mp4").raw_data()[:81-4]
# video = pipe(
#     prompt="视频中的人在做动作",
#     seed=0, tiled=True,
#     input_image=input_image,
#     animate_pose_video=animate_pose_video,
#     animate_face_video=animate_face_video,
#     animate_inpaint_video=animate_inpaint_video,
#     animate_mask_video=animate_mask_video,
#     num_frames=81, height=720, width=1280,
#     num_inference_steps=20, cfg_scale=1,
# )
# save_video(video, "video_2_Wan2.2-Animate-14B.mp4", fps=15, quality=5)
