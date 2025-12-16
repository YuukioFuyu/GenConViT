import gradio as gr
import torch
import os
import urllib.request
from model.pred_func import df_face, pred_vid, real_or_fake

def download_models():
    weight_dir = 'weight'
    ed_url = 'https://huggingface.co/Deressa/GenConViT/resolve/main/genconvit_ed_inference.pth'
    vae_url = 'https://huggingface.co/Deressa/GenConViT/resolve/main/genconvit_vae_inference.pth'
    ed_path = os.path.join(weight_dir, 'genconvit_ed_inference.pth')
    vae_path = os.path.join(weight_dir, 'genconvit_vae_inference.pth')

    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    if not os.path.exists(ed_path):
        print("Downloading ED model weights...")
        urllib.request.urlretrieve(ed_url, ed_path)
        print("Download complete.")

    if not os.path.exists(vae_path):
        print("Downloading VAE model weights...")
        urllib.request.urlretrieve(vae_url, vae_path)
        print("Download complete.")

model = None

def load_model_once():
    global model
    if model is None:
        download_models()
        print("Loading GenConViT model...")
        ed_weight = 'genconvit_ed_inference'
        vae_weight = 'genconvit_vae_inference'
        model = load_genconvit(config, net='genconvit', ed_weight=ed_weight, vae_weight=vae_weight, fp16=False)
        print("Model loaded successfully.")

def detect_deepfake(video_path, num_frames):
    if video_path is None:
        return "Please upload a video file."

    try:
        print(f"Processing video: {video_path}")
        faces = df_face(video_path, num_frames)
        if len(faces) == 0:
            return "No faces were detected in the video. Please try another video."

        y, y_val = pred_vid(faces, model)
        label = real_or_fake(y)
        confidence = y_val if label == 'FAKE' else 1 - y_val
        return { "FAKE": confidence, "REAL": 1 - confidence }

    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred during processing. The video might be corrupted or in an unsupported format."

title = "GenConViT: Deepfake Video Detection"

iface = gr.Interface(
    fn=detect_deepfake,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Slider(1, 200, value=15, step=1, label="Number of Frames")
    ],
    outputs=gr.Label(num_top_classes=2, label="Prediction Result"),
    title=title,
    description=description
)

if __name__ == "__main__":
    iface.queue().launch()
