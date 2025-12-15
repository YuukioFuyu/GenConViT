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

def detect_deepfake(video, num_frames):
    print("Receiving video...")
    return {"REAL": 0.5, "FAKE": 0.5}

iface = gr.Interface(
    fn=detect_deepfake,
    inputs=[gr.Video(), gr.Slider(1, 30, value=15)],
    outputs=gr.Label()
)

if __name__ == "__main__":
    iface.launch()
