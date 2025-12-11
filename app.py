import gradio as gr
import torch
from model.pred_func import df_face
import os
import urllib.request

def download_models():
    """
    Placeholder for the weight model download function
    """
    pass

model = None

def load_model_once():
    global model
    if model is None:
        print("Model not loaded, preparing...")
        # While not containing the original model
        model = "placeholder_model"

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
