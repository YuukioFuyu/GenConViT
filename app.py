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
