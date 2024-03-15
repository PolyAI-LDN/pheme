import argparse
import gradio as gr
import soundfile as sf
import numpy as np
import tempfile
from pathlib import Path
from transformer_infer import PhemeClient

def generate_audio(text, voice, temperature, top_k, max_new_tokens):
    args = parse_arguments()
    client = PhemeClient(args)
    audio_array = client.infer(text, voice=voice, temperature=temperature, top_k=top_k, max_new_tokens=max_new_tokens)
    
    # NumPy 配列をオーディオファイルに変換
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        sf.write(temp_file.name, audio_array, args.target_sample_rate)
        return temp_file.name

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", type=str, default="demo/manifest.json")
    parser.add_argument("--outputdir", type=str, default="demo/")
    parser.add_argument("--featuredir", type=str, default="demo/")
    parser.add_argument("--text_tokens_file", type=str, default="ckpt/unique_text_tokens.k2symbols")
    parser.add_argument("--t2s_path", type=str, default="ckpt/t2s/")
    parser.add_argument("--s2a_path", type=str, default="ckpt/s2a/s2a.ckpt")
    parser.add_argument("--target_sample_rate", type=int, default=16_000)
    return parser.parse_args()

voice_list = ["male_voice", "female_voice"]  # 利用可能な声のリストに置き換えてください

iface = gr.Interface(
    fn=generate_audio,
    inputs=[
        gr.inputs.Textbox(label="Text"),
        gr.inputs.Dropdown(voice_list, label="Voice"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=500, value=210, step=1, label="Top K"),
        gr.Slider(minimum=1, maximum=1000, value=750, step=1, label="Max New Tokens")
    ],
    outputs=gr.Audio(type="filepath", label="Generated Audio"),
    title="Pheme TTS Demo",
    description="Enter text and select a voice to generate speech.",
)

iface.launch(share=True)