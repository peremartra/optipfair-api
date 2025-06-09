import requests

# Possible values for layer_type:
# - "mlp_output" - Output of the MLP block
# - "attention_output" - Output of the attention mechanism
# - "gate_proj" - Output of the gate projection in GLU
# - "up_proj" - Output of the up projection in GLU
# - "down_proj" - Output of the down projection in GLU
# - "input_norm" - Output of the input normalization

payload = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "prompt_pair": [
        "The white doctor examined the patient. The nurse thought",
        "The Black doctor examined the patient. The nurse thought"
    ],
    "layer_type": "gate_proj",  # Changed from layer_key to layer_type
    "figure_format": "png"
}

resp = requests.post("http://127.0.0.1:8000/visualize/mean-diff", json=payload)
resp.raise_for_status()  # Raises error if not 200 OK

with open("mean-diff_python.png", "wb") as f:
    f.write(resp.content)

print("✅ Image saved as mean-diff_python.png")