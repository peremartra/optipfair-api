import requests

payload = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "prompt_pair": [
        "The white doctor examined the patient. The nurse thought",
        "The Black doctor examined the patient. The nurse thought"
    ],
    "layer_key": "attention_output_layer_2",
    "figure_format": "png"
}

resp = requests.post("http://127.0.0.1:8000/visualize/heatmap", json=payload)
resp.raise_for_status()  # Lanza error si no 200 OK

with open("heatmap_python.png", "wb") as f:
    f.write(resp.content)

print("✅ Imagen guardada como heatmap_python.png")
