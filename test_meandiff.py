import requests

# Valores posibles para layer_type:
# - "mlp_output" - Salida del bloque MLP
# - "attention_output" - Salida del mecanismo de atención
# - "gate_proj" - Salida de la proyección gate en GLU
# - "up_proj" - Salida de la proyección up en GLU
# - "down_proj" - Salida de la proyección down en GLU
# - "input_norm" - Salida de la normalización de entrada

payload = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "prompt_pair": [
        "The white doctor examined the patient. The nurse thought",
        "The Black doctor examined the patient. The nurse thought"
    ],
    "layer_type": "gate_proj",  # Cambiado de layer_key a layer_type
    "figure_format": "png"
}

resp = requests.post("http://127.0.0.1:8000/visualize/mean-diff", json=payload)
resp.raise_for_status()  # Lanza error si no 200 OK

with open("mean-diff_python.png", "wb") as f:
    f.write(resp.content)

print("✅ Imagen guardada como mean-diff_python.png")
