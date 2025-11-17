"""
Device selection (GPU, CPU...)
"""

__all__ = [
    "get_best_device"
]

def get_best_device():
    """
    Détecte le meilleur périphérique disponible :
    - CUDA (GPU NVIDIA)
    - DirectML (GPU AMD / Intel)
    - CPU sinon
    """
    if torch.cuda.is_available():
        device, backend_name = torch.device("cuda"), "CUDA"
    elif HAS_DIRECTML:
        device, backend_name = torch_directml.device(), "DirectML"
    else:
        device, backend_name = torch.device("cpu"), "CPU"

    # Informations système
    print("========================================")
    print(f"Backend sélectionné : {backend_name}")
    print(f"Périphérique utilisé : {device}")
    if backend_name == "CUDA":
        print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} Go")
    print(f"RAM système disponible : {psutil.virtual_memory().available / 1e9:.2f} Go")
    print("========================================")

    # AMP GradScaler — activé uniquement si GPU compatible
    scaler = GradScaler("cuda" if backend_name in ["CUDA", "DirectML"] else None)

    return device, backend_name, scaler