import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec_model.to(DEVICE)
wav2vec_model.eval()
print("Model successfully loaded")

def pool_embeddings(emb: np.ndarray) -> np.ndarray:
    mean = np.mean(emb, axis=0)
    std = np.std(emb, axis=0)
    return np.concatenate([mean, std])


def extract_wav2vec_features(waveform: np.ndarray, sr: int) -> np.ndarray:
    inputs = processor(
        waveform,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = wav2vec_model(
            inputs.input_values.to(DEVICE)
        )

    embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    pooled = pool_embeddings(embeddings)

    return np.nan_to_num(pooled).astype(np.float32)
