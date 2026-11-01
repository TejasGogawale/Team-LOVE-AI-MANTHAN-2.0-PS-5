"""Download only inference assets; do not upgrade the existing environment."""
from pathlib import Path
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "models" / "relateanything"
    snapshot_download("maelic/relsgg-vits16plus", revision="2db90096be5217bdc7a9003c042950f45723d105", local_dir=str(target),
        allow_patterns=["model.pth", "text_student.pt", "tokenizer*", "vocab.json",
                        "merges.txt", "special_tokens_map.json", "calibration.json",
                        "predicate_embeddings.npz"])
    print(f"RelateAnything inference assets ready: {target}")
