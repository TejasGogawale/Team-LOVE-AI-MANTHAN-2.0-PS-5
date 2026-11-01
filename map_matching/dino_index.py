from pathlib import Path
import cv2
import numpy as np
from PIL import Image

class DinoIndex:
    def __init__(self, paths):
        import torch
        import faiss
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device).eval()
        vectors, owners = [], []
        for idx, path in enumerate(paths):
            image = Image.open(path).convert('RGB')
            for angle in range(0,360,60):
                vectors.append(self.embed(image.rotate(angle, expand=True, fillcolor='white')))
                owners.append(idx)
        vectors = np.vstack(vectors)
        self.owners = owners
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def embed(self, image):
        with self.torch.no_grad():
            output = self.model(**self.processor(images=image, return_tensors='pt').to(self.device)).last_hidden_state[:,0]
            return self.torch.nn.functional.normalize(output, dim=1).cpu().numpy().astype('float32')

    def shortlist(self, query, count=20):
        scores, ids = self.index.search(self.embed(Image.fromarray(cv2.cvtColor(query,cv2.COLOR_BGR2RGB))), self.index.ntotal)
        unique = list(dict.fromkeys(self.owners[i] for i in ids[0] if i >= 0))
        return unique[:count]
