import json
from pathlib import Path

sources = [
('markdown', '# Image retrieval and patch localization\nDefault: exhaustive SIFT/RANSAC. Optional DINOv2 + multi-rotation FAISS shortlisting is available through Localizer(dino=True).'),
('code', 'from localization import Localizer, make_query\nimport numpy as np\nimport cv2\nfrom PIL import Image\nfrom IPython.display import display\nengine = Localizer(dino=False)\nprint(len(engine.paths), "images")'),
('code', 'rng = np.random.default_rng(42)\nidx = int(rng.integers(len(engine.paths)))\nquery, truth = make_query(engine.images[idx], rng)\nresult = engine.match(query)\nprint("Known source:", engine.paths[idx].name)\nprint("Transform:", truth)\nprint("Retrieved:", result.get("image"))\nprint("Correct source:", result.get("image") == engine.paths[idx].name)'),
('code', 'display(Image.fromarray(cv2.cvtColor(query, cv2.COLOR_BGR2RGB)))\nif result["accepted"]:\n    print("Source image coordinates:", result["coordinates"])\n    for key in ["retrieved", "localized", "aligned", "matches_image"]:\n        print(key)\n        display(Image.fromarray(cv2.cvtColor(result[key], cv2.COLOR_BGR2RGB)))'),
('markdown', 'For an uploaded query, set query = cv2.imread(str(your_image_path)) and call engine.match(query). The app.py interface handles uploads and random demonstrations.'),
('code', 'from evaluate import evaluate\nsummary = evaluate()')]
cells = []
for kind, source in sources:
    cell = dict(cell_type=kind, metadata={}, source=source.splitlines(True), id=f'cell-{len(cells)}')
    if kind == 'code':
        cell.update(outputs=[], execution_count=None)
    cells.append(cell)
Path('image_localization.ipynb').write_text(json.dumps(dict(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}, nbformat=4, nbformat_minor=5), indent=2))
