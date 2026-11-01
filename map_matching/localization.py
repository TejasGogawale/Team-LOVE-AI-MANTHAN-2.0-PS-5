"""Image retrieval and geometric localization in original pixel coordinates."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / '.packages'))
import cv2
import numpy as np

ROOT = Path(__file__).parent
DATASET = ROOT / 'VIT_IMAGES'

def coordinates_from_filename(filename):
    """Read source image metadata; this is not a patch GPS estimate."""
    try:
        stem = Path(filename).stem
        parts = stem.split('_') if '_' in stem else stem.split(',')
        latitude, longitude = map(float, parts)
        if -90 <= latitude <= 90 and -180 <= longitude <= 180:
            return dict(latitude=latitude, longitude=longitude)
    except (ValueError, TypeError):
        pass
    return None

class Localizer:
    def __init__(self, dataset=DATASET, dino=False):
        self.paths = sorted(p for p in Path(dataset).rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if not self.paths:
            raise ValueError('Dataset contains no images')
        self.sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02, edgeThreshold=15)
        self.images = [cv2.imread(str(p)) for p in self.paths]
        self.features = [self.extract(im) for im in self.images]
        self.dino = None
        if dino:
            from dino_index import DinoIndex
            self.dino = DinoIndex(self.paths)

    def extract(self, image):
        return self.sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)

    def match(self, query):
        if query is None or min(query.shape[:2]) < 16:
            raise ValueError('Please provide a readable image at least 16 pixels wide and high')
        kp, des = self.extract(query)
        if des is None or len(kp) < 8:
            return {'accepted': False, 'reason': 'Too little visual texture to locate this patch.', 'ranking': []}
        indices = self.dino.shortlist(query) if self.dino else list(range(len(self.paths)))
        results = []
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
        for idx in indices:
            ck, cd = self.features[idx]
            if cd is None or len(cd) < 2:
                continue
            pairs = matcher.knnMatch(des, cd, k=2)
            good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
            if len(good) < 8:
                continue
            src = np.float32([kp[m.queryIdx].pt for m in good])
            dst = np.float32([ck[m.trainIdx].pt for m in good])
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4, maxIters=5000, confidence=0.995)
            if H is None or mask is None:
                continue
            mask = mask.ravel().astype(bool)
            h, w = query.shape[:2]
            poly = cv2.perspectiveTransform(np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])[None], H)[0]
            ih, iw = self.images[idx].shape[:2]
            area = abs(cv2.contourArea(poly))
            valid = bool(np.isfinite(poly).all() and cv2.isContourConvex(poly) and 100 < area < iw*ih*1.3 and (poly[:,0] >= -iw*.15).all() and (poly[:,0] <= iw*1.15).all() and (poly[:,1] >= -ih*.15).all() and (poly[:,1] <= ih*1.15).all())
            projected = cv2.perspectiveTransform(src[None], H)[0]
            error = float(np.median(np.linalg.norm(projected[mask]-dst[mask], axis=1))) if mask.any() else 999
            coverage = abs(cv2.contourArea(cv2.convexHull(src[mask]))) / (w*h) if mask.sum() >= 3 else 0
            results.append(dict(idx=idx, image=self.paths[idx].name, inliers=int(mask.sum()), matches=len(good), ratio=float(mask.mean()), error=error, coverage=coverage, valid=valid, polygon=poly.tolist(), H=H, good=good, mask=mask))
        results.sort(key=lambda r: (r['valid'], r['inliers'], r['ratio']), reverse=True)
        ranking = [{k:r[k] for k in ('image','inliers','matches','ratio','error','coverage','valid')} for r in results]
        if not results:
            return dict(accepted=False, reason='No geometrically consistent match found.', ranking=[])
        best = results[0]
        accepted = best['valid'] and best['inliers'] >= 8 and best['ratio'] >= .25 and best['coverage'] >= .025 and best['error'] <= 4
        if not accepted:
            return dict(accepted=False, reason='Best candidate has insufficient geometric evidence.', ranking=ranking)
        full = self.images[best['idx']]
        overlay = full.copy()
        cv2.polylines(overlay, [np.int32(best['polygon'])], True, (40,230,40), 3, cv2.LINE_AA)
        aligned = cv2.warpPerspective(full, np.linalg.inv(best['H']), (query.shape[1], query.shape[0]))
        ck, _ = self.features[best['idx']]
        matches = cv2.drawMatches(query, kp, full, ck, best['good'], None, matchesMask=best['mask'].astype(int).tolist(), flags=2, matchColor=(40,230,40))
        return dict(accepted=True, image=best['image'], coordinates=coordinates_from_filename(best['image']), polygon=best['polygon'], inliers=best['inliers'], error=best['error'], ranking=ranking, retrieved=full, localized=overlay, aligned=aligned, matches_image=matches)

def make_query(image, rng, fraction=None, angle=None, scale=None):
    h,w = image.shape[:2]
    fraction = fraction if fraction is not None else float(rng.uniform(.25,.5))
    cw,ch = max(40,int(w*fraction)), max(40,int(h*fraction))
    x,y = int(rng.integers(0,w-cw+1)), int(rng.integers(0,h-ch+1))
    angle = float(rng.uniform(0,360)) if angle is None else angle
    scale = float(rng.uniform(.65,1.5)) if scale is None else scale
    crop = image[y:y+ch,x:x+cw]
    M = cv2.getRotationMatrix2D(((cw-1)/2,(ch-1)/2), angle, scale)
    nw = int(np.ceil(scale*(cw*abs(np.cos(np.deg2rad(angle)))+ch*abs(np.sin(np.deg2rad(angle))))))+2
    nh = int(np.ceil(scale*(cw*abs(np.sin(np.deg2rad(angle)))+ch*abs(np.cos(np.deg2rad(angle))))))+2
    M[:,2] += [(nw-cw)/2,(nh-ch)/2]
    query = cv2.warpAffine(crop, M, (nw,nh), borderValue=(255,255,255))
    transform = np.vstack([M,[0,0,1]]) @ np.array([[1,0,-x],[0,1,-y],[0,0,1]])
    corners = np.float32([[0,0],[nw-1,0],[nw-1,nh-1],[0,nh-1]])
    truth = cv2.perspectiveTransform(corners[None], np.linalg.inv(transform))[0]
    return query, dict(crop=[x,y,cw,ch], angle=angle, scale=scale, polygon=truth.tolist())
