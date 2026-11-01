"""Run with python -m unittest test_api -v."""
import io
import unittest
from PIL import Image
from app import app, engine, match_lock, allowed_origins
from localization import coordinates_from_filename
import cv2
import numpy as np

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def upload(self, raw, url='/api/v1/localize'):
        return self.client.post(url, data={'query': (io.BytesIO(raw), 'query.png')})

    def test_health_and_dataset(self):
        self.assertEqual(self.client.get('/health').status_code, 200)
        self.assertEqual(self.client.get('/api/v1/health').json['dataset_images'], len(engine.paths))
        images = self.client.get('/api/v1/dataset').json['images']
        self.assertEqual(len(images), len(engine.paths))
        self.assertIn('latitude', images[0]['coordinates'])

    def test_retrieval_metadata_and_visuals(self):
        raw = cv2.imencode('.png', engine.images[0])[1].tobytes()
        result = self.upload(raw, '/api/v1/localize?include_images=false')
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.json['accepted'])
        self.assertEqual(result.json['image'], engine.paths[0].name)
        expected = coordinates_from_filename(engine.paths[0].name)
        self.assertEqual(result.json['coordinates']['latitude'], expected['latitude'])
        self.assertEqual(len(result.json['polygon']), 4)
        self.assertNotIn('retrieved', result.json)
        visual = self.upload(raw)
        self.assertTrue(visual.json['localized'].startswith('data:image/png;base64,'))

    def test_invalid_requests(self):
        self.assertEqual(self.client.post('/api/v1/localize').status_code, 400)
        self.assertEqual(self.upload(b'not an image').status_code, 400)
        self.assertEqual(self.upload(b'x', '/api/v1/localize?include_images=invalid').status_code, 400)
        self.assertEqual(self.client.get('/api/v1/unknown').json['code'], 'not_found')

    def test_no_match(self):
        raw = cv2.imencode('.png', np.full((100,100,3),255,np.uint8))[1].tobytes()
        result = self.upload(raw)
        self.assertEqual(result.status_code, 200)
        self.assertFalse(result.json['accepted'])
        self.assertNotIn('coordinates', result.json)

    def test_dimensions_and_body_limit(self):
        buffer = io.BytesIO()
        Image.new('RGB', (2001,2000)).save(buffer, format='PNG')
        self.assertEqual(self.upload(buffer.getvalue()).status_code, 400)
        result = self.client.post('/api/v1/localize', data=b'x' * (16*1024*1024+1), content_type='application/octet-stream')
        self.assertEqual(result.status_code, 413)
        self.assertEqual(result.json['code'], 'request_entity_too_large')

    def test_busy(self):
        match_lock.acquire()
        try:
            result = self.client.post('/api/v1/localize')
            self.assertEqual(result.status_code, 503)
            self.assertEqual(result.headers['Retry-After'], '5')
            self.assertEqual(self.client.get('/health').status_code, 200)
        finally:
            match_lock.release()

    def test_cors_preflight(self):
        origin = 'https://test-dashboard.example'
        allowed_origins.add(origin)
        try:
            result = self.client.options('/api/v1/localize', headers={'Origin':origin,'Access-Control-Request-Method':'POST'})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.headers['Access-Control-Allow-Origin'], origin)
            denied = self.client.options('/api/v1/localize', headers={'Origin':'https://unknown.example'})
            self.assertNotIn('Access-Control-Allow-Origin', denied.headers)
        finally:
            allowed_origins.remove(origin)

if __name__ == '__main__':
    unittest.main()
