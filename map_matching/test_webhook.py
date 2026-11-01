import json
import unittest
from unittest.mock import patch, MagicMock
import app as module
import numpy as np

class WebhookTests(unittest.TestCase):
    def test_payload_and_header(self):
        image = np.zeros((100,150,3),np.uint8)
        result = dict(accepted=True, image='18_73.jpg', coordinates={'latitude':18,'longitude':73}, request_id='test', retrieved='large-image')
        with module.app.test_request_context('/api/v1/localize'), patch.object(module,'WEBHOOK_URL','https://example.com/webhook'), patch.object(module,'WEBHOOK_KEY','test-key'), patch.object(module.threading,'Thread') as thread, patch.object(module.urllib.request,'urlopen') as opening:
            module.notify(result,image,'upload')
            thread.assert_called_once()
            thread.call_args.kwargs['target']()
            outgoing = opening.call_args.args[0]
            body = json.loads(outgoing.data)
            self.assertEqual(body['result']['coordinates']['latitude'],18)
            self.assertNotIn('retrieved',body['result'])
            self.assertTrue(body['thumb'].startswith('data:image/jpeg;base64,'))
            self.assertEqual(outgoing.get_header('X-mesh-gateway-key'),'test-key')
            self.assertEqual(outgoing.method,'POST')

    def test_console_does_not_duplicate(self):
        with module.app.test_request_context('/',headers={'X-Indradhanu-Source':'console'}), patch.object(module,'WEBHOOK_URL','https://example.com/webhook'), patch.object(module.threading,'Thread') as thread:
            module.notify({},None,'upload')
            thread.assert_not_called()

    def test_disabled_by_default(self):
        with module.app.test_request_context('/'), patch.object(module,'WEBHOOK_URL',''), patch.object(module.threading,'Thread') as thread:
            module.notify({},None,'upload')
            thread.assert_not_called()

    def test_delivery_failure_is_contained(self):
        with module.app.test_request_context('/'), patch.object(module,'WEBHOOK_URL','https://example.com/webhook'), patch.object(module.threading,'Thread') as thread, patch.object(module.urllib.request,'urlopen',side_effect=OSError('offline')), patch.object(module.app.logger,'warning') as warning:
            module.notify({},np.zeros((20,20,3),np.uint8),'upload')
            thread.call_args.kwargs['target']()
            warning.assert_called_once()

if __name__ == '__main__':
    unittest.main()
