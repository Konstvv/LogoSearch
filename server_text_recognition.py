import http.server
import json
import socketserver
from Vectors import Vectorize, stringToRGB
import cgi
import numpy as np
import logging
import keras


class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    @staticmethod
    def compute(img):
        global model_text
        arr = []
        img = Vectorize.reshape_resize(img)
        arr.append(img)
        arr = np.asarray(arr).astype('float32')
        pred = model_text.predict(arr)
        if_text = False
        if pred[0] > 0.95:
            if_text = True
        return if_text

    def do_GET(self):
        self._set_response()
        self.wfile.write(bytes(json.dumps({'GET request': 'ok'}), 'utf-8'))
        logging.info('GET response executed')
        return

    def do_POST(self):
        logging.info('POST request: execution started.')
        post_reqid = None
        json_data = json.dumps({"reqid": str(post_reqid), "code": str(500), "data": 'POST request was not made'})
        try:
            self._set_response()
            ctype, pdict = cgi.parse_header(self.headers['Content-Type'])

            length = int(self.headers.get('content-length'))
            payload_string = self.rfile.read(length).decode('utf-8')
            logging.info(payload_string)
            # if payload_string:
            #     message = json.loads(payload_string)
            # else:
            #     logging.exception('POST request: Empty input received.')
            #     return

            post_img = payload_string
            post_img = bytes(post_img, 'utf-8')
            img = stringToRGB(post_img)

            if_text = self.compute(img)
            json_data = json.dumps({"if_text": if_text})
        except Exception as exc:
            code = 500
            errormsg = 'An error occured: {}'.format(exc)  # 'Index does not exist in the database'
            json_data = json.dumps({"code": str(code), "data": errormsg})
            logging.error('An error occured: {}'.format(exc))
        finally:
            self.wfile.write(bytes(json_data, 'utf-8'))
            logging.info('POST request: json has been sent to the server.')
            logging.info('POST request finished.')


logging.basicConfig(level=logging.INFO)
model_text = keras.models.load_model('save_at_16.h5', compile=False)
# Create an object of the above class
handler_object = MyHttpRequestHandler

PORT = 8001
my_server = socketserver.TCPServer(("", PORT), handler_object)

logging.info('The server is ready to work.')

# Start the server
my_server.serve_forever()
