import http.server
import json
import socketserver
from KNN import Similarity
from Vectors import stringToRGB
import cgi
import logging


class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    @staticmethod
    def compute(img):
        global Sim
        code = 200
        vector = Sim.vect.img_to_vec(img).tolist()
        json_data = json.dumps({"ImgVector": vector})
        return json_data

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

            logging.info('Img to Vec: computation started.')
            json_data = self.compute(img)
            logging.info('Img to Vec: done.')
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
Sim = Similarity()
# Create an object of the above class
handler_object = MyHttpRequestHandler

PORT = 8000
my_server = socketserver.TCPServer(("", PORT), handler_object)

logging.info('The server is ready to work.')

# Start the server
my_server.serve_forever()
