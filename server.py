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
    def compute(img, maxlen=5, reqid=None, test_mode=False):
        logging.info('Finding similar images: computation began.')
        global Sim
        code = 200
        values = Sim.ind_similar(img, n_neighbors=maxlen, test_mode=test_mode)
        logging.info('Finding similar images: computation done.')
        json_data = json.dumps({"reqid": reqid, "code": code, "data": values})
        return json_data

    def do_GET(self):
        self._set_response()

        # filename = 'home.html'
        # fin = open(filename)
        # html = fin.read()
        # fin.close()
        # # Writing the HTML contents with UTF-8
        # self.wfile.write(bytes(html, "utf8"))

        Sim.upload_data_ram()
        self.wfile.write(bytes(json.dumps({'GET request': 'ok', 'Data uploaded in RAM': 'ok'}), 'utf-8'))
        logging.info('GET response executed')
        return

    def do_POST(self):
        logging.info('POST request: execution started.')
        post_reqid = None
        json_data = json.dumps({"reqid": str(post_reqid), "code": str(500), "data": 'POST request was not made'})
        try:
            self._set_response()
            ctype, pdict = cgi.parse_header(self.headers['Content-Type'])

            # # refuse to receive non-json content
            # if ctype != 'application/json':
            #     logging.exception('POST request: Wrong format of the content: only json is accepted.')

            length = int(self.headers.get('content-length'))
            payload_string = self.rfile.read(length).decode('utf-8')
            logging.info(payload_string)
            if payload_string:
                message = json.loads(payload_string)
            else:
                logging.exception('POST request: Empty input received.')
                return

            #Заглушка
            post_reqid = 1
            post_maxlen = 5
            test_mode = False
            post_img = payload_string
            post_img = bytes(post_img, 'utf-8')
            img = stringToRGB(post_img)


            #Раскомментить вместо заглушки

            # logging.info("POST request: message received, unpacking.")
            # message = json.loads(message)

            # if "reqid" in message.keys():
            #     post_reqid = message["reqid"]
            # else:
            #     post_reqid = 1
            #
            # if "maxlen" in message.keys():
            #     post_maxlen = message["maxlen"]
            # else:
            #     post_maxlen = 5
            #
            # if type(post_reqid) is not int:
            #     logging.exception('type error: reqid is not int.')
            # if type(post_maxlen) is not int:
            #     logging.exception('type error: maxlen is not int.')
            #
            # logging.info('Reqid = {}; Maxlen = {}'.format(post_reqid, post_maxlen))
            # try:
            #     post_img = bytes(message["img"], 'utf-8')
            #     img = stringToRGB(post_img)
            #     logging.info("POST request: image decoded successfully.")
            # except Exception as exc:
            #     logging.exception('Finding similar images: {}. Image can not be decoded from string to an array.'.format(exc))
            #
            # test_mode = False
            # if "test_mode" in message.keys():
            #     test_mode = message['test_mode']

            logging.info('Finding similar images: function started.')
            json_data = self.compute(img, maxlen=post_maxlen, reqid=post_reqid, test_mode=test_mode)
            logging.info('Finding similar images: data returned.')
        except Exception as exc:
            # self.send_response(500)
            # self.end_headers()
            code = 500
            errormsg = 'An error occured: {}'.format(exc)  # 'Index does not exist in the database'
            json_data = json.dumps({"reqid": str(post_reqid), "code": str(code), "data": errormsg})
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
