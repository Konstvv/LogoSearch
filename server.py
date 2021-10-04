import http.server
import json
import socketserver
from KNN import Similarity
from Vectors import stringToRGB
import cgi


class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    @staticmethod
    def compute(img, maxlen=5, reqid=None):
        global Sim
        code = 200
        values = Sim.ind_similar(img, n_neighbors=maxlen)
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

        self.wfile.write(bytes(json.dumps({'GET': 'request', 'received': 'ok'}), 'utf-8'))
        return

    def do_POST(self):
        post_reqid = None
        json_data = json.dumps({"reqid": str(post_reqid), "code": str(500), "data": 'POST request was not made'})
        try:
            self._set_response()
            ctype, pdict = cgi.parse_header(self.headers['Content-Type'])

            # refuse to receive non-json content
            if ctype != 'application/json':
                raise Exception('Wrong format of the content: only json is accepted.')

            length = int(self.headers.get('content-length'))
            payload_string = self.rfile.read(length).decode('utf-8')
            if payload_string:
                message = json.loads(payload_string)
            else:
                raise Exception('Empty input received.')

            message = json.loads(message)
            post_reqid = message["reqid"]
            post_img = bytes(message["img"], 'utf-8')
            post_maxlen = message["maxlen"]
            if type(post_reqid) is not int:
                raise Exception('type error: reqid is not int.')
            if type(post_maxlen) is not int:
                raise Exception('type error: maxlen is not int.')

            print('Image has been acquired')
            print('Reqid = {}; Maxlen = {}'.format(post_reqid, post_maxlen))
            img = stringToRGB(post_img)
            json_data = self.compute(img, maxlen=post_maxlen, reqid=post_reqid)
        except Exception as exc:
            # self.send_response(500)
            # self.end_headers()
            code = 500
            errormsg = 'An error occured: {}'.format(exc)  # 'Index does not exist in the database'
            json_data = json.dumps({"reqid": str(post_reqid), "code": str(code), "data": errormsg})
        finally:
            self.wfile.write(bytes(json_data, 'utf-8'))


Sim = Similarity()
# Create an object of the above class
handler_object = MyHttpRequestHandler

PORT = 8000
my_server = socketserver.TCPServer(("", PORT), handler_object)

# Start the server
my_server.serve_forever()