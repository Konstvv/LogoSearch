import requests
import json
import logging
r = requests.get('http://127.0.0.1:8000/')
logging.info("GET request from .py script sent.")
print(r.json())