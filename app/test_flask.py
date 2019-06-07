import os
import json
import re
from flask import Flask, make_response, request, jsonify
import logging

# Example URL: http://127.0.0.1:5000/cameras?camera_no=135&dd=2016-10-12&time_start=150000&time_end=160000

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG, filename='test_flask_log.log')

app = Flask(__name__) # make your Flask application that is the name of this file

def filter_cameras(sub_dir, camera_no, dd, time_start, time_end):
    all_files = os.listdir('../../BlockedBikeLaneTrainingSingleCam/' + sub_dir)
    
    filtered_files = []
    for f in all_files:
        file_match = re.search(dd + "\s(\d+)\scam" + camera_no, f)
        if file_match is None:
            continue
        elif (int(file_match[1]) >= int(time_start) and int(file_match[1]) <= int(time_end)):
            filtered_files.append(f)
    return(filtered_files)

@app.route("/cameras")
def call_my_app():
    ccc = str(request.args.get('camera_no'))
    ddd = str(request.args.get('dd'))
    tsts = str(request.args.get('time_start'))
    tete = str(request.args.get('time_end'))

    files = filter_cameras('blocked', ccc, ddd, tsts, tete)
    
    json_data = json.dumps(files)
    
    resp = make_response(json_data)
    resp.mimetype = 'application/json' # return json as mimetype
    
    return(resp)
