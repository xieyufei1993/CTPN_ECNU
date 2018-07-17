from bottle import route, run, request, response
from mydetector import mydetector
import json
import sys
sys.path.append("mydetector/src")

worker = mydetector.ImageDetector()
detector_model = worker.load_model(prototxt_dir='models/deploy.prototxt',
        caffemodel_dir='models/ctpn_trained_model.caffemodel')
@route('/v1/ocr/detection', method='POST')
def predict():
    if request.json == None:
        response.code = 500
        return {'code': 1, 'msg': 'no request data'}
    try:
        uri = request.json['data']['uri']
    except Exception:
        response.code = 500
        return {'code': 2, 'msg': 'request data is invalid'}
    try:
        ret = worker.detect(uri,detector_model)
    except Exception:
        response.code = 500
        return {'code': 3, 'msg': 'server error'}
    return {'code': 0, 'results': ret}

if __name__ == '__main__':
    run(host='0.0.0.0', port='8080', debug=True)