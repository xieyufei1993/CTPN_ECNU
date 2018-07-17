from mydetector import mydetector
import sys
sys.path.append("mydetector/src")
def main():
    worker = mydetector.ImageDetector()
    detector_model = worker.load_model(
        prototxt_dir='models/deploy.prototxt',
        caffemodel_dir='models/ctpn_trained_model.caffemodel'
    )
    result = worker.detect('http://pbuq2pt5s.bkt.clouddn.com/ecnu_test1.png',detector_model)
    print result

if __name__ == '__main__':
    main()