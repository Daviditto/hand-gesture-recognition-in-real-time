import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json

class HandGestureModel(object):
    gesture = {0: '下面', 1: '八', 2: '五', 3: '四', 4: '左边', 5: '九', 6: '一', 7: '右边', 8: '七', 9: '六', 10: '停下', 11: '三', 12:'二', 13: '上面', 14: '零'}

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, 'r') as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)

    def predict_gesture(self, img):
        self.preds = self.loaded_model.predict(img)
        return HandGestureModel.gesture.get(int(np.argmax(self.preds)))
