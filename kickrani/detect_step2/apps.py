from django.apps import AppConfig
from models.experimental import attempt_load
from utils.torch_utils import select_device

class DetectStep2Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detect_step2'

    weights1 = './kickboard.pt'
    weights2 = './helmet.pt'
    weights3 = './person.pt'
    device = ''
    device = select_device(device)


    model1 = attempt_load(weights1, map_location=device)  # load FP32 model
    model2 = attempt_load(weights2, map_location=device)  # load FP32 model
    model3 = attempt_load(weights3, map_location=device)  # load FP32 model
