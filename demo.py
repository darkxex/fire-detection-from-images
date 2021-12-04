"""
Source: https://github.com/AK391/yolov5/blob/master/utils/gradio/demo.py
"""

import gradio as gr
import torch
from PIL import Image

# Model
# model = torch.hub.load('C:/Users/darkx/Documents/GitHub/yolov5', 'yolov5s')  # force_reload=True to update
model = torch.hub.load('C:/Users/darkx/Documents/GitHub/yolov5', 'custom', 'pytorch/object-detection/yolov5/experiment1/best.pt', source='local')  # local repo


def yolo(im, size=640):
    g = (size / max(im.size))  # gain
    im = im.resize((int(x * g) for x in im.size), Image.ANTIALIAS)  # resize

    results = model(im)  # inference

    #Se convierte los resultados a una lista Pandas, para luego tomar su largo, dependiendo de la cantidad detectada, es el resultado de emisores de fuego.
    cantidadf = len(results.pandas().xyxy[0])
    if (cantidadf > 0):
        print("Peligro, "+ str(cantidadf) +" emisor(es) de fuego detectado.")
        #aqui puede agregar que envie un correo o un mensaje de texto y etc.
    else:
        print("No se detecta fuego en la imagen.")

   
    
    results.render()  # updates results.imgs with boxes and labels
    return Image.fromarray(results.imgs[0])


inputs = gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Image(type="pil", label="Output Image")

#Se reescribe el HTML para mejor interpretacion del codigo.
title = "Detectar Fuego / Proyecto Utem"
description = "Red neuronal basada en YOLOv5 entrenada para detectar fuego."
article = "<p style='text-align: center'>YOLOv5 is a family of compound-scaled object detection models trained on the COCO dataset, and includes " \
          "simple functionality for Test Time Augmentation (TTA), model ensembling, hyperparameter evolution, " \
          "and export to ONNX, CoreML and TFLite. <a href='https://github.com/ultralytics/yolov5'>Source code</a> |" \
          "<a href='https://apps.apple.com/app/id1452689527'>iOS App</a> | <a href='https://pytorch.org/hub/ultralytics_yolov5'>PyTorch Hub</a></p>"


#se agrega la imagen de un perro como ejemplo para ver si lo detectaba como fuego.
examples = [['images/pan-fire.jpg'], ['images/fire-basket.jpg'], ['images/perro.jpg']]
gr.Interface(yolo, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(
    debug=True)