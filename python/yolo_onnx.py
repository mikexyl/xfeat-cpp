from ultralytics import YOLO, YOLOE
# model = YOLO("yolo26n-seg.pt")  # It will auto-download the weights
# Load the open-vocabulary Nano segmentation model
model = YOLOE("yoloe26n-seg.pt")

# Set the 'persistent' classes so the ONNX engine is specialized
model.set_classes(["sky", "cloud"])
model.export(format="engine", half=True, imgsz=512, opset=17, workspace=4)