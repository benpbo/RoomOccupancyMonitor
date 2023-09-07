from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Load the model
model = YOLO('yolov8n.pt')

# Predict with the model
results: list[Results] = model.predict('https://ultralytics.com/images/bus.jpg')

for result in results:
    im_array = result.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
