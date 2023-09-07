from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

MINIMUM_CONFIDENCE = 0.5

# Load the model
model = YOLO('yolov8n.pt')

# Predict with the model
results: list[Results] = model.predict(
    'https://ultralytics.com/images/bus.jpg',
    classes=0)

for result in results:
    detection_count = sum(
        confidence >= MINIMUM_CONFIDENCE
        for (_,_,_,_, confidence, _name_index)
        in result.boxes.data)
    
    print(f'Detected {detection_count} persons')

    im_array = result.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
