# Model weights

Drop trained model files here to activate the corresponding AI backend. No code
changes are required — backends discover weights by convention.

    detection/   *.onnx     Generic object detection (YOLOv5/YOLOv8 export, COCO classes)
    components/  *.onnx     Electrical-component detector (custom-trained)
    components/  labels.txt Optional: one class name per line (defaults to the built-in 18 classes)

    fire/        *.onnx     Fire / smoke / explosion detector      (labels.txt optional)
    weapon/      *.onnx     Weapon detector: gun / rifle / knife   (labels.txt optional)
    ppe/         *.onnx     PPE detector: helmet/vest/gloves/goggles (+ no_* violation classes)
    human/       *.onnx     Person detector (a COCO YOLO export works; filtered to "person")
    vehicle/     *.onnx     Vehicle detector (COCO YOLO export works; car/truck/bus/motorcycle)
    violence/    *.onnx     Violence / fighting / assault classifier
    fall/        *.onnx     Fall / posture detector (standing/sitting/falling/lying)

Each of the above module directories takes an optional `labels.txt` (one class
name per line) to override the built-in default class order for that model.

Then select + enable the backend on the AI Models page, or via the API:

    POST /api/ai/models/detection/select  {"backend_id": "onnx_yolo"}
    POST /api/ai/models/detection/enable  {"enabled": true}

For real ArcFace face recognition, install InsightFace (`pip install insightface`)
and select the `insightface_arcface` backend; its models are managed by that library.

No suitable public pretrained model exists for the electrical component/wire
classes, so those backends stay inert (returning empty results, never fabricated
detections) until you supply trained weights here. The same is true for the
fire / weapon / ppe / violence / fall / human / vehicle modules: each reports
`weights_missing` ("model unavailable") in the AI status until you drop a
trained `.onnx` into its directory, at which point it runs real inference with
no code changes. The `fall` task also ships an opt-in classical aspect-ratio
heuristic (`heuristic_fall`) that needs no weights but is explicitly a baseline,
not a trained model.
