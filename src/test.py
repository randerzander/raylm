from nemotron_ocr.inference.pipeline import NemotronOCR

ocr = NemotronOCR()

predictions = ocr("/localhome/local-jdyer/nemotron-ocr-v1/ocr-example-input-1.png")

for pred in predictions:
    print(
        f"  - Text: '{pred['text']}', "
        f"Confidence: {pred['confidence']:.2f}, "
        f"Bbox: [left={pred['left']:.4f}, upper={pred['upper']:.4f}, right={pred['right']:.4f}, lower={pred['lower']:.4f}]"
    )