import csv
import math
from pathlib import Path
import cv2
import numpy as np

from src.cnn.annotations import read_yolo_labels as readYoloLabels
from src.hsv.segment import segment_leaf as segmentLeaf

def findImages(imageFolder):
    imagePaths = {}

    for imagePath in sorted(imageFolder.rglob("*")):
        if imagePath.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            if imagePath.stem in imagePaths:
                raise ValueError("Duplicate image name: " + imagePath.stem)
            imagePaths[imagePath.stem] = imagePath

    return imagePaths

def getBoxBounds(box, imageWidth, imageHeight):
    x, y, width, height = box
    left = max(0, math.floor(x))
    top = max(0, math.floor(y))
    right = min(imageWidth, math.ceil(x + width))
    bottom = min(imageHeight, math.ceil(y + height))

    return left, top, right, bottom

def measureBox(mask, box):
    imageHeight, imageWidth = mask.shape
    left, top, right, bottom = getBoxBounds(box, imageWidth, imageHeight)
    boxMask = mask[top:bottom, left:right]
    boxPixels = boxMask.size

    if boxPixels == 0:
        raise ValueError("An annotation has no pixels inside the image.")

    retainedPixels = int(np.count_nonzero(boxMask))
    percentRetained = round(retainedPixels / boxPixels * 100, 2)

    return retainedPixels, boxPixels, percentRetained

def saveComparison(image, mask, maskedImage, annotations, outputPath):
    maskImage = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    panels = [image.copy(), maskImage, maskedImage.copy()]
    titles = ["Original + labels", "HSV mask + labels", "Masked + labels"]
    imageHeight, imageWidth = mask.shape
    labeledPanels = []

    for panelIndex in range(len(panels)):
        panel = panels[panelIndex]

        for annotation in annotations:
            left, top, right, bottom = getBoxBounds(
                annotation.box, imageWidth, imageHeight
            )
            cv2.rectangle(
                panel, (left, top), (right - 1, bottom - 1), (0, 255, 0), 1
            )

        panel = cv2.copyMakeBorder(
            panel, 35, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        cv2.putText(
            panel, titles[panelIndex], (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
        labeledPanels.append(panel)

    comparison = np.hstack(labeledPanels)

    if not cv2.imwrite(str(outputPath), comparison):
        raise OSError("Could not save: " + str(outputPath))

def runCheck():
    projectFolder = Path(__file__).resolve().parent
    imageFolder = projectFolder / "data/raw/plantvillage_apple"
    labelFolder = projectFolder / "data/labels/shared_sample"
    outputFolder = projectFolder / "outputs/hsvCheck"
    labelPaths = sorted(labelFolder.glob("*.txt"))

    if not labelPaths:
        raise FileNotFoundError("No annotation files found in " + str(labelFolder))

    imagePaths = findImages(imageFolder)

    for labelPath in labelPaths:
        if labelPath.stem not in imagePaths:
            raise FileNotFoundError("Missing original photo: " + labelPath.stem)

    outputFolder.mkdir(parents=True, exist_ok=True)
    rows = []
    emptyBoxes = 0

    for labelPath in labelPaths:
        imagePath = imagePaths[labelPath.stem]
        image = cv2.imread(str(imagePath))

        if image is None:
            raise ValueError("Could not read: " + str(imagePath))

        imageHeight, imageWidth = image.shape[:2]
        annotations = readYoloLabels(labelPath, imageWidth, imageHeight)
        result = segmentLeaf(image)

        for boxIndex in range(len(annotations)):
            annotation = annotations[boxIndex]
            retainedPixels, boxPixels, percentRetained = measureBox(
                result.lesion_mask, annotation.box
            )
            emptyBox = retainedPixels == 0

            if emptyBox:
                emptyBoxes += 1

            rows.append([
                imagePath.name, annotation.label_name, boxIndex + 1,
                retainedPixels, boxPixels, percentRetained, emptyBox
            ])

        outputPath = outputFolder / (labelPath.stem + "_comparison.png")
        saveComparison(
            image, result.lesion_mask, result.masked, annotations, outputPath
        )

    with open(outputFolder / "boxCoverage.csv", "w", newline="") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([
            "image", "disease", "boxNumber", "retainedPixels",
            "boxPixels", "percentRetained", "emptyBox"
        ])
        writer.writerows(rows)

    print("HSV MASK CHECK")
    print("Images checked:", len(labelPaths))
    print("Annotated boxes:", len(rows))
    print("Boxes with no pixels retained:", emptyBoxes)
    print("Comparison images: outputs/hsvCheck/")
    print("CSV results: outputs/hsvCheck/boxCoverage.csv")
    print("Coverage measures box area retained, not detection accuracy.")

if __name__ == "__main__":
    runCheck()