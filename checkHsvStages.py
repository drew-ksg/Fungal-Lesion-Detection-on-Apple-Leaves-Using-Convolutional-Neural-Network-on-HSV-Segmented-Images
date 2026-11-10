import csv
from pathlib import Path
import cv2
import numpy as np

from checkHsvMasks import findImages, getBoxBounds, measureBox
from src.cnn.annotations import read_yolo_labels as readYoloLabels
from src.hsv.segment import to_hsv as toHsv
from src.hsv.segment import threshold_mask as thresholdMask
from src.hsv.segment import leaf_region as leafRegion
from src.hsv.segment import clean_mask as cleanMask
from src.hsv.segment import segment_leaf as segmentLeaf
from src.hsv.thresholds import DISEASE_RANGES as diseaseRanges

def getStages(image):
    hsvImage = toHsv(image)
    colorMask = np.zeros(hsvImage.shape[:2], dtype=np.uint8)

    for ranges in diseaseRanges.values():
        currentMask = thresholdMask(hsvImage, ranges)
        colorMask = cv2.bitwise_or(colorMask, currentMask)

    leafMask = leafRegion(hsvImage)
    insideLeaf = cv2.bitwise_and(colorMask, leafMask)
    openedMask = cleanMask(insideLeaf, open_size=3, close_size=0)
    finalMask = cleanMask(openedMask, open_size=0, close_size=5)
    originalResult = segmentLeaf(image)

    if not np.array_equal(finalMask, originalResult.lesion_mask):
        raise ValueError("Stage steps no longer match segment_leaf. Check its settings.")

    return [colorMask, leafMask, insideLeaf, openedMask, finalMask]

def getEmptyReason(pixelCounts):
    if pixelCounts[4] > 0:
        return "notEmpty"
    if pixelCounts[0] == 0:
        return "threshold"
    if pixelCounts[2] == 0:
        return "leafMask"
    if pixelCounts[3] == 0:
        return "opening"
    return "closing"

def saveStages(image, masks, annotations, outputPath):
    panels = [image.copy()]
    titles = [
        "Original + labels", "Color thresholds", "Leaf mask",
        "Inside leaf", "After opening", "Final cleanup"
    ]

    for mask in masks:
        panels.append(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    imageHeight, imageWidth = image.shape[:2]
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

    topRow = np.hstack(labeledPanels[:3])
    bottomRow = np.hstack(labeledPanels[3:])
    comparison = np.vstack([topRow, bottomRow])

    if not cv2.imwrite(str(outputPath), comparison):
        raise OSError("Could not save: " + str(outputPath))

def runCheck():
    projectFolder = Path(__file__).resolve().parent
    imageFolder = projectFolder / "data/raw/plantvillage_apple"
    labelFolder = projectFolder / "data/labels/shared_sample"
    outputFolder = projectFolder / "outputs/hsvStages"
    labelPaths = sorted(labelFolder.glob("*.txt"))

    if not labelPaths:
        raise FileNotFoundError("No annotation files found in " + str(labelFolder))

    imagePaths = findImages(imageFolder)

    for labelPath in labelPaths:
        if labelPath.stem not in imagePaths:
            raise FileNotFoundError("Missing original photo: " + labelPath.stem)

    outputFolder.mkdir(parents=True, exist_ok=True)
    stageNames = ["threshold", "leafMask", "insideLeaf", "opening", "final"]
    reasons = {"threshold": 0, "leafMask": 0, "opening": 0, "closing": 0}
    columns = ["image", "disease", "boxNumber", "boxPixels"]

    for stageName in stageNames:
        columns.append(stageName + "Pixels")
        columns.append(stageName + "Percent")
    columns.append("finalEmptyReason")
    rows = []

    for labelPath in labelPaths:
        imagePath = imagePaths[labelPath.stem]
        image = cv2.imread(str(imagePath))

        if image is None:
            raise ValueError("Could not read: " + str(imagePath))

        imageHeight, imageWidth = image.shape[:2]
        annotations = readYoloLabels(labelPath, imageWidth, imageHeight)
        masks = getStages(image)

        for boxIndex in range(len(annotations)):
            annotation = annotations[boxIndex]
            pixelCounts = []
            stageValues = []

            for mask in masks:
                retainedPixels, boxPixels, percentRetained = measureBox(
                    mask, annotation.box
                )
                pixelCounts.append(retainedPixels)
                stageValues.extend([retainedPixels, percentRetained])

            reason = getEmptyReason(pixelCounts)
            if reason != "notEmpty":
                reasons[reason] += 1

            row = [imagePath.name, annotation.label_name, boxIndex + 1, boxPixels]
            row.extend(stageValues)
            row.append(reason)
            rows.append(row)

        outputPath = outputFolder / (labelPath.stem + "_stages.png")
        saveStages(image, masks, annotations, outputPath)

    with open(outputFolder / "stageCoverage.csv", "w", newline="") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(columns)
        writer.writerows(rows)

    print("HSV STAGE CHECK")
    print("Images checked:", len(labelPaths))
    print("Annotated boxes:", len(rows))
    print("Final boxes with no pixels retained:", sum(reasons.values()))
    print()
    print("FINAL EMPTY BOX BREAKDOWN")
    print("Empty at color thresholds:", reasons["threshold"])
    print("Emptied by leaf mask:", reasons["leafMask"])
    print("Emptied by opening:", reasons["opening"])
    print("Emptied by closing:", reasons["closing"])
    print()
    print("Stage images: outputs/hsvStages/")
    print("CSV results: outputs/hsvStages/stageCoverage.csv")
    print("Coverage measures box area retained, not detection accuracy.")

if __name__ == "__main__":
    runCheck()