## Project

The project includes two models:

- **Face detection.** 
  Detection is done by outputing a heatmap of faces. Using this heatmap bounding boxes are created.
   
- **Face recognition.**
  Detected faces are cropped using the bounding boxes and encoded into a vector of 126 values.
  This vector is then compared to the vectors of faces stored in data/infer_data.

:heavy_exclamation_mark: Performance of the detection model is already reasonable but I'm still strugling with recognition model.

## Project structure
```
    .
    ├── checkpoints/                    # dir for model weights
    ├── data/                           # dir for data storing and generation 
    │   ├── infer_faces/                # directory for storing pictures of faces to detect during inference
    │   ├── imdb_face.py                # imdb dataset utils
    │   └── wider_face.py               # wider face dataset utils
    ├── models/                         # dir for model definitions
    │   ├── detection.py                # -
    │   ├── recognition.py              # -
    │   └── face_align.py               # not implemented
    ├── infer.py                        # -
    ├── train_detection.py              # -
    ├── train_recognition.py            # -
    ├── utils.py                        # utility functions
    └── README.md                       # -
```

## Installation

```
git clone https://github.com/RaidasGrisk/tf2-face-detection-recognition.git  
cd tf2-face-detection-recognition  

conda env create -f environment.yml  
conda activate tf2-face-detection-recognition  
```

## Inference
Place the images of people faces inside data/infer_data
```
python infer.py
```

## Training
Explore train_detection.py train_recognition.py and data directory.  

To re-train or continue training the models you would need datasets   
which are not included as part of this git.  

Detection model is trained on [wider face dataset](http://shuoyang1213.me/WIDERFACE/)  
Recognition model is trained on [imdb face dataset](https://github.com/fwang91/IMDb-Face)
