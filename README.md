## Project

The project includes two models:

- **Face detection.** 
  Detection is done by outputing a heatmap of faces. Using this heatmap bounding boxes are created.
   
- **Face recognition.**
  Detected faces are cropped using the bounding boxes and encoded into a vector of 126 values.
  This vector is then compared to the vectors of faces stored in data/infer_faces.

:heavy_exclamation_mark: Performance of the detection model is reasonable but I'm still strugling with recognition model.

## Demo

![Demo](data/other/gif.gif)

## Project structure
```
    .
    ├── checkpoints/                    # dir for model weights
    ├── data/                           # dir for data storing and generation 
    │   ├── infer_faces/                # directory for storing pictures of faces to detect during inference
    │   ├── other/                      # other none code files video/gifs etc.
    │   ├── imdb_face.py                # imdb dataset utils (used for recognition training)
    │   └── wider_face.py               # wider face dataset utils (used for detection training)
    ├── models/                         # dir for model definitions
    │   ├── detection.py                # -
    │   ├── recognition.py              # -
    │   └── face_align.py               # not implemented
    ├── infer.py                        # -
    ├── train_detection.py              # -
    ├── train_recognition.py            # -
    ├── utils.py                        # -
    └── README.md                       # -
```

## Installation and dependencies

To run this project the following libraries are required: tensorflow, numpy, opencv, pillow, pandas.  
Run this on your own env or create a new one using [environment.yml](environment.yml)

```
git clone https://github.com/RaidasGrisk/tf2-face-detection-recognition.git  
cd tf2-face-detection-recognition  

conda env create -f environment.yml  
conda activate tf2-face-detection-recognition  
```

## Inference
Place images of people's faces inside data/infer_faces (see examples inside this dir). These images will be encoded and compared with faces detected during inference.
```
python infer.py
```

## Training
For detection model explore train_detection.py and data/wider_face.py files.
Fore recognition model explore train_recognition.py data/imdb_face.py files.

To re-train or continue training the models you would need datasets which are not included as part of this git.  

Detection model is trained on [wider face dataset](http://shuoyang1213.me/WIDERFACE/).  
Recognition model is trained on [imdb face dataset](https://github.com/fwang91/IMDb-Face).
