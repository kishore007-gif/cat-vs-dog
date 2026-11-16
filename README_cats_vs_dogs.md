# Cats vs Dogs Image Classifier 🐱🐶

A Convolutional Neural Network (CNN) built with Keras that looks at an image and classifies it as a **cat** or a **dog**.

**Author:** Kishore R

---

## 📌 Project Overview

This project trains a CNN from scratch on labelled cat and dog images. It uses image augmentation to improve generalisation, then predicts the class of a new, unseen image.

## 🎯 Objectives

- Build a CNN for binary image classification (cat vs dog).
- Load and augment images directly from folders with `ImageDataGenerator`.
- Train on the training set and validate on a separate test set.
- Predict the class of a single new image.

## 🧠 Model Architecture

| Layer | Details |
|---|---|
| Input | 64 × 64 × 3 RGB image |
| Conv2D | 32 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 32 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Flatten | Converts feature maps to a vector |
| Dense | 128 units, ReLU |
| Dense (output) | 1 unit, Sigmoid (probability of "dog") |

- **Optimizer:** Adam
- **Loss:** Binary cross-entropy
- **Metric:** Accuracy

## 🔄 Methodology

1. **Load images** from `training_set` and `test_set` folders, resized to 64 × 64, in batches of 32.
2. **Augment training images:** rescale pixels by 1/255, shear range 0.2, zoom range 0.2, horizontal flip.
3. **Rescale test images** by 1/255 only (no augmentation).
4. **Train** the CNN for 25 epochs, validating on the test set.
5. **Predict** a single image: load it, resize to 64 × 64, convert to an array, and pass it to the model. The output is labelled "dog" or "cat".

## 📂 Dataset Structure

Organise the images in one sub-folder per class, which `flow_from_directory` uses as labels:

```
dataset/
├── training_set/
│   ├── cats/
│   └── dogs/
├── test_set/
│   ├── cats/
│   └── dogs/
└── Dog or Cat ? Predict/
    └── cat_or_dog_1.jpg
```

Dataset used: **[add dataset name and source, number of training and test images]**

## 🛠️ Technologies Used

- Python 3
- Keras / TensorFlow
- NumPy
- Jupyter Notebook / IPython

## ⚙️ Installation and Usage

```bash
# 1. Clone the repository
git clone <your-repository-url>
cd <your-repository-folder>

# 2. Install dependencies
pip install tensorflow numpy pillow jupyter

# 3. Add the dataset in the folder structure above and update the
#    training_data_path, test_data_path and test image paths in the script

# 4. Run
python cats_vs_dogs_challenge.py
```

## 📊 Results

| Metric | Value |
|---|---|
| Training accuracy | **[add after training]** |
| Validation (test) accuracy | **[add after training]** |

## ⚠️ Notes for the Script

- The cell that sets `training_data_path` is missing a line break, so it runs the path and the `train_datagen.flow_from_directory(...)` call together. This causes a syntax error and needs to be split into two lines.
- `steps_per_epoch = 8000` and `validation_steps = 2000` are meant to be the number of **batches**, not images. With a batch size of 32, use about `number_of_images / 32` (for example 8000 images gives 250 steps).
- `fit_generator` is deprecated in current TensorFlow; use `classifier.fit(...)` instead.
- The prediction check `result[0][0] == 1` should be `result[0][0] > 0.5`, because the sigmoid output is a probability and rarely exactly 1.
- The prediction is a dog when the sigmoid output is close to 1. Confirm this with `training_set.class_indices` (alphabetical order puts `cats` as 0 and `dogs` as 1).

## 🔮 Future Scope

- Add more convolutional layers, dropout and batch normalisation.
- Use transfer learning with a pretrained model such as VGG16 or MobileNet for higher accuracy.
- Use a larger image size than 64 × 64.
- Plot accuracy and loss curves and add a confusion matrix.
- Build a simple web interface for uploading an image and getting the prediction.

## 📜 License

This project is created for educational purposes.
