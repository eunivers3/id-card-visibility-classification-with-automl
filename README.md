***


# Classifying the visibility of ID cards in photos

The folder of images contains several different types of ID documents taken in different conditions and backgrounds. The goal is to use the images stored in this folder and to design an algorithm that identifies the visibility of the card on the photo (FULL_VISIBILITY, PARTIAL_VISIBILITY, NO_VISIBILITY).

## Data

Inside the data folder you can find the following:
- gicsd_labels.csv
- original_images
- split_augmented_images
    - labels.csv
    - train
        - full_visibility
        - partial_visibility
        - no_visibility
    - val
        - full_visibility
        - partial_visibility
        - no_visibility
    - test
        - full_visibility
        - partial_visibility
        - no_visibility

### original_images
[original_images](data/original_images) contains the original images. Original dataset from [MIDV-500](https://arxiv.org/abs/1807.05786): A Dataset for Identity Documents Analysis and Recognition on Mobile Devices in Video Stream.

### gicsd_labels.csv
[gicsd_labels.csv](data/gicsd_labels.csv) contains the ground truth labels for the original images.

### split_augmented_images
[split_augmented_images](data/split_augmented_images) contains the modified, split images that was used to train the classification model.

### split_augmented_images/labels.csv
[labels.csv](data/split_augmented_images/labels.csv) is a file mapping each challenge image with its correct label, location on Google Cloud and whether it was used to train validate or test the model.
- **SET**: Whether it was used to train, validate or test the model. 
- **GCS_URI**: GCS location of each image.
- **LABEL**: The label of each image, which can be one of these values: FULL_VISIBILITY, PARTIAL_VISIBILITY or NO_VISIBILITY. 
    
## Dependencies

- Make sure you have [gcloud](https://cloud.google.com/sdk/downloads#interactive) command line tool installed
- Set environment variable to the path to the AutoMLservice account key JSON that authenticates the prediction model usage. 
`export GOOGLE_APPLICATION_CREDENTIALS=path-to-key-file`
- Install requirements
`pip install -r /path/to/requirements.txt`

## Run Instructions

`python predict.py (path_to_image)`

All responses in [predict.py](predict.py) will have the the form:
```json
{
    "filepath" : "string, path to the image file provided",
    "visibility": "string, ID card visibility prediction of the image",
    "score" : "float, prediction score"
}
```

## Approach
Built object classification models to label the visibility of ID cards in images using Google AutoML. Evaluated each model and deployed the chosen model on GCP.

See [Visibility_Classification.ipynb](Visibility_Classification.ipynb) for the full breakdown, reasoning and model evaluation.

### Summary
1. Data exploration and preprocessing determined that out of the 800 images  provided (each size; 192x192), 646 were labelled FULL_VISIBILITY, 123 were labelled PARTIAL_VISIBILITY and 31 were labelled NO_VISIBILITY. The data contained no missing values or duplications.
2. Images were split in to 3 separate folders by their visibility classification.
3. Data augmentation applied to increase the **training** dataset for PARTIAL_VISIBILITY (5-fold) and NO_VISIBILITY (10-fold) via various geometric transformations. This included; rotations, translations, zooms, brightness changes, perspective tilts and horizontal/vertical flips.
4. New augmented **training** dataset for PARTIAL_VISIBILITY and NO_VISIBILITY replaced their original training dataset before splitting the images into train-validate-split batches (70:15:15). 
5. Images converted from RGB into single-channel arrays for feature engineering. Methods used included:
    - Gaussian blurring to reduce noise (due to image corruptions)
    - Contrasting, Histogram Equalisations
6. Formatted a training data [csv](https://cloud.google.com/vision/automl/object-detection/docs/csv-format) for AutoML called [labels.csv](data/labels.csv) to point to the location of each image on Google Cloud Storage, their labels and whether to use them for training, validation or testing during model creation
7. Make bucket to store the images `gsutil mb gs://bucket-name`. Uploaded the final, modified image dataset + [labels.csv](data/labels.csv) to a Google Cloud Bucket (parallel copy for large file numbers)
`gsutil -m cp -r dir gs://YOUR-BUCKET-NAME-HERE`
8. Create a new Google AutoML dataset and import the labelled images + `labels.csv` to this **
9. Create the model based on that datset and start training **
10. Display model evaluations **
11. Repeat steps 6 to 10 to build models based on different image datasets generated in `Visibility_Classification.ipynb` **
12. Deployed the chosen model by using AutoML's Python client endpoint in `predict.py`.
13. `predict.py` takes an image from a local directory, preprocesses it (converts to gray, applies gaussian filter), saving the modified image on a temp file which is discarded once the prediction on it is complete.

** Done via Google AutoML web interface

## Future Work
* Set up data pipline for ETL of images to AutoMl model creation/re-training.
* Maybe remove NO_VISIBILITY altogether due to little data
* Look at image metadata to extract other information e.g.
    - timestamp of capture
    - altitude
    - location
    - address
    - device used etc.
* Experiment with other image preprocessing techniques to see if they impove the model e.g.
	- edge dilation (growing boundary regions)
	- filling in of image region boundary pixels
* Add more images to any labels with low quality. Collect more images per uneven classes - ideally obtain uncorrupted images to build the model from
* Object segmentation (of the card), looking at complete edges to indicate full visibility
* Object detection --> crop the image --> process cropped image with less noise, which should increase the accuracy of the model
* TTA (test time augmentation)
    - data augmentation of test image if model is unsure
    - collect average label of those classes
* Research & build a custom NN model, e.g. with Tensorflow or Keras, do k fold x-validations, train model on modified instances of the original image (vs saving all modified images to build the AutoML model on)
* Delete low quality images in each class (images that are difficult to label by human eye) to increase model accuracy