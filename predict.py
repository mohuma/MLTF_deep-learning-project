import numpy as np
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_input():
    '''
    This function gets the input from the command line and parses it.
    Required input --> image path and model name
    Optional input --> top_k integer value (default is 1) and the Path to a JSON file mapping labels to flower names (default is label_map.json)

    Args:
        None

    Returns:
        Image path, model name, top_k value and label map.
    '''
    parser = argparse.ArgumentParser(description='Get the input from the command line')

    parser.add_argument('image_path', action="store")
    parser.add_argument('model_name', action="store")
    parser.add_argument('--top_k', action="store", dest="top_k", type=int, default=1)
    parser.add_argument('--category_names', action="store", dest="label_map", default='label_map.json')

    results = parser.parse_args()

    return results.image_path, results.model_name, results.top_k, results.label_map

def get_labels(label_map):
    '''
    This function loads the flower names along with their classes (labels) stored in a JSON file into a dictionary.

    Args:
        Path to a JSON file

    Returns:
        A dictionary mapping classes (labels) to flower names.
    '''
    with open(label_map, 'r') as f:
        return json.load(f)

def load_model(model_name):
    '''
    This function loads a pretrained model in order to predict the flower name from an image.

    Args:
        Model name

    Returns:
        A pretrained model capable of predicting the name of a flower using its image.
    '''
    return tf.keras.models.load_model(model_name, custom_objects = {'KerasLayer':hub.KerasLayer}, compile = False)

def process_image(image):
    '''
    This function preprocesses the image so that it can be fed into the pretrained model.

    Args:
        Image

    Returns:
        A processed image.
    '''
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()

def predict(image_path, my_model, top_k):
    '''
    This function predicts the top K flower names (or classes) from the image using the pretrained model.

    Args:
        Path to the image, model name, top_k value

    Returns:
        Top K flower names (or classes) along with their corresponding probabilities.
    '''
    processed_image = np.expand_dims(process_image(np.asarray(Image.open(image_path))), axis = 0)
    prediction = my_model.predict(processed_image)
    result = tf.math.top_k(prediction, k = top_k)
    probs = result.values.numpy()[0]
    classes = result.indices.numpy()[0]+1
    return probs, classes.astype(str)

def main():
    '''
    This main function calls other functions to predict the top K flower names from an image along with their corresponding probabilities using a pretrained network model.
    '''
    image_path, model_name, top_k, label_map = get_input()

    class_names = get_labels(label_map)

    my_model = load_model(model_name)

    probs, classes = predict(image_path, my_model, top_k)

    flower_names = [class_names[x] for x in classes]

    if top_k == 1:
        print('\nThe model predicted the image as {} having class {} with a probability of {}'.format(flower_names, classes, probs))
    elif top_k > 1 and top_k <= 102:
        print('\nThe top {} predictions for the given image are: {}'.format(top_k, flower_names))
        print('\nThe predicted flowers have these classes respectively:',classes)
        print('\nThe corresponding probabilities of prediction are:',probs)

if __name__ == "__main__":
    main()
