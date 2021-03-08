from __future__ import division, absolute_import, print_function

import streamlit as st
import cv2
import pandas as pd
import os
import argparse
import numpy as np
import tensorflow as tf

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """
    text = pytesseract.image_to_string(Image.open(filename))  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    text = text.replace('\n', ' ')
    return text

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.io.decode_png(file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.io.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.io.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.io.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.compat.v1.Session()
    result = sess.run(normalized)

    return result

def load_labels(label_file): 
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

style = """
    <style>
        .title-app, sub-title-app {
            text-align: center;
        }
        .title-app {
            margin-top: -30px !important;
        }
        .sub-title-app {
            margin-top: -10px !important;
            margin-bottom: 20px !important;
        }
        button {
            width: 100% !important;
        }
        img {
            text-align: center !important;
        }
        div[data-testid="caption"] {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .caption {
            text-align: center;
            font-size: 15px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin-bottom: 10px;
        }
        input {
            border: 1px solid rgb(185, 185, 185) !important;
        }
    </style>
"""

# SIDEBAR WIDGETS
st.sidebar.markdown(style, unsafe_allow_html=True)
st.sidebar.markdown('<h1 class="title-app">LAOD</h1>', unsafe_allow_html=True)
st.sidebar.markdown('<h3 class="sub-title-app">Library Automatic Object Detection</h3>', unsafe_allow_html=True)
face = st.sidebar.file_uploader('Upload the face image here')
book = st.sidebar.file_uploader('Upload the book cover image here')
incorrect = st.sidebar.checkbox('The title of the book is printed incorrectly')
if incorrect:
    new_title = st.sidebar.text_input('Input the title of the book here')
predict = st.sidebar.button('Start Prediction')

# MAIN PAGE
st.info('Hi there, all resources that have you been uploaded will be shown here.')
col1, col2 = st.beta_columns(2)

if face:
    with open(os.path.join("test",'test.jpg'),"wb") as f: 
      f.write(face.getbuffer())         
    col1.success("Face image added successfully!")
    col1.image(face, width=300, caption=face.name, use_column_width=True)
if book:
    with open(os.path.join("assets/books",'test.jpg'),"wb") as f: 
      f.write(book.getbuffer())         
    col2.success("Book cover image added successfully!")
    col2.image(book, width=300, caption=book.name, use_column_width=True)
if predict:
    st.markdown('<div class="caption">Please wait, this process takes a minute...</div>', unsafe_allow_html=True)
    pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'
    path = "assets/books/test.jpg"
    title = ''

    if __name__ == "__main__":
        file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
        model_file = \
            "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
        label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
        input_height = 299
        input_width = 299
        input_mean = 0
        input_std = 255
        input_layer = "input"
        output_layer = "InceptionV3/Predictions/Reshape_1"

        model_file = "assets/retrained_graph.pb"
        file_name = "test/test.jpg"
        input_layer = "Placeholder"
        label_file = "assets/retrained_labels.txt"
        output_layer = "final_result"

        graph = load_graph(model_file)
        t = read_tensor_from_image_file(
            file_name,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        with tf.compat.v1.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)

        if incorrect:
            title = new_title
        else:
            title = ocr_core(path)

        for i in top_k:
            st.success("The student ID who will borrow the books : "+labels[i])
            st.success("The title of the book that is will be borrowed : "+title)
            break
        