import tensorflow as tf
import os
import hashlib
import yaml
import numpy as np
import cv2 as cv
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont

tf.enable_eager_execution()

IMG_WIDTH = 128
IMG_HEIGHT = 128
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.heic'}

data_path = os.path.abspath("./data")
face_cascade_name = os.path.join(data_path, "opencv/data/haarcascades/haarcascade_frontalface_alt.xml")
face_cascade = cv.CascadeClassifier(face_cascade_name)
train_data_path = os.path.join(data_path, "database_imgs")
train_names = []


def extract_face(img_path):
    """
    Extracts the list of faces from an input images
    :param img_path: the path to image file. The input-file's extension must be in ALLOWED_EXTENSIONS
    :return: 2 arrays: first array contains the image of faces, the second one contains the location (x, y, w, h)
    """
    frame_ori = cv.imread(img_path)
    frame_gra = cv.cvtColor(frame_ori, cv.COLOR_BGR2GRAY)
    frame_gra = cv.equalizeHist(frame_gra)
    faces = face_cascade.detectMultiScale(frame_gra)
    img_faces = []
    img_frame = []
    for (x, y, w, h) in faces:
        frame_face = frame_gra[y:y + h, x:x + w]
        frame_face = cv.resize(frame_face, dsize=(IMG_HEIGHT, IMG_WIDTH))
        img_faces.append((frame_face / 127.5) - 1)
        img_frame.append([x, y, w, h])
    return img_faces, img_frame


def load_train_data(train_dir):
    """
    Load the content of training dir. It extracts the faces then create the pair of (data, label)
    :param train_dir: the path to training dir
    :return: an array which contains all pairs of (data, label)
    """
    faces_data = []
    if train_dir:
        train_dir = train_dir.strip()
    if not train_dir or not os.path.isdir(train_dir):
        train_dir = "."
    train_dir = os.path.abspath(train_dir)
    for per in os.listdir(train_dir):
        sub_dir = os.path.join(train_dir, per)
        if os.path.isdir(sub_dir):
            for img in os.listdir(sub_dir):
                img_path = os.path.join(sub_dir, img)
                if os.path.isfile(img_path):
                    img_ext = os.path.splitext(img_path)[-1]
                    if img_ext.lower() in ALLOWED_EXTENSIONS:
                        faces = extract_face(img_path)[0]
                        for f in faces:
                            faces_data.append([f, per])
    return faces_data


def build_model():
    """
    Builds the tensorflow model
    :return: the tf-model
    """
    global train_names
    load_faces = load_train_data(train_data_path)
    train_images = np.array([x[0] for x in load_faces], dtype=np.float32)
    train_names = list(set([x[1] for x in load_faces]))
    train_labels = np.array([train_names.index(x[1]) for x in load_faces], dtype=np.int)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(len(train_names), activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)
    return model


def recognize_face(model, input_img):
    """
    Recognize the name from the input image
    :param model: the tf-model
    :param input_img: the path to input image
    :return: the list of names and their location
    """
    faces = extract_face(input_img)
    face_list = []
    for i, f in enumerate(faces[0]):
        predictions_single = model.predict(np.expand_dims(f, 0))
        label = np.argmax(predictions_single[0])
        face_list.append([train_names[label], faces[1][i]])
    return face_list


def init_program(rebuild_flag=None):
    """
    Initialize program
    :param rebuild_flag: force to re-build model
    :return: the tf model
    """
    global train_names
    key1 = "data_hash"
    key2 = "model_name"
    key3 = "name_list"
    new_model = None
    data_yaml = os.path.join(data_path, "data.yaml")
    data_hash = ""
    old_model_file = ""
    current_database_hash = hash_md5_dir(train_data_path)
    if rebuild_flag:
        new_model = True
    else:
        if os.path.exists(data_yaml):
            data_config = yaml.load(open(data_yaml, "rt"))
            if key1 in data_config and key2 in data_config and key3 in data_config:
                data_hash = data_config[key1]
                old_model_file = data_config[key2]
                train_names = data_config[key3]
            else:
                new_model = True
        else:
            new_model = True
        if not new_model:
            if not data_hash or data_hash != current_database_hash:
                new_model = True
    if not new_model and os.path.isfile(old_model_file):
        # load model here
        model = keras.models.load_model(old_model_file)
    else:
        if os.path.isfile(old_model_file):
            os.remove(old_model_file)
        model = build_model()
        new_model_file = os.path.join(data_path, current_database_hash + ".model")
        data_config = {
            key1: current_database_hash,
            key2: new_model_file,
            key3: train_names
        }
        try:
            model.save(new_model_file)
        except Exception as e:
            print("Warning: error as saving tf-model. (%s)" % e)
        with open(data_yaml, 'w') as config_file:
            yaml.dump(data_config, config_file)
    return model


def hash_md5_dir(dir_path):
    """
    Calculates the mMD5 hash of a folder.
    :param dir_path: the os path to folder
    :return: MD5 string
    """
    str_md5 = ""
    if os.path.isdir(dir_path):
        h = hashlib.sha1()
        for name in sorted(os.listdir(dir_path)):
            path_name = os.path.join(dir_path, name)
            file_hash = ""
            if os.path.isdir(path_name):
                file_hash = hash_md5_dir(path_name)
            else:
                file_hash = hash_md5_file(path_name)
            h.update(file_hash.encode('utf-8'))
        str_md5 = h.hexdigest()
    return str_md5


def hash_md5_file(file_path):
    """
    Calculates the mMD5 hash of file.
    :param file_path: the os path to file
    :return: MD5 string
    """
    str_md5 = ""
    if os.path.isfile(file_path):
        str_md5 = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
    return str_md5


def display_result(img_path, faces):
    """
    Draw the images and recognized faces.
    :param img_path: path of image
    :param faces: recognized faces.
    :return: nothing
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    txt_font = ImageFont.truetype("arial.ttf", 16)
    for name, (x, y, w, h) in faces:
        draw.rectangle(((x, y), (x+w, y+h)), outline=(0, 255, 0), width=8)
        text_w, text_h = draw.textsize(name, font=txt_font)
        draw.rectangle(((x, y + h), (x + text_w + 10, y + h + text_h + 10)), fill=(0, 255, 0), outline=(0, 255, 0))
        draw.text((x + 5, y + h + text_h - 5), name, fill=(255, 255, 255, 255), font=txt_font)
    del draw
    pil_image.show()
    return


def main():
    model = init_program()
    data_test = "./data/test_img/Feb 26- 2019 at 5-08 PM.HEIC"
    faces = recognize_face(model, data_test)
    for f in enumerate(faces):
        print(f)
    display_result(data_test, faces)
    return


main()
