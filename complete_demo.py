from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Alignment of Images in DB

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
from time import sleep
import glob

output_dir_path = ('/Users/parth/facenet/demo_dataset_align')
output_dir = os.path.expanduser(output_dir_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

datadir = ('/Users/parth/facenet/demo_dataset')
dataset = facenet.get_dataset(datadir)

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, ('/Users/parth/facenet/src/align'))

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
margin = 44
image_size = 182

# Add a random key to the filename to allow alignment using multiple processes
random_key = np.random.randint(0, high=99999)
bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
print('Goodluck')

with open(bounding_boxes_filename, "w") as text_file:
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename + '.png')
            print(image_path)
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(image_path)
                    print('read data dimension: ', img.ndim)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim < 2:
                        print('Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                        print('to_rgb data dimension: ', img.ndim)
                    img = img[:, :, 0:3]
                    print('after data dimension: ', img.ndim)

                    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print('detected_face: %d' % nrof_faces)
                    

                    if nrof_faces>0:
                        det = bounding_boxes[:,0:4]
                        det_arr = []
                        img_size = np.asarray(img.shape)[0:2]
                        if nrof_faces>1:
                            if args.detect_multiple_faces:
                                for i in range(nrof_faces):
                                    det_arr.append(np.squeeze(det[i]))
                            else:
                                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                img_center = img_size / 2
                                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                det_arr.append(det[index,:])
                        else:
                            det_arr.append(np.squeeze(det))

                        for i, det in enumerate(det_arr):
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0]-margin/2, 0)
                            bb[1] = np.maximum(det[1]-margin/2, 0)
                            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                            nrof_successfully_aligned += 1
                            
                            misc.imsave(output_filename, scaled)
                            text_file.write('%s %d %d %d %d\n' % (output_filename, bb[0], bb[1], bb[2], bb[3]))
                    else:
                        print('Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))




                    # if nrof_faces > 0:
                    #     det = bounding_boxes[:, 0:4]
                    #     img_size = np.asarray(img.shape)[0:2]
                    #     if nrof_faces > 1:
                    #         bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    #         img_center = img_size / 2
                    #         offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    #                              (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    #         offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    #         index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                    #         det = det[index, :]
                    #     det = np.squeeze(det)
                    #     bb_temp = np.zeros(4, dtype=np.int32)

                    #     bb_temp[0] = det[0]
                    #     bb_temp[1] = det[1]
                    #     bb_temp[2] = det[2]
                    #     bb_temp[3] = det[3]

                    #     cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                    #     scaled_temp = misc.imresize(cropped_temp, (image_size, image_size), interp='bilinear')

                    #     nrof_successfully_aligned += 1
                    #     misc.imsave(output_filename, scaled_temp)
                    #     text_file.write('%s %d %d %d %d\n' % (output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))
                    # else:
                    #     print('Unable to align "%s"' % image_path)
                    #     text_file.write('%s\n' % (output_filename))

print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


# Calculating Embeddings for Images

import tensorflow as tf
import numpy as np
import argparse
import facenet
import detect_face
import os
import sys
import math
import pickle
from sklearn.svm import SVC
import glob


with tf.Graph().as_default():

    with tf.Session() as sess:

        datadir = '/Users/parth/facenet/demo_dataset_align'
        dataset = facenet.get_dataset(datadir)
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))

        print('Loading feature extraction model')
        modeldir = '/Users/parth/facenet/models/20180402-114759/20180402-114759.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # Create a list of class names
        class_names = os.listdir('/Users/parth/facenet/demo_dataset_align')
        #---------change here for deletion-----------if class_names
        if '.DS_Store' in class_names:
            class_names.remove('.DS_Store')
        matching = [s for s in class_names if "bounding_boxes_" in s]
        for j in range(len(matching)):
            class_names.remove(matching[j])
            os.remove('/Users/parth/facenet/demo_dataset_align/%s'%matching[j])
            
        class_names.sort()
        print(class_names)

        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        batch_size = 1000
        image_size = 160
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
            #emb_array=[sess.run(embeddings, feed_dict=feed_dict)]
            #class_names=[class_names]
            #det_array=[class_names, emb_array]

        classifier_filename = '/Users/parth/Downloads/real-time-deep-face-recognition-master/complete-face-recog/demo_embeddings.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)

        # # Train classifier
        # print('Training classifier')
        # model = SVC(kernel='linear', probability=True)
        # model.fit(emb_array, labels)
        #class_names = [cls.name.replace('_', '') for cls in dataset]
        # Saving classifier model

        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((class_names,emb_array), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)
        print('Goodluck')


# Taking Image and Recognizing in real-time

import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import facenet
import detect_face
import os
from os.path import join as pjoin
import sys
import time
import copy
import math
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    sess=tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, '/Users/parth/facenet/src/align')

        minsize = 200  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160


        print('Loading feature extraction model')
        modeldir = '/Users/parth/facenet/models/20180402-114759/20180402-114759.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename = 'demo_embeddings.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (class_names, emb_array) = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)

        video_capture = cv2.VideoCapture(0)
        c = 0

        # #video writer
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # out = cv2.VideoWriter('3F_0726.avi', fourcc, fps=30, frameSize=(640,480))

        print('Start Recognition!')
        prevTime = 0
        while True:
            ret, frame = video_capture.read()
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

            curTime = time.time()    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print(nrof_faces)
                print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    for i in range(nrof_faces):
                        
                        emb_darray = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is inner of range!')
                            continue
                        #Preprocessing detected image
                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[0] = facenet.flip(cropped[0], False)
                        scaled.append(misc.imresize(cropped[0], (image_size, image_size), interp='bilinear'))
                        scaled[0] = cv2.resize(scaled[0], (input_image_size,input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled[0] = facenet.prewhiten(scaled[0])
                        scaled_reshape.append(scaled[0].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[0], phase_train_placeholder: False}
                        emb_darray[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                        
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20
                        
                        # with open('db_embeddings.pkl', 'rb') as handle:
                        #     dbpkl = pickle.load(handle)

                        min_dist=10
                        #class_names_rev=class_names[::-1]
                        #print("The class names found out are: ",class_names)
                        for i in range(len(class_names)):
                            dist = np.linalg.norm(emb_darray-emb_array[i])
                            print('distance for %s is %s' %(class_names[i], dist))
                            if dist < min_dist:
                                min_dist = dist
                                identity = class_names[i]
                        if min_dist > 0.90:
                            identity = "Unknown!"
                            #print ("Unknown!")
                        else:
                            print("Welcome! ", identity)
                            
                        
                        cv2.putText(frame, identity, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                             1, (0, 0, 255), thickness=1, lineType=2)
                else:
                    print('Unable to align')

            sec = curTime - prevTime
            prevTime = curTime
            fps = 1 / (sec)
            str = 'FPS: %2.3f' % fps
            text_fps_x = len(frame[0]) - 150
            text_fps_y = 20
            cv2.putText(frame, str, (text_fps_x, text_fps_y),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
            # c+=1
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        
        video_capture.release()
        # #video writer
        # out.release()
        cv2.destroyAllWindows()