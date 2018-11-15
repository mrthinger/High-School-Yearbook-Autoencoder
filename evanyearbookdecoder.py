import os
import cv2
import numpy as np
import tensorflow as tf
import pickle


scale = 4
imgDim = (680,550)
squishDim = (55*2,68*2) 
reshapeDim = (68*2,55*2)
growDim = (55*scale,68*scale)



with open('data.pickle', 'rb') as f:
    yb_wts = pickle.load(f)

graph = tf.Graph()
constraintNodes = 15
with graph.as_default():
    
    nums35 = tf.placeholder(tf.float32, shape=(1,constraintNodes))

    wt4 = tf.constant(yb_wts['wt4'], dtype=tf.float32)
    b4 = tf.constant(yb_wts['b4'], dtype=tf.float32)

    wt5 = tf.constant(yb_wts['wt5'], dtype=tf.float32)
    b5 = tf.constant(yb_wts['b5'], dtype=tf.float32)

    wt6 = tf.constant(yb_wts['wt6'], dtype=tf.float32)
    b6 = tf.constant(yb_wts['b6'], dtype=tf.float32)


    l4_logits = tf.matmul(nums35, wt4) + b4
    l4_a = tf.nn.leaky_relu(l4_logits)

    l5_logits = tf.matmul(l4_a, wt5) + b5
    l5_a = tf.nn.leaky_relu(l5_logits)

    l6_logits = tf.matmul(l5_a, wt6) + b6
    predictions = tf.nn.sigmoid(l6_logits)




iterations = 1000000
#iterations = 1000
with tf.Session(graph=graph) as sess:

    tf.global_variables_initializer().run()

    evannums = [0.5633238,  0.51287085, 0.43402222, 0.49466243, 0.4538433, 0.45387006,
                0.5457608,  0.5050695,  0.5516213,  0.5419111,  0.5442175,  0.49160782,
                0.43522444, 0.4943684,  0.52396894]


    racNums=[0.50854343, 0.55007267, 0.5540648,  0.48120978, 0.45412308, 0.47189927,
             0.44210568, 0.58722657, 0.41266158, 0.5093304,  0.45152256, 0.49140742,
             0.4704169,  0.5132763,  0.5029932]

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('etor.avi',fourcc, 30, growDim, 0)

    totalFrames = 150
    for j in range(constraintNodes):
        for i in range (totalFrames):
            e = np.asarray(evannums.copy())
            r = np.asarray(racNums.copy())
            part = float(i+1)/float(totalFrames)
            d = np.subtract(r, e)
            d = np.multiply(d, part)
            modnums = np.add(e,d)

            #r[j] =  (0.35) +  (part * 0.35)
            feednums = np.asarray(modnums)

            feednums = np.atleast_2d(feednums)

            trainEvalDict = {nums35:feednums}

            pds = sess.run([predictions], feed_dict=trainEvalDict)


            img = np.resize(pds[0][0], reshapeDim)

            img = np.multiply(img, 256.0)

            img = img.astype(np.uint8)

            img = cv2.resize(img, (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC) 
            out.write(img)

            #cv2.imshow("Face Evan", img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

    out.release()