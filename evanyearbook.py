import os
import cv2
import numpy as np
import tensorflow as tf
import pickle


imgDim = (680,550)
squishDim = (55*2,68*2) 
reshapeDim = (68*2,55*2)
facesFolder = "C:\\Users\\miste\\Desktop\\ProjectNov18\\faces"

faces = os.listdir(facesFolder)

faces = [os.path.join(facesFolder, face) for face in faces]
images = []

for i, face in enumerate(faces):
    print(i, face)
    image = cv2.imread(face, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, squishDim)


    image = image.flatten()

    image = image / 256
    images.append(image)

images = np.asarray(images, dtype=np.float32)


#cv2.imshow("Face 0", np.resize(images[0], reshapeDim))
#cv2.waitKey(0)

graph = tf.Graph()
constraintNodes = 15
with graph.as_default():

    trainData = tf.constant(images, tf.float32)
    p5 = tf.constant(0.5, tf.float32)
    keepRate = tf.placeholder(tf.float32)

    wt1 = tf.Variable(tf.truncated_normal([14960, 1000], dtype=tf.float32, stddev=.1))
    b1 = tf.Variable(tf.zeros([1000], dtype=tf.float32))

    wt2 = tf.Variable(tf.truncated_normal([1000, 100], dtype=tf.float32, stddev=.1))
    b2 = tf.Variable(tf.zeros([100], dtype=tf.float32))

    wt3 = tf.Variable(tf.truncated_normal([100, constraintNodes], dtype=tf.float32, stddev=.1))
    b3 = tf.Variable(tf.zeros([constraintNodes], dtype=tf.float32))

    wt4 = tf.Variable(tf.truncated_normal([constraintNodes, 100], dtype=tf.float32, stddev=.1))
    b4 = tf.Variable(tf.zeros([100], dtype=tf.float32))

    wt5 = tf.Variable(tf.truncated_normal([100, 1000], dtype=tf.float32, stddev=.1))
    b5 = tf.Variable(tf.zeros([1000], dtype=tf.float32))

    wt6 = tf.Variable(tf.truncated_normal([1000, 14960], dtype=tf.float32, stddev=.1))
    b6 = tf.Variable(tf.zeros([14960], dtype=tf.float32))

    l1_logits = tf.matmul(trainData, wt1) + b1
    l1_a = tf.nn.leaky_relu(l1_logits)
    l1_drop = tf.nn.dropout(l1_a, keep_prob=keepRate)

    l2_logits = tf.matmul(l1_drop, wt2) + b2
    l2_a = tf.nn.leaky_relu(l2_logits)
    l2_drop = tf.nn.dropout(l2_a, keep_prob=keepRate)

    l3_logits = tf.matmul(l2_drop, wt3) + b3
    l3_a = tf.nn.sigmoid(l3_logits)
    #l3_drop = tf.nn.dropout(l3_a, keep_prob=keepRate)

    l4_logits = tf.matmul(l3_a, wt4) + b4
    l4_a = tf.nn.leaky_relu(l4_logits)
    l4_drop = tf.nn.dropout(l4_a, keep_prob=keepRate)

    l5_logits = tf.matmul(l4_drop, wt5) + b5
    l5_a = tf.nn.leaky_relu(l5_logits)
    l5_drop = tf.nn.dropout(l5_a, keep_prob=keepRate)

    l6_logits = tf.matmul(l5_drop, wt6) + b6
    predictions = tf.nn.sigmoid(l6_logits)

    #loss = tf.reduce_mean(tf.square(predictions - trainData))

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=l6_logits, labels=trainData)) +  tf.reduce_mean(tf.squared_difference(l3_a, p5))

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = .01).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(learning_rate = .00001).minimize(loss)


    global_step = tf.Variable(0, trainable=False)
    decayLR = .00001 * ((.9) ** (global_step/75000))
    optimizer = tf.train.AdamOptimizer(learning_rate=decayLR).minimize(loss, global_step=global_step)




iterations = 1000000
#iterations = 1000

checkpointfile = 'C:\\Projects\\autoencoder\\model\\textbookAE_8NODES.ckpt'
with tf.Session(graph=graph) as sess:

    saver = tf.train.Saver()
    tf.global_variables_initializer().run()

    #ckpt = tf.train.get_checkpoint_state('C:\\Projects\\autoencoder\\model') for if there is a checkpoint
    saver.restore(sess, checkpointfile)



    trainTrainDict = {keepRate: 0.82}
    trainEvalDict = {keepRate: 1}

    nus = sess.run([l3_a], feed_dict=trainEvalDict)
    print('Evan', nus[0][109])
    print('Rach', nus[0][110])


    w4, bi4, w5, bi5, w6, bi6 = sess.run([wt4, b4, wt5, b5, wt6, b6], feed_dict=trainEvalDict)

    with open('data.pickle', 'wb') as f:
        yb_wts = {'wt4': w4, 'b4': bi4,
                    'wt5': w5, 'b5': bi5,
                    'wt6': w6, 'b6': bi6}
        pickle.dump(yb_wts, f)


    for i in range(iterations):
        sess.run([optimizer], feed_dict=trainEvalDict)

        if (i % 1000 == 0):
            l, pds = sess.run([loss,predictions], feed_dict=trainEvalDict)

            print('Iteration: ', i, 
                    '\n\tLoss: ', l)
            
            cv2.imshow("Face Jose", np.resize(pds[0], reshapeDim))
            cv2.imshow("Face Jacky", np.resize(pds[1], reshapeDim))
            cv2.imshow("Face Evan", np.resize(pds[109], reshapeDim))
            cv2.imshow("Face Rachael", np.resize(pds[110], reshapeDim))
            cv2.imshow("Face BG", np.resize(pds[279], reshapeDim))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                    
        if (i % 25000 == 0):
            saver.save(sess,checkpointfile)

    print(saver.save(sess, checkpointfile))



'''


        if (i % 3000 == 0):
            cv2.imshow("Face Jose", np.resize(pds[0], reshapeDim))
            cv2.imshow("Face Jacky", np.resize(pds[1], reshapeDim))
            cv2.imshow("Face Evan", np.resize(pds[109], reshapeDim))
            cv2.imshow("Face Rachael", np.resize(pds[110], reshapeDim))
            cv2.imshow("Face BG", np.resize(pds[279], reshapeDim))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
  
'''

        

    