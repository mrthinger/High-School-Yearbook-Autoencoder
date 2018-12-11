import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm


scale = 1
imgDim = (550,680) 
squishDim = (56*scale,68*scale) 
reshapeDim = (68*scale,56*scale)
facesFolder = "C:\\Users\\miste\\Desktop\\ProjectNov18\\faces"

faces = os.listdir(facesFolder)

faces = [os.path.join(facesFolder, face) for face in faces]
images = []

for i, face in tqdm(enumerate(faces), total=len(faces), desc='Loading Images'):
    #print(i, face)
    image = cv2.imread(face, cv2.IMREAD_COLOR)
    image = cv2.resize(image, squishDim, interpolation=cv2.INTER_AREA)


    #image = image.flatten()

    image = image / 256
    images.append(image)

images = np.asarray(images, dtype=np.float32)


def enlargeImg(img, imgScale=3):
    return cv2.resize(img, (0,0), fx=imgScale, fy=imgScale, interpolation=cv2.INTER_CUBIC)



#cv2.imshow("Face 0", enlargeImg(images[0]))
#cv2.waitKey(0)

def conv_layer(input, inputChannels, outputChannels, filterWidth, filterHeight, strideX, strideY, pooling=True):
    conv_layer.count += 1
    with tf.name_scope('conv_layer_'+ str(conv_layer.count)):

        #filter configuration (shape is from tensorflow docs)
        filterShape = [filterHeight, filterWidth, inputChannels, outputChannels]
        ft = tf.Variable(tf.truncated_normal(filterShape, dtype=tf.float32, stddev=0.1))
        biases = tf.Variable(tf.zeros([outputChannels]), dtype=tf.float32)

        #stride
        strides = [1, strideX, strideY, 1]

        #conv layer
        clayer = tf.nn.conv2d(input, ft, strides, "SAME")

        #add bias
        clayer += biases

        #non-linearity
        clayer = tf.nn.leaky_relu(clayer)

        #default to 2x2 maxpooling
        if pooling:
            clayer = tf.nn.max_pool(clayer,[1,2,2,1],[1,2,2,1],"SAME")

        return clayer
#static tracking of layercount
conv_layer.count = 0

def trans_conv_layer(input, inputChannels, outputShape, outputChannels, filterWidth, filterHeight, strideX, strideY, activation=True):
    trans_conv_layer.count += 1
    with tf.name_scope('trans_conv_layer_'+ str(trans_conv_layer.count)):

        #filter configuration (shape is from tensorflow docs)
        filterShape = [filterHeight, filterWidth, outputChannels, inputChannels]
        ft = tf.Variable(tf.truncated_normal(filterShape, dtype=tf.float32, stddev=0.1))
        biases = tf.Variable(tf.zeros([outputChannels]), dtype=tf.float32)

        #stride
        strides = [1, strideX, strideY, 1]

        #conv layer
        clayer = tf.nn.conv2d_transpose(input,filter=ft,output_shape=outputShape,strides=strides, padding="SAME")

        #add bias
        clayer += biases

        #non-linearity
        if activation:
            clayer = tf.nn.leaky_relu(clayer)

        return clayer
#layer tracking
trans_conv_layer.count = 0

def fc_layer(input, numInput, numOutput, activation=True):
    fc_layer.count += 1
    with tf.name_scope('fc_layer_'+ str(fc_layer.count)):
        weights = tf.Variable(tf.truncated_normal([numInput, numOutput], dtype=tf.float32, stddev=.1))
        biases = tf.Variable(tf.zeros([numOutput], dtype=tf.float32))

        logits = tf.matmul(input, weights) + biases

        if activation:
            logits = tf.nn.leaky_relu(logits)

        return logits
fc_layer.count = 0


##################
#GRAPH DEFINITION#
##################



c1out = 9
c1filter = 3
c2out = 18
c2filter = 3
c3out = 36
c3filter = 3

fc1out = 700
fc2out = 100

constraintNodes = 20

regularization = 0.1

graph = tf.Graph()
with graph.as_default():

    trainData = tf.constant(images, tf.float32)
    p5 = tf.constant(0.5, tf.float32)
    regularizationTensor = tf.constant(regularization, tf.float32)

    #Encoder
    with tf.name_scope('Encoder'):
        #convolutional
        c1 = conv_layer(trainData, inputChannels=3, outputChannels=c1out, filterWidth=c1filter,filterHeight=c1filter,strideX=1,strideY=1)
        c2 = conv_layer(c1, inputChannels=c1out, outputChannels=c2out, filterWidth=c2filter,filterHeight=c2filter,strideX=1,strideY=1)
        c3 = conv_layer(c2, inputChannels=c2out, outputChannels=c3out, filterWidth=c3filter,filterHeight=c3filter,strideX=1,strideY=1)

        #reshape then run through fully connected

        #this will return [batchsize, imgW, imgH, layers] so number of features is
        # imgW * imgH * layers
        c3Dims = c3.get_shape()
        c3Features = c3Dims[-3:].num_elements() #num elements does what was commented above

        fcInput = tf.reshape(c3, [-1, c3Features]) #this is 2268 features with c1out=9 c2out=18 c3out=36

        fc1 = fc_layer(fcInput, c3Features, fc1out)
        fc2 = fc_layer(fc1, fc1out, fc2out)

        constraint_logits = fc_layer(fc2, fc2out, constraintNodes, activation=False)
        constraints = tf.nn.sigmoid(constraint_logits) 

    #decoder
    with tf.name_scope('Decoder'):
        fc4 = fc_layer(constraints, constraintNodes, fc2out)
        fc5 = fc_layer(fc4, fc2out, fc1out)
        fc6 = fc_layer(fc5, fc1out, c3Features)

        convInput = tf.reshape(fc6, c3Dims)
        c4 = trans_conv_layer(convInput, c3out, tf.shape(c2), c2out, c3filter, c3filter, 2, 2)
        c5 = trans_conv_layer(c4, c2out, tf.shape(c1), c1out, c2filter, c2filter, 2, 2)
        result_logits = trans_conv_layer(c5, c1out, tf.shape(trainData), 3, c1filter, c1filter, 2, 2, activation=False)
        result = tf.nn.sigmoid(result_logits)

    #loss = tf.reduce_mean(tf.square(predictions - trainData))

    loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=result_logits, labels=trainData))
             +  (regularizationTensor * tf.reduce_mean(tf.squared_difference(constraints, p5))))

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = .01).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(learning_rate = .00001).minimize(loss)


    global_step = tf.Variable(0, trainable=False)
    decayLR = .00001 * ((.9) ** (global_step/75000))
    optimizer = tf.train.AdamOptimizer(learning_rate=decayLR).minimize(loss, global_step=global_step)




iterations = 1000000
statusEvery = 21000
#iterations = 1000

checkpointfile = 'C:\\Projects\\ML\\autoencoder\\model\\CONV_textbookAE.ckpt'
with tf.Session(graph=graph) as sess:

    saver = tf.train.Saver()
    tf.global_variables_initializer().run()

    #To restore state
    saver.restore(sess, checkpointfile)


    pbar = tqdm(total=statusEvery, desc='Training')

    for i in range(iterations):
        pbar.update(1)
        sess.run([optimizer])
        
        if (i % statusEvery == 0):
            l, pds = sess.run([loss,result])

            print('\nIteration: ', i, 
                    '\n\tLoss: ', l)
            
            #imshow
            cv2.imshow("Face Jacky", enlargeImg(pds[1]))
            cv2.imshow("Face BG", enlargeImg(pds[279]))
            cv2.imshow("Face Jose", enlargeImg(pds[0]))
            cv2.imshow("Face Rachael", enlargeImg(pds[110]))
            cv2.imshow("Face Evan", enlargeImg(pds[109]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            #pbar reset
            pbar.close()
            pbar = tqdm(total=statusEvery, desc='Training')
                    
        if (i % 5000 == 0):
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

        

    