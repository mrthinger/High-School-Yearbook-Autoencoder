import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm

#image dimensions are 550x680
scale = 1
squishDim = (56*scale,68*scale) 
facesFolder = "C:\\Users\\miste\\Desktop\\ProjectNov18\\testfaces"

faces = os.listdir(facesFolder)

faces = [os.path.join(facesFolder, face) for face in faces]
images = []

for i, face in tqdm(enumerate(faces), total=len(faces), desc='Loading Images'):
    #print(i, face)
    image = cv2.imread(face, cv2.IMREAD_COLOR)
    image = cv2.resize(image, squishDim, interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    #image = image.flatten()

    image = image / 256
    images.append(image)

images = np.asarray(images, dtype=np.float32)


def enlargeImg(img, imgScale=3, toBGR=True):
    if toBGR:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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

        #Summary
        tf.summary.histogram('conv_filter', ft)
        tf.summary.histogram('conv_biases', biases)
        tf.summary.histogram('conv_activations', clayer)

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

        #Summary
        tf.summary.histogram('trans_filter', ft)
        tf.summary.histogram('trans_biases', biases)
        tf.summary.histogram('trans_activations', clayer)

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

        #Summary
        tf.summary.histogram('fc_weights', weights)
        tf.summary.histogram('fc_biases', biases)
        tf.summary.histogram('fc_activations', logits)

        return logits
fc_layer.count = 0


##################
#GRAPH DEFINITION#
##################



c1out = 27
c1filter = 6
c2out = 81
c2filter = 3
c3out = 243
c3filter = 2

fc1out = 1000
fc2out = 250

constraintNodes = 25

regularization = 0.0

graph = tf.Graph()
with graph.as_default():

    trainData = tf.constant(images, tf.float32)
    p5 = tf.constant(0.5, tf.float32)
    regularizationTensor = tf.constant(regularization, tf.float32)

    encodedHolder = tf.placeholder(tf.float32, shape=[None, constraintNodes])
    shouldDecode = tf.placeholder(tf.bool)

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
        constraints = tf.cond(shouldDecode, lambda: encodedHolder, lambda: constraints) 
        fc4 = fc_layer(constraints, constraintNodes, fc2out)
        fc5 = fc_layer(fc4, fc2out, fc1out)
        fc6 = fc_layer(fc5, fc1out, c3Features)

        convInput = tf.reshape(fc6, c3Dims)
        c4 = trans_conv_layer(convInput, c3out, tf.shape(c2), c2out, c3filter, c3filter, 2, 2)
        c5 = trans_conv_layer(c4, c2out, tf.shape(c1), c1out, c2filter, c2filter, 2, 2)
        result_logits = trans_conv_layer(c5, c1out, tf.shape(trainData), 3, c1filter, c1filter, 2, 2, activation=False)
        result = tf.nn.sigmoid(result_logits)

    tf.summary.image('original_img', trainData, max_outputs=4)
    tf.summary.image('reconstuction_img', result, max_outputs=4)

    #loss = tf.reduce_mean(tf.square(result_logits - trainData))

    loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=result_logits, labels=trainData))
             +  (regularizationTensor * tf.reduce_mean(tf.squared_difference(constraints, p5))))


    tf.summary.scalar('loss', loss)

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = .01).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(learning_rate = .00001).minimize(loss)


    global_step = tf.Variable(0, trainable=False, name='global_step')
    decayLR = .00001 * ((.9) ** (global_step/75000))
    optimizer = tf.train.AdamOptimizer(learning_rate=decayLR).minimize(loss, global_step=global_step)



networkNames = ['conv20', 'conv15', 'conv30', 'conv30v2', '4conv25']
networkIndex = 4
checkpointfile = 'C:\\Projects\\ML\\autoencoder\\model\\{}\\CONV_textbookAE.ckpt'.format(networkNames[networkIndex])
with tf.Session(graph=graph) as sess:

    saver = tf.train.Saver()
    tf.global_variables_initializer().run()

    #To restore state
    saver.restore(sess, checkpointfile)

    evan = [0.6284481,  0.43505114, 0.5621194,  0.4470127,  0.56308216, 0.45016283,
        0.5196135,  0.5566554,  0.44984707, 0.5131169,  0.52698505, 0.60874164,
        0.3774942,  0.6907064,  0.5901814,  0.570823,   0.6850461,  0.6540827,
        0.40694383, 0.32772502, 0.35705832, 0.3605123,  0.42152128, 0.46846083,
        0.5406743 ]
    nat = [0.629774,   0.34528926, 0.5801978,  0.5119312,  0.6002703,  0.39179265,
        0.40346754, 0.61298615, 0.57161516, 0.45736697, 0.2752079,  0.5498024,
        0.61861384, 0.3812257,  0.41523,    0.42546853, 0.5350875,  0.35232106,
        0.42722198, 0.5038681,  0.43566447, 0.3439939,  0.43201643, 0.4262252,
        0.48517814]


    frames = 150
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('convEvan.avi',fourcc, 30, (56*3, 68*3), True)

    for i in tqdm(range(len(evan))):
        for f in range(frames):
            
            
            feednums = np.asarray(evan.copy())

            if (feednums[i] == 1.0):
                continue

            feednums[i] = ((f / float(frames))*0.4) + 0.3 # 0 - > 1   #0.3 -> 0.6

            feednums = np.atleast_2d(feednums)

            for a in range(len(images)-1):
                zeros = np.zeros(constraintNodes)
                zeros = np.atleast_2d(zeros)
                feednums = np.vstack((feednums, zeros))


            evalDict = {
                shouldDecode: True,
                encodedHolder: feednums
            }



            #MANIP
            prediction = sess.run([result], feed_dict=evalDict)

            img = enlargeImg(prediction[0][0])

            img = np.multiply(img, 256.0)

            img = img.astype(np.uint8)


            out.write(img)

    out.release()



        


    #TRAIN
    # for i in range(iterations):
    #     sess.run([optimizer], feed_dict=evalDict)
        

    #     if (i % boardLogEvery == 0):
    #         s, step = sess.run([summ, global_step], feed_dict=evalDict)
    #         writer.add_summary(s, step)

                    
    #     if (i % 5000 == 0):
    #         saver.save(sess,checkpointfile)
    #         print('\nCheckpoint Created\n')
    