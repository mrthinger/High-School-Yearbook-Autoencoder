# High-School-Yearbook-Autoencoder
Compresses high school yearbook photos into a discrete amount of numbers that each represent an abstract feature of the image

***
First Idea: Use all fully connected layers
* __evanyearbook.py__ contains the full neural network
* __evanyearbookdecoder.py__ contains the decoder and a script to generate videos maniplulating 1 of the values at a time
  Fully Connected Results:
  Training    
![training](demo\anntraining.gif)
  Me    
![Me](demo\annevan.gif)
  Girl    
![girl](demo\anngirl.gif)
***
Next Idea: Use Convolutional layers -> Fully Connected -> Convolutional Transpose
  Convolutional Results:
  Me    
![Me conv](demo\evanconv.gif)
  Girl    
![girl conv](demo\girlconv.gif)
  Neural Network Graph    
![conv graph](demo\conv_graph.png)

Demo video coming soon.
