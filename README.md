# High-School-Yearbook-Autoencoder
Compresses high school yearbook photos into a discrete amount of numbers that each represent an abstract feature of the image

***
First Idea: Use all fully connected layers
* __evanyearbook.py__ contains the full neural network
* __evanyearbookdecoder.py__ contains the decoder and a script to generate videos maniplulating 1 of the values at a time
 

  Fully Connected Results:  
  Training
  
  ![](https://github.com/mrthinger/High-School-Yearbook-Autoencoder/blob/master/demo/anntraining.gif)
  
  Me   
  
  ![](https://github.com/mrthinger/High-School-Yearbook-Autoencoder/blob/master/demo/annevan.gif)  
  
  Girl 
  
  ![](https://github.com/mrthinger/High-School-Yearbook-Autoencoder/blob/master/demo/anngirl.gif) 
  
  Me 2 Girl    
  
  ![](https://github.com/mrthinger/High-School-Yearbook-Autoencoder/blob/master/demo/annevan2girl.gif)  
  
***
Next Idea: Use Convolutional layers -> Fully Connected -> Convolutional Transpose

   Convolutional Results:  

   Me    

   ![](https://github.com/mrthinger/High-School-Yearbook-Autoencoder/blob/master/demo/evanconv.gif)  

   Girl   

   ![](https://github.com/mrthinger/High-School-Yearbook-Autoencoder/blob/master/demo/girlconv.gif)  

   Neural Network Graph 

   ![](https://github.com/mrthinger/High-School-Yearbook-Autoencoder/blob/master/demo/conv_graph.png) 

