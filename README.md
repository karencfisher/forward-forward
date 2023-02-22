# forward-forward

Experimenting with the forward-foward algorithm based on https://keras.io/examples/vision/forwardforward/, 
implementing Geoffrey Hinton's paper https://arxiv.org/abs/2212.13345

## Image classifier

Having built a working FF MLP, the next step is 
to build an image classifier. The first attempt is to use a
forward-forward network as the classifier on top of a
pretrained CNN, such as Mobilenet_v2. 

Currently, the training of the model
appears to be functioning.
