'''
Transfer learning - Using already trained modules (inceprtion v4 , resnet ,vgg net etc) for different use cases . 

Here the already trained convolutional neural net will work as a feature extractor. 

Procedure :

1.Go through the structure . Identify the name scopes of convolutional , max pool and fully connected layers 
2.Identify the name scopes of the fully connected layers (logits layer)
3.Dowload the set of weight from pretrained CNN as a .ckpt 
4.Load the graph (Tensorflow slim has difffernet achitectural graphs) (Inside the "net" folder)
5.Inside a session load the pre- train weights (remeber to exclude the weight set of fully connected layer)
6.Define your new number of class labels . (Here you are creating new softmax layer and fully connected layers acoding to the number of classes)

Transfer Learning  - 

Normally above mentioned achitectures are well trained for the Imagenet data set which has 1000 classes . So they have trained those networks for few weeks with huge GPUs. With the amount of computational power and amount of data they will eventually come near to a Global Minima . 

So what if we have a problem where we have only 2 or 4 classes. Lets say a binary classification problem where we want to classify our images in to images with christmas background or not. 

Do we have to trained from the scratch ? 

Well we can. But we refrain from doing that. Why?
1.Computational cost
2.Time consuming 
3.We might not have huge amound of data . So we can easily overfit these huge modules 

So what can we do?
We can use alraedy trained modulesas a feature extractor. 

differnt layers of CNN are suppose to extract amazing features. Normally first few would extract features which are more relevent to pixel distribution like edges , fading effects etc (low level features ) and final layers will try to extract contextual features. So CNNs can genaralize the image classification part locally. This is the whole point of end to end training or deep learning 

So we can take the layers up to final max pool layer of a pre trained CNN as a good feature extractor . Noramally we do not take the fully conected layer with batch normalization because it can bit mess with the distribution of data . 

So what we remove fully conected layer and softmax layer. Then we replaced with new layer which can output the score relevent to new number of classes. If only four classes we have 4 nodes in the softmax score layer. 

The training procedure 

First we will load the aready trained checkpoints (.ckpt files in TF) for all the weights except the weights for obove mentioned FC layers. 

Then first we will train only the weight set of newly added fully connected layers. So all the other layers will act as a feature extractor .After we getting fair amount of accuarcy we will stop it.

Then we will train the CNN net with all the weights.(FC layer weight + Pre trained weights from convolutional layers) . But in this step we should be really careful with the Learning Rate 

Why?

Noramlly above mentioned networks are trained for a Global Minima or near Golbal Minima. So we should be careful with playing around them. So it is advised to use a very low learning rate. Like 0.0001


'''


set -e

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=datasets/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=datasets/custom-models/inception_v3

# Where the dataset is saved to.
DATASET_DIR=datasets/custom

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
  tar -xvf inception_v3_2016_08_28.tar.gz
  mv inception_v3.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt
  rm inception_v3_2016_08_28.tar.gz
fi

# Download the dataset
python3 download_and_convert_data.py \
  --dataset_name=custom \
  --dataset_dir=${DATASET_DIR}

# Fine-tune only the new layers for 1000 steps.
python3 train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=custom \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=1000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python3 eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=custom \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3

# Fine-tune all the new layers for 500 steps.
python3 train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=custom \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=500 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python3 eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=custom \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3
