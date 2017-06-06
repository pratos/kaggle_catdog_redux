from keras import applications
from keras.models import Model, Sequential
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
import h5py


def baseline(train_src, validation_src, img_width, img_height, non_train_layers, batch_size, epochs):
	"""Baseline Keras model using VGG16
	=============================================
	Input:
	--------
	* train_src : Training folder location
	* img_width : Width of the image that you want
	* img_height : Height of the image that you want
	* non_train_layers: No of layers that you want to make as non trainable
	                    (layer.trainable = False)
	* batch_size : Batch Size
	* epochs: number of epochs

	Output:
	-------- 
	* Keras model is saved.

	"""
    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    
    for layer in model.layers[:non_train_layers]:
        layer.trainable = False
    
    print("Layers made non trainable...")

    #Adding custom Layers 
    x = model.output
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(16, activation="softmax")(x)
    
    # creating the final model 
    model_final = Model(input = model.input, output = predictions)
    
    print("The final model has been created...")

    # compile the model 
    model_final.compile(loss = "categorical_crossentropy", optimizer = \
                        optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

    print("The model has been compiled")

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            #'data/train',  # this is the target directory
            train_src,
            target_size=(256, 256),  # all images will be resized to 256x256
            batch_size=batch_size,
            class_mode='categorical')  

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            validation_src,
            target_size=(256, 256),
            batch_size=batch_size,
            class_mode='categorical')
    
    print("Fitting the generator...")

    model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
    
    model.save_weights('first_try.h5')  # always save your weights after training or during training

    print("The Keras model has been saved successfully...")
