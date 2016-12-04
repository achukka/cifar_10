# LeNet model
class CifarCNN:
    @staticmethod
    def build(width, height, depth, classes, 
              kernel_size=(3, 3), pool_size=(2,2), weightsPath=None):
        # Initialzing the model
        model = Sequential()
        
        ''' First set of CONV => RELU => POOL => DROPOUT'''
        # Add Convolution Layers '32' filters and receptive filed of size (5 x 5)
        ''' Note - You need to provide the 'input_shape' only for the first layer in keras
                  BORDER_MODE - 'valid' => No zero padding to the input,\n",
                                'same'  => Padding such that input_dim=output_dim  '''
        model.add(Convolution2D(32, kernel_size[0], kernel_size[0], border_mode='same', 
                                input_shape=(depth, height, width)))
        # Add Recitified Linear Units
        model.add(Activation('relu'))        
        # Add Convolution Layers '32' filters and receptive field of dims "kernel_size"
        model.add(Convolution2D(32, kernel_size[0], kernel_size[1],border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))
        # Then Max Pooling with 'pool size -"pool_size"', Default Stride is same as "pool_size"
        model.add(MaxPooling2D(pool_size=pool_size))
        
        ''' Second Set of CONV => RELU => POOL '''
        # Add Convolution Layers '64' filters and receptive field of dims "kernel_size"
        model.add(Convolution2D(64, kernel_size[0], kernel_size[1], border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))        
        # Add Convolution Layers '64' filters and receptive filed of size (5 x 5)
        model.add(Convolution2D(64, kernel_size[0], kernel_size[1],border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))
        # Then Max Pooling with 'pool size -"pool_size"', Default Stride is same as "pool_size"
        model.add(MaxPooling2D(pool_size=pool_size))
        
        
        ''' Third Set of CONV => RELU => POOL '''
        # Add Convolution Layers '128' filters and receptive field of dims "kernel_size"
        model.add(Convolution2D(128, kernel_size[0], kernel_size[1], border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))        
        # Add Convolution Layers '128' filters and receptive filed of size (5 x 5)
        model.add(Convolution2D(128, kernel_size[0], kernel_size[1],border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))
        # Then Max Pooling with 'pool size -"pool_size"', Default Stride is same as "pool_size"
        model.add(MaxPooling2D(pool_size=pool_size))
        
        ''' Fourth Set of CONV => RELU => POOL '''
        # Add Convolution Layers '256' filters and receptive field of dims "kernel_size"
        model.add(Convolution2D(256, kernel_size[0], kernel_size[1], border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))        
        # Add Convolution Layers '256' filters and receptive field of dims "kernel_size"
        model.add(Convolution2D(256, kernel_size[0], kernel_size[1], border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))        
        # Add Convolution Layers '256' filters and receptive field of dims "kernel_size"
        model.add(Convolution2D(256, kernel_size[0], kernel_size[1],  border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))
        # Then Max Pooling with 'pool size -"pool_size"', Default Stride is same as "pool_size"
        model.add(MaxPooling2D(pool_size=pool_size))

        ''' Fifth Set of CONV => RELU => POOL '''
        # Add Convolution Layers '512' filters and receptive field of dims "kernel_size"
        model.add(Convolution2D(512, kernel_size[0], kernel_size[1], border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))        
        # Add Convolution Layers '512' filters and receptive field of dims "kernel_size"
        model.add(Convolution2D(512, kernel_size[0], kernel_size[1], border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))        
        # Add Convolution Layers '512' filters and receptive field of dims "kernel_size"
        model.add(Convolution2D(512, kernel_size[0], kernel_size[1], border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))        
        # Add Convolution Layers '512' filters and receptive field of dims "kernel_size"
        model.add(Convolution2D(512, kernel_size[0], kernel_size[1], border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))
        # Then Max Pooling with 'pool size -"pool_size"', Default Stride is same as "pool_size"
        model.add(MaxPooling2D(pool_size=pool_size))
        # Add a Dropout layer with 0.5 percentage
        model.add(Dropout(0.5))
        
        ''' Sixth Set of CONV => RELU => POOL '''
        # Add Convolution Layers '512' filters and receptive field of dims "kernel_size"
        model.add(Convolution2D(512, kernel_size[0], kernel_size[1], border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))        
        # Add Convolution Layers '512' filters and receptive field of dims "kernel_size"
        model.add(Convolution2D(512, kernel_size[0], kernel_size[1], border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))        
        # Add Convolution Layers '512' filters and receptive field of dims "kernel_size"
        model.add(Convolution2D(512, kernel_size[0], kernel_size[1], border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))        
        # Add Convolution Layers '512' filters and receptive field of dims "kernel_size"
        model.add(Convolution2D(512, kernel_size[0], kernel_size[1], border_mode='same'))
        # Add Recitified Linear Units
        model.add(Activation('relu'))
        # Then Max Pooling with 'pool size -"pool_size"', Default Stride is same as "pool_size"
        model.add(MaxPooling2D(pool_size=pool_size))
        
        ''' Fully Connected Layers, followed by 'RELU' layer and 'DROPUT' '''
        # First flatten the input 
        model.add(Flatten())
        # Add FC (Dense) Layer with 'output_dim' - 4096
        model.add(Dense(1152))
        # Add Recitified Linear Units
        model.add(Activation('relu'))
        # Add a Dropout layer with 0.25 percentage
        model.add(Dropout(0.5))
        # Add FC (Dense) Layer with 'output_dim' - 4096
        model.add(Dense(1152))
        # Add Recitified Linear Units
        model.add(Activation('relu'))
        # Add a Dropout layer with 0.25 percentage
        model.add(Dropout(0.5))
        
        ''' Final Soft Max Layer '''
        # FC Layer with 'output_dim' - number_of_classes
        model.add(Dense(classes))
        # Add Final Soft Max Activation
        model.add(Activation('softmax'))
        
        # Load weights if given
        if weightsPath is not None:
            model.load_weights(weightsPath)
            
        # Return the constructed model
        return model
