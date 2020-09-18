from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, MaxPooling2D, Lambda, BatchNormalization, add, multiply
from tensorflow.keras.layers import Conv2DTranspose, Dropout, GlobalAveragePooling2D,  Dense, Average, Reshape, Input
from tensorflow.nn import relu, relu6, swish

def MinimNet(input_shape, output_shape):
	inputs = Input(shape=input_shape)
	output = createLayers_(inputs, output_shape)

	model = Model(inputs=inputs, outputs=output)

	return model

def bottleNeckLayer(input_x, n, s, factor):
	x = Conv2D(n, kernel_size=1)(input_x)
	x = relu6(x)
	x = BatchNormalization()(x)
	x = DepthwiseConv2D(depth_multiplier=factor, kernel_size=3, 
						padding='same', depthwise_initializer='he_uniform')(x)
	x = swish(x)
	x = BatchNormalization()(x)
	x = Conv2D(n, kernel_size=1)(x)
	x = relu6(x)
	x = BatchNormalization()(x)
	x = DepthwiseConv2D(kernel_size=3, padding='same', 
						depthwise_initializer='he_uniform', strides=s)(x)
	x = swish(x)
	x = BatchNormalization()(x)
	x = DepthwiseConv2D(depth_multiplier=factor, kernel_size=3, 
						padding='same', depthwise_initializer='he_uniform')(x)
	x = swish(x)
	x = BatchNormalization()(x)

	return x

def invertedBottleNeckLayer(input_x, n, s, factor):
	x = Conv2D(n, kernel_size=1)(input_x)
	x = relu6(x)
	x = BatchNormalization()(x)
	x = DepthwiseConv2D(depth_multiplier=factor, kernel_size=3, padding='same', depthwise_initializer='he_uniform', strides=s)(x)
	x = swish(x)
	x = BatchNormalization()(x)
	x = Conv2D(n, kernel_size=1)(x)
	x = relu6(x)
	x = BatchNormalization()(x)
	return x

def squeezeGlobalEmbedLayer(input_x, n, r):
	x = GlobalAveragePooling2D()(input_x)
	se_shape = (1, 1, x.shape[1])
	x = Reshape(se_shape)(x)
	x = Dense(n//r)(x)
	x = relu6(x)
	x = Dense(n, activation='sigmoid')(x)
	return x

def seModule(input_x, n, r, s=1, factor=2):
	x2 = Lambda(lambda x: 1 * x)(input_x)

	x1 = Lambda(lambda x: 1 * x)(input_x)
	x1 = invertedBottleNeckLayer(x1, n, s, factor)
	
	x = squeezeGlobalEmbedLayer(input_x, n, r)

	# scale
	x = multiply([x1,x])
	x2 = Conv2D(n, activation='relu', kernel_size=1, strides=s)(x2)

	# residual
	x = add([x,x2])

	return x

def createLayers_(input_x, out):
	x = Conv2D(4, activation='swish', kernel_size=5, padding='same')(input_x)
	x = Conv2D(8, activation='swish', kernel_size=3, padding='same')(x)
	x = MaxPooling2D(pool_size=(2,2), strides=2)(x)

	#				layers 	N 	R  S  F
	mid_modules = [	['inv', 16,    1, 2],
					['inv',	32,	   2, 2],
					['se' , 64, 2, 1, 4],
					['inv',	64,	   2, 4],
					['se' ,128, 4, 1, 4],
					['inv',128,	   2, 8]]

	# build mid
	x_inv = None
	for layer in mid_modules:
		if layer[0] == 'inv':
			x = bottleNeckLayer(x, layer[1], layer[2], layer[3])
		elif layer[0] == 'se':
			x = seModule(x, layer[1], layer[2], layer[3], layer[4])

	# tail
	x = GlobalAveragePooling2D()(x)
	x = Dense(200)(x)
	x = Dense(out, activation='softmax')(x)

	return x

if __name__ == '__main__':
	model = MinimNet((32, 32, 1), 94)
	model.summary()