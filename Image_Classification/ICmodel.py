import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, ResNet50, InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Set up the data generators with categorical labels
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    '/kaggle/input/vehicle-classification/Vehicles',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Categorical for multi-class one-hot encoded labels
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    '/kaggle/input/vehicle-classification/Vehicles',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Ensure this is categorical as well
    subset='validation'
)

# Define the DenseNet model
def build_densenet():
    base_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(7, activation='softmax')(x)  # 7 output units for 7 classes
    model = Model(base_model.input, x)
    return model

# Define the ResNet model
def build_resnet():
    base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(7, activation='softmax')(x)  # 7 output units for 7 classes
    model = Model(base_model.input, x)
    return model

# Define the InceptionV3 model
def build_inception():
    base_model = InceptionV3(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(7, activation='softmax')(x)  # 7 output units for 7 classes
    model = Model(base_model.input, x)
    return model

# Compile each model
densenet_model = build_densenet()
densenet_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

resnet_model = build_resnet()
resnet_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

inception_model = build_inception()
inception_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train each model and display outputs
print("Training DenseNet model...")
history_densenet = densenet_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    verbose=1
)

print("Training ResNet model...")
history_resnet = resnet_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    verbose=1
)

print("Training InceptionV3 model...")
history_inception = inception_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    verbose=1
)

# training history from each model:
print("DenseNet Training Accuracy: ", history_densenet.history['accuracy'])
print("ResNet Training Accuracy: ", history_resnet.history['accuracy'])
print("InceptionV3 Training Accuracy: ", history_inception.history['accuracy'])
