import os
import datetime
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
from keras import backend as K
K.clear_session()

# Set environment variable to avoid OpenMP error
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'selected_data/train',  # Training directory path
    target_size=(256, 256),
    batch_size=2,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    'selected_data/validation',  # Validation directory path
    target_size=(256, 256),
    batch_size=2,
    class_mode='binary'
)

# Load pre-trained ResNet50 model without the top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define a log directory for TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_path = 'model\\resnet50_model_2.keras'

checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', save_weights_only = False)

# Train the model
history_training = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[tensorboard_callback]
)

# plot training process

# plt.figure(figsize=(8, 8))
# epochs_range = range(10)
# plt.plot(epochs_range, history_training.history['accuracy'], label="Training Accuracy")
# plt.plot(epochs_range, history_training.history['val_accuracy'], label="Validation Accuracy")
# plt.axis(ymin=0.4, ymax=1)
# plt.grid()
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.legend(['train', 'validation'])
# plt.savefig("resnet_train_fit.jpg")
# plt.show()

# Unfreeze some layers in the base model for fine-tuning
for layer in base_model.layers:
    layer.trainable = True

# Compile the model again with a lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Continue training the model
history_tuning = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[tensorboard_callback]
)

# plot fine tuning process

# plt.figure(figsize=(8, 8))
# epochs_range = range(10)
# plt.plot(epochs_range, history_tuning.history['accuracy'], label="Training Accuracy")
# plt.plot(epochs_range, history_tuning.history['val_accuracy'], label="Validation Accuracy")
# plt.axis(ymin=0.4, ymax=1)
# plt.grid()
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.legend(['train', 'validation'])
# plt.savefig("resnet_finetune_fit.jpg")
# plt.show()

# Save the fine-tuned model to a file
model.save(checkpoint_path)



