import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r"model\resnet50_model_2.keras")

datagen = ImageDataGenerator(rescale=1./255)

# Load train data
train_generator = datagen.flow_from_directory(
    'selected_data/train',  # Training directory path
    target_size=(256, 256),  # Adjust based on your model's input size
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Load validation data
validation_generator = datagen.flow_from_directory(
    'selected_data/validation',  # Validation directory path
    target_size=(256, 256),  # Adjust based on your model's input size
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Load test data
test_generator = datagen.flow_from_directory(
    'selected_data/test',  # Test directory path
    target_size=(256, 256),  # Adjust based on your model's input size
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Define a function to evaluate a generator
def evaluate_generator(generator, model):
    predictions = model.predict(generator)
    threshold = 0.5
    predicted_classes = ["ktp" if p < threshold else "non-ktp" for p in predictions]

    # Reverse class indices to map indices back to class labels
    reverse_class_indices = {v: k for k, v in generator.class_indices.items()}

    correct_predictions = 0
    total_predictions = len(predicted_classes)

    for i, (pred, true_label, file_path) in enumerate(zip(predicted_classes, generator.classes, generator.filepaths)):
        true_class = reverse_class_indices[true_label]
        if pred == true_class:
            correct_predictions += 1
        print(f"Image {file_path}: Prediction score {predictions[i][0]} - Predicted class: {pred} - True class: {true_class}")

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Evaluate on training data
print("Evaluating on training data...")
print(evaluate_generator(train_generator, model))

# Evaluate on validation data
print("Evaluating on validation data...")
print(evaluate_generator(validation_generator, model))

# Evaluate on test data
print("Evaluating on test data...")
print(evaluate_generator(test_generator, model))

