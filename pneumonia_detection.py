import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

# Define data directories
train_dir = 'content/chest-xray-pneumonia/chest_xray/train/'
val_dir = 'content/chest-xray-pneumonia/chest_xray/val'
test_dir = 'content/chest-xray-pneumonia/chest_xray/test'

# Define data generators
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)


def bar_plot(dir):
    x = len(os.listdir(os.path.join(dir, 'NORMAL')))
    y = len(os.listdir(os.path.join(dir, 'PNEUMONIA')))
    category = ['NORMAL', 'PNEUMONIA']
    count = [x, y]
    plot = plt.bar(category, count)
    plot[0].set_color('orange')
    plt.title('Plot of number of values for each category')
    plt.show()


print('Training images:\n')
print('NORMAL:', len(os.listdir(os.path.join(train_dir, 'NORMAL'))))
print('PNEUMONIA:', len(os.listdir(os.path.join(train_dir, 'PNEUMONIA'))))
print('Total Training images:',
      len(os.listdir(os.path.join(train_dir, 'NORMAL'))) + len(os.listdir(os.path.join(train_dir, 'PNEUMONIA'))))
print('*' * 49)
bar_plot(train_dir)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical')

# Define model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.1),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(2, activation='sigmoid')
])
#model = tf.keras.models.load_model('pneumonia_model.keras')
# Compile model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

#Train model
#history = model.fit(
   # train_generator,
  #  epochs=25,
 #   validation_data=val_generator)

#model.save('pneumonia_model2.keras')

# Evaluate model on test data
loss, accuracy = model.evaluate(test_generator)
print('The accuracy of the model on test dataset is', accuracy * 100)


# Perform inference on individual images
def predict_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    load_model = tf.keras.models.load_model('pneumonia_model.keras')
    result = load_model.predict(img_array)
    class_index = tf.argmax(result, axis=1).numpy()[0]
    classes = ['NORMAL', 'PNEUMONIA']
    print('Predicted class:', classes[class_index])


# Example usage
predict_image("content/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0001-0001.jpeg")
predict_image("content/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg")
