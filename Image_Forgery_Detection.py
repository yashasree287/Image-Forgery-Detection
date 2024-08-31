
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image, ImageChops
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
import seaborn as sns

# Define the ELA function
def ELA(img_path, quality=100, threshold=60):
    TEMP = 'ela_' + 'temp.jpg'
    SCALE = 10
    original = Image.open(img_path)
    diff = ""

    try:
        original.save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original, temporary)

    except:
        original.convert('RGB').save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original.convert('RGB'), temporary)

    d = diff.load()
    WIDTH, HEIGHT = diff.size

    for x in range(WIDTH):
        for y in range(HEIGHT):
            r, g, b = d[x, y]
            modified_intensity = int(0.2989 * r + 0.587 * g + 0.114 * b)
            d[x, y] = modified_intensity * SCALE, modified_intensity * SCALE, modified_intensity * SCALE

    calculated_threshold = threshold * (quality / 100)
    binary_mask = diff.point(lambda p: 255 if p >= calculated_threshold else 0)

    return binary_mask

# Dataset paths
dataset_path = "C:\Users\Yashasree\Downloads\image-forgery-detection-with-ela\Dataset\
"
path_original = 'Original/'
path_tampered = 'Forged/'

total_original = os.listdir(dataset_path + path_original)
total_tampered = os.listdir(dataset_path + path_tampered)

# Resizing the images and saving in output directory
output_path = './resized_images/'
if not os.path.exists(output_path):
    os.makedirs(output_path + "fake_images/")
    os.makedirs(output_path + "pristine_images/")
    height, width = 224, 224
    for fake_image in total_tampered:
        try:
            img = Image.open(dataset_path + path_tampered + fake_image).convert("RGB")
            img = img.resize((height, width), Image.Resampling.LANCZOS)
            img.save(output_path + "fake_images/" + fake_image)
        except Exception as e:
            print(f"Error processing {fake_image}: {e}")
        
    for pristine_image in total_original:
        try:
            img = Image.open(dataset_path + path_original + pristine_image).convert("RGB")
            img = img.resize((height, width), Image.Resampling.LANCZOS)
            img.save(output_path + "pristine_images/" + pristine_image)
        except Exception as e:
            print(f"Error processing {pristine_image}: {e}")
else:
    print('Images already resized.')

# Converting images to ELA format
ela_images_path = './ELA_IMAGES/'
ela_real = ela_images_path + 'Original/'
ela_fake = ela_images_path + 'Forged/'

if not os.path.exists(ela_images_path):
    os.makedirs(ela_real)
    os.makedirs(ela_fake)
    for i in os.listdir(output_path + "fake_images/"):
        ELA(output_path + "fake_images/" + i).save(ela_fake + i)
    for i in os.listdir(output_path + "pristine_images/"):
        ELA(output_path + "pristine_images/" + i).save(ela_real + i)
else:
    print('ELA conversion already done.')

# Load ELA images into arrays
X = []
Y = []

for file in os.listdir(ela_real):
    img = Image.open(ela_real + file)
    img = np.array(img)
    X.append(img)
    Y.append(0)

for file in os.listdir(ela_fake):
    img = Image.open(ela_fake + file)
    img = np.array(img)
    X.append(img)
    Y.append(1)

X = np.array(X)
Y = np.array(Y)

# Splitting the data into train and test sets
x_train, x_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.2, random_state=1, shuffle=True)
y_train = to_categorical(y_train, 2)
y_dev = to_categorical(y_dev, 2)

# Define and compile the model
base_model = DenseNet121(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

for layer in base_model.layers:
    layer.trainable = True

x = base_model.output
x = Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
x = Dropout(0.8)(x)
x = Dense(16, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
x = Dense(2, activation='softmax')(x)

model = Model(base_model.input, x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
batch_size = 64

hist = model.fit(x_train, y_train,
                 epochs=epochs, batch_size=batch_size,
                 validation_data=(x_dev, y_dev),
                 verbose=1, shuffle=True)

# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2, 1)
ax[0].plot(hist.history['loss'], color='b', label="Training loss")
ax[0].plot(hist.history['val_loss'], color='r', label="Validation loss")
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(hist.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(hist.history['val_accuracy'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

plt.show()

# Evaluate the model
y_pred = model.predict(x_dev)
y_pred_classes = np.argmax(y_pred, axis=1)
y_dev_classes = np.argmax(y_dev, axis=1)

conf_matrix = confusion_matrix(y_dev_classes, y_pred_classes)

precision = precision_score(y_dev_classes, y_pred_classes)
recall = recall_score(y_dev_classes, y_pred_classes)
f1 = f1_score(y_dev_classes, y_pred_classes)
accuracy = accuracy_score(y_dev_classes, y_pred_classes)
fpr, tpr, _ = roc_curve(y_dev_classes, y_pred[:, 1])
roc_auc = auc(fpr, tpr)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

#vgg19
from tensorflow.keras.applications import DenseNet121, Xception, VGG19,ResNet101
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D,BatchNormalization,Dropout,MaxPooling2D
from tensorflow.keras.regularizers import l1,l2,l1_l2

# Create the model based on DenseNet121
base_model2 = ResNet101(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Unfreeze the layers in the base model
for layer in base_model2.layers:
    layer.trainable = True
x=base_model2.output
x=Conv2D(1024,(3,3),padding='same',activation='relu')(x)
x=GlobalAveragePooling2D()(x)
x=Flatten()(x)
x=Dense(1024,activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
x=Dropout(0.8)(x)
x=Dense(16,activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
x=Dense(2,activation='softmax')(x)
model2=Model(base_model2.input,x)
model2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

hist = model2.fit(x_train,y_train,
                 epochs = 20, batch_size = batch_size,
                validation_data = (x_dev,y_dev),
                #callbacks = [early_stop,reduce_lr],
                verbose=1,shuffle=True)
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(hist.history['loss'], color='b', label="Training loss")
ax[0].plot(hist.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(hist.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(hist.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


