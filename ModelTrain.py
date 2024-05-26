import os

# Set ClearML environment variables
os.environ['CLEARML_WEB_HOST'] = 'https://app.clear.ml'
os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'
os.environ['CLEARML_FILES_HOST'] = 'https://files.clear.ml'
os.environ['CLEARML_API_ACCESS_KEY'] = 'I7O2G108OEWA9MU3S2Y5'
os.environ['CLEARML_API_SECRET_KEY'] = 'G5QDGfcv2BvYiorxuiGiQTEPtGLx4ZeZNTEf8GMwg41nTUssoA'

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LambdaCallback, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import math
from clearml import Task, Dataset, OutputModel
import matplotlib.pyplot as plt
import pickle
 
def train_and_log_model(project_name, task_name, dataset_project, dataset_name, test_size, random_state, initial_lr, drop, epochs_drop, num_epochs, batch_size):
    task = Task.init(project_name=project_name, task_name=task_name, task_type=Task.TaskTypes.training)
 
    test_size = float(test_size)
    random_state = int(random_state)
    initial_lr = float(initial_lr)
    drop = float(drop)
    epochs_drop = int(epochs_drop)
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
 
    dataset = Dataset.get(dataset_project=dataset_project, dataset_name=dataset_name, alias='latest')
    dataset_path = dataset.get_local_copy()
 
    data = np.load(os.path.join(dataset_path, "data_preprocessed.npy"))
    labels = np.load(os.path.join(dataset_path, "labels_preprocessed.npy"))
 
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    labels = to_categorical(integer_encoded)
 
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)
 
    baseModel = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    headModel = Dense(len(np.unique(integer_encoded)), activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
 
    # Configure callbacks for logging
    logger = task.get_logger()
    lambda_clbk = LambdaCallback(
        on_epoch_end=lambda epoch, logs: [
            logger.report_scalar("loss", "train", iteration=epoch, value=logs["loss"]),
            logger.report_scalar("accuracy", "train", iteration=epoch, value=logs["accuracy"]),
            logger.report_scalar("val_loss", "validation", iteration=epoch, value=logs["val_loss"]),
            logger.report_scalar("val_accuracy", "validation", iteration=epoch, value=logs["val_accuracy"]),
        ]
    )
 
    # Learning rate scheduler
    def lr_decay(epoch):
        return initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
 
    lr_scheduler = LearningRateScheduler(lr_decay)
 
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=initial_lr), metrics=["accuracy"])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, batch_size=batch_size, callbacks=[lr_scheduler, lambda_clbk])
 
    # Save the trained model and report
    model_file_path = os.path.join(os.getcwd(), "model.keras")
    model.save(model_file_path)  # Use the native Keras format
    output_model = OutputModel(task=task)
    output_model.update_weights(model_file_path)
    output_model.publish()
    task.upload_artifact("trained_model", artifact_object=model_file_path)
 
    # Retrieve the model ID from the OutputModel object
    model_id = output_model.id
 
    # Close the task
    task.close()
    return model_id  # Return the model_id instead of history
