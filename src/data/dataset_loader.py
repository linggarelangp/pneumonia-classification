import tensorflow as tf
from src.config.load_config import load_config

def create_generators(
    df_train,
    df_val,
    df_test,
    use_data_augmentation=True,
):
    config = load_config()
    
    img_size = config["data"]["image_size"]
    batch_size = config["training"]["batch_size"]
    
    if use_data_augmentation:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.08,
            brightness_range=(0.9, 1.1),
            fill_mode='nearest',
            horizontal_flip=False,
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input
        )
        
    else:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input
        )
    
    test_val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input
    )

    print(f"[INFO] Creating Generator with Batch Size: {batch_size}")

    train_generator = train_datagen.flow_from_dataframe(
        df_train,
        x_col="img_path",
        y_col="label",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        class_mode="binary"
    )

    val_generator = test_val_datagen.flow_from_dataframe(
        df_val,
        x_col="img_path",
        y_col="label",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
        class_mode="binary"
    )

    test_generator = test_val_datagen.flow_from_dataframe(
        df_test,
        x_col="img_path",
        y_col="label",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
        class_mode="binary"
    )

    return train_generator, val_generator, test_generator