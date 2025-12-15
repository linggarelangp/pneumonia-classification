import tensorflow as tf
from src.config.load_config import load_config

def build_model(
    model_name="ResNet50_classifier",
    dropout_rate=0.5,
    learning_rate=1e-4,
    use_transfer_learning=True,
):
    
    config = load_config()
    shape = config["data"]["image_size"]
    img_shape = (shape, shape, 3)
    
    inputs = tf.keras.Input(shape=img_shape)

    weights_type = "imagenet" if use_transfer_learning else None
    
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=weights_type,
        input_shape=img_shape,
        input_tensor=inputs
    )
    
    base_model.trainable = not use_transfer_learning

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(
        inputs=inputs, 
        outputs=outputs, 
        name=model_name
    )
    
    metrics = [
        'accuracy',
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=metrics,
    )

    return model