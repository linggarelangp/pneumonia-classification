import tensorflow as tf

from src.config.load_config import load_config

def train_model(model, train_gen, val_gen, callbacks):
    config = load_config()
    epochs = config["training"]["epochs"]
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
        ],
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )

    return history
