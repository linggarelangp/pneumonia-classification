from src.config.load_config import load_config

def train_model(model, train_gen, val_gen, class_weight, callbacks):
    config = load_config()
    epochs = config["training"]["epochs"]
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks
    )

    return history
