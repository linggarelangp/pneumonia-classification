import tensorflow as tf

def get_callbacks():
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_recall',
        mode='max',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_recall',
        mode='max',
        factor=0.5,
        patience=2,
        verbose=1
    )

    return [early_stopping, reduce_lr]
