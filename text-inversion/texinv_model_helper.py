import tensorflow as tf
import numpy as np
import keras_cv

def traverse_layers(layer):
    if hasattr(layer, "layers"):
        for layer in layer.layers:
            yield layer
    if hasattr(layer, "token_embedding"):
        yield layer.token_embedding
    if hasattr(layer, "position_embedding"):
        yield layer.position_embedding

def add_new_token(stable_diffusion, closest_token):
    tokenized_initializer = stable_diffusion.tokenizer.encode(closest_token)[1]
    new_weights = stable_diffusion.text_encoder.layers[2].token_embedding(
        tf.constant(tokenized_initializer)
    )


    # The embedding layer is the 2nd layer in the text encoder
    old_token_weights = stable_diffusion.text_encoder.layers[
        2
    ].token_embedding.get_weights()
    old_position_weights = stable_diffusion.text_encoder.layers[
        2
    ].position_embedding.get_weights()

    old_token_weights = old_token_weights[0]
    new_weights = np.expand_dims(new_weights, axis=0)
    new_weights = np.concatenate([old_token_weights, new_weights], axis=0)

    # Get len of .vocab instead of tokenizer
    new_vocab_size = new_weights.shape[0]

    # Have to set download_weights False so we can init (otherwise tries to load weights)
    new_encoder = keras_cv.models.stable_diffusion.TextEncoder(
        keras_cv.models.stable_diffusion.stable_diffusion.MAX_PROMPT_LENGTH,
        vocab_size=new_vocab_size,
        download_weights=False,
    )
    for index, layer in enumerate(stable_diffusion.text_encoder.layers):
        # Layer 2 is the embedding layer, so we omit it from our weight-copying
        if index == 2:
            continue
        new_encoder.layers[index].set_weights(layer.get_weights())


    new_encoder.layers[2].token_embedding.set_weights([new_weights])
    new_encoder.layers[2].position_embedding.set_weights(old_position_weights)

    stable_diffusion._text_encoder = new_encoder
    stable_diffusion._text_encoder.compile(jit_compile=True)


def freeze_layers(stable_diffusion):
    stable_diffusion.diffusion_model.trainable = False
    stable_diffusion.decoder.trainable = False
    stable_diffusion.text_encoder.trainable = True

    stable_diffusion.text_encoder.layers[2].trainable = True

    for layer in traverse_layers(stable_diffusion.text_encoder):
        if isinstance(layer, tf.keras.layers.Embedding) or "clip_embedding" in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    stable_diffusion._text_encoder.layers[2].position_embedding.trainable = False