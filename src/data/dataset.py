import tensorflow as tf


FEATURE_NAMES = [
    "adgroup_id",
    "cate_id",
    "brand",
    "cate_history",
    "brand_history",
    "btag_history",
    "price",
]


def create_tf_dataset(
    features,
    target,
    batch_size=256,
    shuffle=False,
):
    """
    Convert model-ready NumPy features into a TensorFlow Dataset.
    """

    inputs = {
        name: features[name]
        for name in FEATURE_NAMES
    }

    dataset = tf.data.Dataset.from_tensor_slices(
        (inputs, target)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(target),
            seed=42,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset