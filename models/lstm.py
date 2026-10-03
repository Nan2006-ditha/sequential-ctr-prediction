import tensorflow as tf
from tensorflow.keras import layers


class LSTMCTRModel(tf.keras.Model):
    """
    LSTM-based CTR prediction model.

    The model learns a representation of the user's historical
    behavior and combines it with the candidate advertisement
    features to predict click probability.

    use_mask:
      True  - padding (id 0) is explicitly masked, so the LSTM ignores it.
              This matches how the Transformer model handles padding.
      False - original behaviour. tf.concat drops the Keras mask, so the
              LSTM reads padding as real input.
    """

    def __init__(
        self,
        cate_vocab_size,
        brand_vocab_size,
        btag_vocab_size,
        adgroup_vocab_size,
        category_embedding_dim=16,
        brand_embedding_dim=16,
        btag_embedding_dim=8,
        adgroup_embedding_dim=16,
        hidden_dim=64,
        dropout=0.3,
        use_mask=True,
    ):
        super().__init__()

        self.use_mask = use_mask

        # --------------------------------------------------
        # 1. History Embeddings
        # --------------------------------------------------

        self.cate_embedding = layers.Embedding(
            input_dim=cate_vocab_size,
            output_dim=category_embedding_dim,
            mask_zero=True,
            name="category_embedding",
        )

        self.brand_embedding = layers.Embedding(
            input_dim=brand_vocab_size,
            output_dim=brand_embedding_dim,
            mask_zero=True,
            name="brand_embedding",
        )

        self.btag_embedding = layers.Embedding(
            input_dim=btag_vocab_size,
            output_dim=btag_embedding_dim,
            mask_zero=True,
            name="btag_embedding",
        )

        # --------------------------------------------------
        # 2. LSTM Encoder
        # --------------------------------------------------

        self.lstm = layers.LSTM(
            units=hidden_dim,
            return_sequences=False,
            name="lstm_encoder",
        )

        # --------------------------------------------------
        # 3. Candidate Advertisement Embeddings
        # --------------------------------------------------

        self.adgroup_embedding = layers.Embedding(
            input_dim=adgroup_vocab_size,
            output_dim=adgroup_embedding_dim,
            name="adgroup_embedding",
        )

        # --------------------------------------------------
        # 4. Candidate Projection
        # --------------------------------------------------

        self.candidate_dense = layers.Dense(
            hidden_dim,
            activation="relu",
            name="candidate_projection",
        )

        # --------------------------------------------------
        # 5. Prediction Head
        # --------------------------------------------------

        self.dense_1 = layers.Dense(128, activation="relu", name="dense_1")
        self.dropout = layers.Dropout(dropout, name="dropout")
        self.dense_2 = layers.Dense(64, activation="relu", name="dense_2")
        self.output_layer = layers.Dense(
            1,
            activation="sigmoid",
            name="click_probability",
        )

    def call(self, inputs, training=False):

        # --------------------------------------------------
        # 1. Encode user history
        # --------------------------------------------------

        cate_history = self.cate_embedding(inputs["cate_history"])
        brand_history = self.brand_embedding(inputs["brand_history"])
        btag_history = self.btag_embedding(inputs["btag_history"])

        history = tf.concat(
            [cate_history, brand_history, btag_history],
            axis=-1,
        )

        if self.use_mask:
            # tf.concat drops the Keras mask, so rebuild it explicitly
            mask = tf.not_equal(inputs["cate_history"], 0)
            user_representation = self.lstm(history, mask=mask)
        else:
            user_representation = self.lstm(history)

        # --------------------------------------------------
        # 2. Encode candidate advertisement
        # --------------------------------------------------

        adgroup = self.adgroup_embedding(inputs["adgroup_id"])

        cate = self.cate_embedding(tf.expand_dims(inputs["cate_id"], axis=-1))
        cate = tf.squeeze(cate, axis=1)

        brand = self.brand_embedding(tf.expand_dims(inputs["brand"], axis=-1))
        brand = tf.squeeze(brand, axis=1)

        price = tf.expand_dims(inputs["price"], axis=-1)

        candidate = tf.concat([adgroup, cate, brand, price], axis=-1)
        candidate = self.candidate_dense(candidate)

        # --------------------------------------------------
        # 3. Combine user + candidate
        # --------------------------------------------------

        combined = tf.concat([user_representation, candidate], axis=-1)

        # --------------------------------------------------
        # 4. Prediction head
        # --------------------------------------------------

        x = self.dense_1(combined)
        x = self.dropout(x, training=training)
        x = self.dense_2(x)

        return self.output_layer(x)