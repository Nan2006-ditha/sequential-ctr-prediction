import tensorflow as tf
from tensorflow.keras import layers


class TransformerEncoderBlock(layers.Layer):
    """
    Standard Transformer encoder block: multi-head self-attention
    followed by a feed-forward network, each with residual + LayerNorm.
    """

    def __init__(self, d_model, num_heads, ff_dim, dropout, **kwargs):
        super().__init__(**kwargs)

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(d_model),
            ]
        )
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, attention_mask, training=False):
        attn = self.mha(
            query=x,
            value=x,
            key=x,
            attention_mask=attention_mask,
            training=training,
        )
        x = self.norm1(x + self.drop1(attn, training=training))

        ffn = self.ffn(x)
        return self.norm2(x + self.drop2(ffn, training=training))


class TransformerCTRModel(tf.keras.Model):
    """
    Transformer-based CTR prediction model with target-aware attention.

    Pipeline:
      1. Embed the behavior history (category, brand, btag) - identical
         embeddings to the LSTM model.
      2. Project to d_model and add learned positional embeddings.
      3. Transformer encoder (self-attention over the history).
      4. Target-aware attention: the candidate ad is the query, the
         encoded history items are keys/values, so each past behavior is
         weighted by its relevance to THIS candidate.
      5. Same prediction head as the LSTM model.

    attention_type:
      "dot" - scaled dot-product scoring between candidate and history
      "din" - DIN-style MLP scoring on [q, k, q-k, q*k]
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
        num_heads=2,
        num_layers=1,
        ff_dim=128,
        max_seq_len=50,
        attention_type="dot",
    ):
        super().__init__()

        if attention_type not in ("dot", "din"):
            raise ValueError("attention_type must be 'dot' or 'din'")

        self.hidden_dim = hidden_dim
        self.attention_type = attention_type

        # --------------------------------------------------
        # 1. Embeddings (same as LSTM model)
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
        self.adgroup_embedding = layers.Embedding(
            input_dim=adgroup_vocab_size,
            output_dim=adgroup_embedding_dim,
            name="adgroup_embedding",
        )

        # --------------------------------------------------
        # 2. History encoder
        # --------------------------------------------------

        self.history_projection = layers.Dense(
            hidden_dim,
            name="history_projection",
        )
        self.position_embedding = layers.Embedding(
            input_dim=max_seq_len,
            output_dim=hidden_dim,
            name="position_embedding",
        )
        self.encoder_blocks = [
            TransformerEncoderBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                name=f"encoder_block_{i}",
            )
            for i in range(num_layers)
        ]

        # --------------------------------------------------
        # 3. Candidate projection (same as LSTM model)
        # --------------------------------------------------

        self.candidate_dense = layers.Dense(
            hidden_dim,
            activation="relu",
            name="candidate_projection",
        )

        # --------------------------------------------------
        # 4. Target-aware attention
        # --------------------------------------------------

        self.query_proj = layers.Dense(hidden_dim, name="target_query")
        self.key_proj = layers.Dense(hidden_dim, name="target_key")

        if attention_type == "din":
            self.att_mlp = tf.keras.Sequential(
                [
                    layers.Dense(32, activation="relu"),
                    layers.Dense(1),
                ],
                name="din_attention_mlp",
            )

        # --------------------------------------------------
        # 5. Prediction head (same as LSTM model)
        # --------------------------------------------------

        self.dense_1 = layers.Dense(128, activation="relu", name="dense_1")
        self.dropout = layers.Dropout(dropout, name="dropout")
        self.dense_2 = layers.Dense(64, activation="relu", name="dense_2")
        self.output_layer = layers.Dense(
            1,
            activation="sigmoid",
            name="click_probability",
        )

    def _encode(self, inputs, training=False):
        """
        Returns (user_representation, attention_weights, candidate).
        """

        # ---------- history ----------

        cate_history = self.cate_embedding(inputs["cate_history"])
        brand_history = self.brand_embedding(inputs["brand_history"])
        btag_history = self.btag_embedding(inputs["btag_history"])

        history = tf.concat(
            [cate_history, brand_history, btag_history],
            axis=-1,
        )

        # Padding mask: id 0 is padding. tf.concat drops the Keras mask,
        # so it is rebuilt explicitly here.
        mask = tf.not_equal(inputs["cate_history"], 0)  # (B, L)
        seq_len = tf.shape(history)[1]

        x = self.history_projection(history)
        positions = tf.range(seq_len)
        x = x + self.position_embedding(positions)[tf.newaxis, :, :]

        # Key-padding mask for self-attention: (B, L, L)
        attention_mask = tf.tile(mask[:, tf.newaxis, :], [1, seq_len, 1])

        for block in self.encoder_blocks:
            x = block(x, attention_mask=attention_mask, training=training)

        # ---------- candidate ----------

        adgroup = self.adgroup_embedding(inputs["adgroup_id"])

        cate = self.cate_embedding(tf.expand_dims(inputs["cate_id"], axis=-1))
        cate = tf.squeeze(cate, axis=1)

        brand = self.brand_embedding(tf.expand_dims(inputs["brand"], axis=-1))
        brand = tf.squeeze(brand, axis=1)

        price = tf.expand_dims(inputs["price"], axis=-1)

        candidate = self.candidate_dense(
            tf.concat([adgroup, cate, brand, price], axis=-1)
        )

        # ---------- target-aware attention ----------

        q = self.query_proj(candidate)  # (B, D)
        k = self.key_proj(x)  # (B, L, D)

        if self.attention_type == "din":
            q_tiled = tf.tile(q[:, tf.newaxis, :], [1, seq_len, 1])
            att_input = tf.concat(
                [q_tiled, k, q_tiled - k, q_tiled * k],
                axis=-1,
            )
            scores = tf.squeeze(self.att_mlp(att_input), axis=-1)
        else:
            scores = tf.einsum("bd,bld->bl", q, k)
            scores = scores / tf.sqrt(tf.cast(self.hidden_dim, scores.dtype))

        scores = tf.where(
            mask,
            scores,
            tf.constant(-1e9, dtype=scores.dtype),
        )
        weights = tf.nn.softmax(scores, axis=-1)

        # Zero out padding, and zero everything for users with no history
        weights = weights * tf.cast(mask, weights.dtype)
        has_history = tf.cast(
            tf.reduce_any(mask, axis=1),
            weights.dtype,
        )[:, tf.newaxis]
        weights = weights * has_history

        user_representation = tf.einsum("bl,bld->bd", weights, x)

        return user_representation, weights, candidate

    def call(self, inputs, training=False):
        user_representation, _, candidate = self._encode(
            inputs,
            training=training,
        )

        combined = tf.concat([user_representation, candidate], axis=-1)

        x = self.dense_1(combined)
        x = self.dropout(x, training=training)
        x = self.dense_2(x)

        return self.output_layer(x)

    def explain(self, inputs):
        """
        Returns attention weights over the history, shape (B, L).
        Useful for showing which past behaviors drove a prediction.
        """
        _, weights, _ = self._encode(inputs, training=False)
        return weights