from typing import List, Dict, Any, Callable
import os
import sys
import json
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))
for p in [_CURRENT_DIR, _PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from modules.dnnuuyzzo_transform import (
        CATEGORICAL_FEATURES,
        NUMERICAL_FEATURES,
        LABEL_KEY,
        transformed_name
    )
except ImportError:
    from dnnuuyzzo_transform import (
        CATEGORICAL_FEATURES,
        NUMERICAL_FEATURES,
        LABEL_KEY,
        transformed_name
    )


def _gzip_reader_fn(filenames: List[str]) -> tf.data.TFRecordDataset:
    """Membaca berkas TFRecord yang terkompresi dengan format GZIP.

    Args:
        filenames: Daftar jalur berkas yang akan dibaca.

    Returns:
        Objek TFRecordDataset.
    """
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(
    file_pattern: List[str],
    tf_transform_output: tft.TFTransformOutput,
    batch_size: int = 64
) -> tf.data.Dataset:
    """Menghasilkan dataset batch untuk proses pelatihan dan evaluasi model.

    Args:
        file_pattern: Pola berkas data yang sesuai dengan partisi.
        tf_transform_output: Objek TFTransformOutput yang memuat graph transformasi.
        batch_size: Jumlah sampel dalam satu batch data.

    Returns:
        Objek tf.data.Dataset yang telah dibatch.
    """
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    return tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=None,
        label_key=transformed_name(LABEL_KEY)
    )


def _build_keras_model(
    hp_dict: Dict[str, Any],
    tf_transform_output: tft.TFTransformOutput
) -> tf.keras.Model:
    """Membangun model Keras berdasarkan konfigurasi hyperparameter terbaik.

    Args:
        hp_dict: Kamus yang berisi nilai hyperparameter terbaik dari Tuner.
        tf_transform_output: Objek TFTransformOutput.

    Returns:
        Instance model tf.keras.Model yang telah dikompilasi.
    """
    input_features: List[tf.keras.Input] = []

    # Inisialisasi tensor input numerik
    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(
                shape=(1,),
                name=transformed_name(feature),
                dtype=tf.float32
            )
        )

    # Inisialisasi tensor input kategorikal
    for feature in CATEGORICAL_FEATURES:
        input_features.append(
            tf.keras.Input(
                shape=(1,),
                name=transformed_name(feature),
                dtype=tf.int64
            )
        )

    transformed_inputs: List[tf.Tensor] = []

    # Menambahkan tensor input numerik
    for idx in range(len(NUMERICAL_FEATURES)):
        transformed_inputs.append(input_features[idx])

    # Menambahkan representasi embedding kategorikal
    for idx, (feature, vocab_size) in enumerate(
        CATEGORICAL_FEATURES.items(),
        start=len(NUMERICAL_FEATURES)
    ):
        embed_dim = hp_dict.get(f'embed_dim_{feature}', 8)
        embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size + 2,
            output_dim=embed_dim
        )(input_features[idx])
        flattened = tf.keras.layers.Flatten()(embedding)
        transformed_inputs.append(flattened)

    # Menggabungkan seluruh tensor fitur
    concat = tf.keras.layers.concatenate(transformed_inputs)

    # Membangun lapisan dense tersembunyi
    num_layers = hp_dict.get('num_layers', 2)
    hidden_layer = concat
    for i in range(num_layers):
        units = hp_dict.get(f'units_{i}', 64)
        dropout_rate = hp_dict.get(f'dropout_{i}', 0.2)
        hidden_layer = tf.keras.layers.Dense(units, activation='relu')(hidden_layer)
        hidden_layer = tf.keras.layers.Dropout(dropout_rate)(hidden_layer)

    # Lapisan output dengan aktivasi sigmoid
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer)
    model = tf.keras.Model(inputs=input_features, outputs=outputs)
    learning_rate = hp_dict.get('learning_rate', 0.001)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model


def _get_serve_tf_examples_fn(
    model: tf.keras.Model,
    tf_transform_output: tft.TFTransformOutput
) -> Callable:
    """Membuat fungsi serving signature yang menyematkan graph prapemrosesan TFT.

    Args:
        model: Model Keras yang telah dilatih.
        tf_transform_output: Objek TFTransformOutput yang memuat graph transformasi.

    Returns:
        Fungsi TensorFlow untuk melayani serialisasi tf.train.Example.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        ]
    )
    def serve_tf_examples_fn(serialized_tf_examples: tf.Tensor) -> Dict[str, tf.Tensor]:
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return {'outputs': model(transformed_features)}

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs) -> None:
    """Fungsi utama eksekusi pelatihan yang dipanggil oleh komponen TFX Trainer.

    Args:
        fn_args: Argumen eksekusi dari pipeline runner TFX.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Memuat hyperparameter terbaik dari artefak Tuner jika tersedia
    hp_dict: Dict[str, Any] = {}
    if fn_args.hyperparameters:
        try:
            hp_dict = json.loads(fn_args.hyperparameters)
            if 'values' in hp_dict:
                hp_dict = hp_dict['values']
        except Exception:
            hp_dict = {}

    train_dataset = _input_fn(
        fn_args.train_files,
        tf_transform_output,
        batch_size=64
    )
    eval_dataset = _input_fn(
        fn_args.eval_files,
        tf_transform_output,
        batch_size=64
    )

    model = _build_keras_model(hp_dict, tf_transform_output)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir,
        update_freq='batch'
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        train_dataset,
        epochs=10,
        steps_per_epoch=fn_args.train_steps or 20,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps or 10,
        callbacks=[tensorboard_callback, early_stopping_callback],
        verbose=1
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output)
    }

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures
    )
