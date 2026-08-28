from typing import NamedTuple, Dict, Any, Text, List
import os
import sys
import keras_tuner as kt
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

TunerFnResult = NamedTuple('TunerFnResult', [
    ('tuner', kt.engine.base_tuner.BaseTuner),
    ('fit_kwargs', Dict[Text, Any]),
])


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
    """Menghasilkan dataset batch untuk proses pencarian hyperparameter.

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


def get_model(
    hp: kt.HyperParameters,
    tf_transform_output: tft.TFTransformOutput
) -> tf.keras.Model:
    """Membangun model klasifikasi DNN Keras dengan hyperparameter yang dapat disetel.

    Args:
        hp: Objek HyperParameters dari KerasTuner.
        tf_transform_output: Metadata hasil transformasi dari TFTransformOutput.

    Returns:
        Model tf.keras.Model yang telah dikompilasi.
    """
    input_features: List[tf.keras.Input] = []

    # Inisialisasi tensor input untuk fitur numerik
    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(
                shape=(1,),
                name=transformed_name(feature),
                dtype=tf.float32
            )
        )

    # Inisialisasi tensor input untuk fitur kategorikal
    for feature in CATEGORICAL_FEATURES:
        input_features.append(
            tf.keras.Input(
                shape=(1,),
                name=transformed_name(feature),
                dtype=tf.int64
            )
        )

    transformed_inputs: List[tf.Tensor] = []

    # Menambahkan input numerik
    for idx in range(len(NUMERICAL_FEATURES)):
        transformed_inputs.append(input_features[idx])

    # Menambahkan lapisan embedding untuk fitur kategorikal
    for idx, (feature, vocab_size) in enumerate(
        CATEGORICAL_FEATURES.items(),
        start=len(NUMERICAL_FEATURES)
    ):
        embedding_dim = hp.Int(
            f'embed_dim_{feature}',
            min_value=4,
            max_value=16,
            step=4,
            default=8
        )
        embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size + 2,
            output_dim=embedding_dim
        )(input_features[idx])
        flattened = tf.keras.layers.Flatten()(embedding)
        transformed_inputs.append(flattened)

    # Menggabungkan seluruh tensor fitur
    concat = tf.keras.layers.concatenate(transformed_inputs)

    # Lapisan tersembunyi (Dense) dengan parameter yang dioptimasi
    num_layers = hp.Int('num_layers', min_value=1, max_value=3, default=2)
    hidden_layer = concat
    for i in range(num_layers):
        units = hp.Int(
            f'units_{i}',
            min_value=32,
            max_value=128,
            step=32,
            default=64
        )
        hidden_layer = tf.keras.layers.Dense(units, activation='relu')(hidden_layer)
        dropout_rate = hp.Float(
            f'dropout_{i}',
            min_value=0.1,
            max_value=0.4,
            step=0.1,
            default=0.2
        )
        hidden_layer = tf.keras.layers.Dropout(dropout_rate)(hidden_layer)

    # Lapisan output untuk klasifikasi biner
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer)
    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4], default=1e-3)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Fungsi utama tuner yang dipanggil oleh komponen TFX Tuner.

    Args:
        fn_args: Argumen eksekusi dari pipeline runner TFX.

    Returns:
        TunerFnResult yang berisi instance KerasTuner dan fit_kwargs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = _input_fn(fn_args.train_files, tf_transform_output, batch_size=64)
    eval_set = _input_fn(fn_args.eval_files, tf_transform_output, batch_size=64)

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: get_model(hp, tf_transform_output),
        objective=kt.Objective('val_auc', direction='max'),
        max_trials=5,
        directory=fn_args.working_dir,
        project_name='credit_card_churn_tuning'
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=5,
        restore_best_weights=True
    )

    fit_kwargs = {
        'x': train_set,
        'validation_data': eval_set,
        'steps_per_epoch': fn_args.train_steps,
        'validation_steps': fn_args.eval_steps,
        'callbacks': [early_stopping]
    }

    return TunerFnResult(tuner=tuner, fit_kwargs=fit_kwargs)
