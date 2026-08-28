from typing import Dict, List
import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES: Dict[str, int] = {
    'Gender': 2,
    'Education_Level': 7,
    'Marital_Status': 4,
    'Income_Category': 6,
    'Card_Category': 4
}

NUMERICAL_FEATURES: List[str] = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

LABEL_KEY: str = 'Attrition_Flag'


def transformed_name(key: str) -> str:
    """Menghasilkan nama fitur yang telah ditransformasi dengan penambahan sufiks.

    Args:
        key: Nama fitur asli.

    Returns:
        String nama fitur hasil transformasi.
    """
    return f"{key}_xf"


def preprocessing_fn(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Melakukan prapemrosesan fitur mentah menggunakan TensorFlow Transform.

    Args:
        inputs: Kamus pemetaan nama fitur mentah ke tensor.

    Returns:
        Kamus pemetaan nama fitur transformasi ke tensor hasil transformasi.
    """
    outputs: Dict[str, tf.Tensor] = {}

    # Standarisasi fitur numerik menggunakan penskalaan z-score
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(inputs[feature])

    # Menghitung dan menerapkan representasi vocabulary untuk fitur kategorikal
    for feature, vocab_size in CATEGORICAL_FEATURES.items():
        outputs[transformed_name(feature)] = tft.compute_and_apply_vocabulary(
            inputs[feature],
            top_k=vocab_size,
            num_oov_buckets=1,
            vocab_filename=feature
        )

    # Mengonversi label target biner ke tipe int64
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
