import tensorflow as tf
import numpy as np


def main():
    # model_name = "model_cnn"
    model_name = "model_sp"

    tflite_model = convert(model_name)
    tflite_model_qua = convert_qua(model_name)

    with open(f"models/tflite/{model_name}.tflite", "wb") as f:
        f.write(tflite_model)

    with open(f"models/tflite/{model_name}_int8.tflite", "wb") as f:
        f.write(tflite_model_qua)


def convert(model_name):
    converter = tf.lite.TFLiteConverter.from_saved_model(f"models/{model_name}")

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        # tf.lite.OpsSet.SELECT_TF_OPS
    ]

    return converter.convert()


# Edge TPU int8 quantization
def convert_qua(model_name):
    converter = tf.lite.TFLiteConverter.from_saved_model(f"models/{model_name}")

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    x_test = np.load("datasets/x_test.npy")

    def representative_dataset():
        for i in range(500):
            data = x_test[i].reshape((1, 50, 50, 3))
            yield [data.astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()


if __name__ == "__main__":
    main()
