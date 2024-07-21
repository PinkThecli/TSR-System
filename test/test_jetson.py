import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import numpy as np
import time


def main():
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    saved_model_loaded = tf.saved_model.load("model_sp-tf2.8", tags=[tag_constants.SERVING])
    signature_keys = list(saved_model_loaded.signatures.keys())
    print("Signature keys:", signature_keys)

    infer = saved_model_loaded.signatures["serving_default"]
    print("Infer outputs:", infer.structured_outputs)

    # Warmup
    for i in range(50):
        x = x_test[i].reshape([1, 50, 50, 3])
        pred = infer(tf.convert_to_tensor(x))

    t1 = time.time()
    hits = 0
    for i in range(x_test.shape[0]):
        x = x_test[i].reshape([1, 50, 50, 3])
        pred = infer(tf.convert_to_tensor(x))
        if np.argmax(pred["output"][0].numpy()) == y_test[i]:
            hits += 1
    t2 = time.time()

    print(f"{x_test.shape[0]} images processed in {t2 - t1:.3f} s")
    print(f"Accuracy: {hits/x_test.shape[0]:.4f}    Average time: {(t2-t1)/x_test.shape[0]*1000:.0f} ms")


if __name__ == "__main__":
    main()
