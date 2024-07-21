import tensorflow as tf
import numpy as np
import time


def main():
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    interpreter = tf.lite.Interpreter(model_path="model_sp.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    hits = 0
    t1 = time.time()
    for i in range(x_test.shape[0]):
        image = x_test[i].reshape((1, 50, 50, 3))
        input_tensor = tf.convert_to_tensor(image)
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        if np.argmax(output_data[0]) == y_test[i]:
            hits += 1
    t2 = time.time()

    print(f"{x_test.shape[0]} images processed in {t2 - t1:.3f} s")
    print(f"Accuracy: {hits/x_test.shape[0]:.4f}    Average time: {(t2-t1)/x_test.shape[0]*1000:.0f} ms")


if __name__ == "__main__":
    main()
