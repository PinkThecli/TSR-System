from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify
import numpy as np
import time


def main():
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    interpreter = edgetpu.make_interpreter("model_sp_int8.tflite")
    interpreter.allocate_tensors()

    # Warmup
    for i in range(50):
        image = x_test[i].reshape((1, 50, 50, 3))
        common.set_input(interpreter, image)
        interpreter.invoke()

    hits = 0
    t1 = time.time()
    for i in range(x_test.shape[0]):
        image = x_test[i].reshape((1, 50, 50, 3))
        common.set_input(interpreter, image)
        interpreter.invoke()
        classes = classify.get_classes(interpreter, top_k=1)
        if classes[0].id == y_test[i]:
            hits += 1
    t2 = time.time()

    print(f"{x_test.shape[0]} images processed in {t2 - t1:.3f} s")
    print(f"Accuracy: {hits/x_test.shape[0]:.4f}    Average time: {(t2-t1)/x_test.shape[0]*1000:.0f} ms")


if __name__ == "__main__":
    main()
