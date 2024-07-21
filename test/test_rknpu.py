from rknnlite.api import RKNNLite
import numpy as np
import time


def main():
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    rknn_lite = RKNNLite()
    rknn_lite.load_rknn("model_sp.rknn")
    rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    # Warmup
    for i in range(50):
        input_image = x_test[i].reshape((1, 50, 50, 3))
        predictions = rknn_lite.inference(inputs=[input_image])

    hits = 0
    t1 = time.time()
    for i in range(x_test.shape[0]):
        input_image = x_test[i].reshape((1, 50, 50, 3))
        predictions = rknn_lite.inference(inputs=[input_image])
        if np.argmax(predictions[0][0]) == y_test[i]:
            hits += 1
    t2 = time.time()

    rknn_lite.release()

    print(f"{x_test.shape[0]} images processed in {t2 - t1:.3f} s")
    print(f"Accuracy: {hits/x_test.shape[0]:.4f}    Average time: {(t2-t1)/x_test.shape[0]*1000:.0f} ms")


if __name__ == "__main__":
    main()
