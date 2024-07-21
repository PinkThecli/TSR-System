from openvino.inference_engine import IECore
import numpy as np
import time


def main():
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    ie = IECore()
    net = ie.read_network(model="model_sp.xml", weights="model_sp.bin")
    exec_net = ie.load_network(network=net, device_name="MYRIAD")  # ["CPU", "GPU", "MYRIAD"]

    input_layers = list(exec_net.input_info)
    output_layers = list(exec_net.outputs)
    print("Net inputs:", input_layers)
    print("Net outputs:", output_layers)

    # Warmup
    for i in range(50):
        input_image = x_test[i].reshape((1, 50, 50, 3))
        predictions = exec_net.infer({input_layers[0]: input_image})

    hits = 0
    t1 = time.time()
    for i in range(x_test.shape[0]):
        input_image = x_test[i].reshape((1, 50, 50, 3))
        predictions = exec_net.infer({input_layers[0]: input_image})
        if np.argmax(predictions[output_layers[0]]) == y_test[i]:
            hits += 1
    t2 = time.time()

    print(f"{x_test.shape[0]} images processed in {t2 - t1:.3f} s")
    print(f"Accuracy: {hits/x_test.shape[0]:.4f}    Average time: {(t2-t1)/x_test.shape[0]*1000:.0f} ms")


if __name__ == "__main__":
    main()
