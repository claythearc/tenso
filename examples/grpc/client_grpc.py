import grpc
import numpy as np
import tenso
from tenso.grpc import tenso_msg_pb2
from tenso.grpc import tenso_msg_pb2_grpc
import time


def run():
    MAX_MESSAGE_LENGTH = 128 * 1024 * 1024
    options = [
        ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
    ]

    # Create the channel with options
    with grpc.insecure_channel("localhost:50051", options=options) as channel:
        stub = tenso_msg_pb2_grpc.TensorInferenceStub(channel)
        # 1. Create a large dummy tensor (e.g., a 1024x1024 matrix)
        data = np.random.rand(1024, 1024).astype(np.float32)
        print(f"Sending tensor: {data.shape} ({data.nbytes / 1024 / 1024:.2f} MB)")

        # 2. Serialize to Tenso format
        t0 = time.perf_counter()
        packet = tenso.dumps(data)

        # 3. Pack into gRPC message and send
        request = tenso_msg_pb2.PredictRequest(
            model_name="production_model_v1", tensor_packet=bytes(packet)
        )
        response = stub.Predict(request)

        # 4. Deserialize result instantly
        result = tenso.loads(response.result_packet)
        t_total = (time.perf_counter() - t0) * 1000

        print(f"Response status: {response.status}")
        print(f"Result mean: {result.mean():.4f}")
        print(f"Roundtrip + Serialization time: {t_total:.2f} ms")


if __name__ == "__main__":
    run()
