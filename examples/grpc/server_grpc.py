import grpc
from concurrent import futures
import numpy as np
import tenso
from tenso.grpc import tenso_msg_pb2
from tenso.grpc import tenso_msg_pb2_grpc


class TensorInferenceServicer(tenso_msg_pb2_grpc.TensorInferenceServicer):
    def Predict(self, request, context):
        # 1. Instant zero-copy deserialization from the gRPC bytes field
        input_tensor = tenso.loads(request.tensor_packet)
        print(f"Received {request.model_name} request. Shape: {input_tensor.shape}")

        # 2. Perform high-performance computation
        # (Example: Simple doubling operation)
        result = input_tensor * 2

        # 3. Serialize result back to Tenso format
        result_packet = tenso.dumps(result)

        # Convert memoryview to bytes for gRPC compatibility
        return tenso_msg_pb2.PredictResponse(
            result_packet=bytes(result_packet), status="SUCCESS"
        )


def serve():
    # Define options to increase the message size limit (e.g., 128MB)
    MAX_MESSAGE_LENGTH = 128 * 1024 * 1024
    options = [
        ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
    ]

    # Pass the options to the server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)

    tenso_msg_pb2_grpc.add_TensorInferenceServicer_to_server(
        TensorInferenceServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    print("Tenso gRPC Server starting on port 50051...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
