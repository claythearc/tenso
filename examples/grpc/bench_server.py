import grpc
from concurrent import futures
import numpy as np
import tenso
from tenso.grpc import benchmark_msg_pb2
from tenso.grpc import benchmark_msg_pb2_grpc


class Benchmarker(benchmark_msg_pb2_grpc.BenchmarkerServicer):
    def SendTenso(self, request, context):
        # Optimized: Instant zero-copy mapping
        arr = tenso.loads(request.data)
        return benchmark_msg_pb2.BenchResponse(status="OK")

    def SendStandard(self, request, context):
        # Slow: Standard Protobuf list parsing into a NumPy array
        arr = np.array(request.data, dtype=np.float32)
        return benchmark_msg_pb2.BenchResponse(status="OK")


def serve():
    # Increase message limits to handle large tensors
    MAX_MESSAGE_LENGTH = 128 * 1024 * 1024
    options = [
        ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
    ]

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=options)
    benchmark_msg_pb2_grpc.add_BenchmarkerServicer_to_server(Benchmarker(), server)
    server.add_insecure_port("[::]:50051")
    print("Tenso Benchmark Server starting on port 50051...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
