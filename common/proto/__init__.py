"""Proto stubs package — centralizes the sys.path fix for generated gRPC code."""
import os
import sys

_proto_dir = os.path.dirname(os.path.abspath(__file__))
if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)

# Re-export so other modules can do: from common.proto import inference_pb2, inference_pb2_grpc
import inference_pb2 as inference_pb2  # noqa: E402
import inference_pb2_grpc as inference_pb2_grpc  # noqa: E402
