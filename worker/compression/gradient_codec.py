import io
import json
import struct
import torch
import numpy as np

def encode_bf16_gradient(pseudo_grad: dict) -> bytes:
    """
    Serialize pseudo-gradient dict to BF16 bytes for upload.
    BF16 is preferred over FP16 for gradient-like quantities to prevent overflow.
    """
    buffer = io.BytesIO()
    for name, tensor in pseudo_grad.items():
        name_bytes = name.encode("utf-8")
        # To avoid numpy lacking native bfloat16 support, we view as int16
        tensor_bf16 = tensor.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
        shape_bytes = json.dumps(list(tensor.shape)).encode("utf-8")
        
        buffer.write(struct.pack(">I", len(name_bytes)))
        buffer.write(name_bytes)
        buffer.write(struct.pack(">I", len(shape_bytes)))
        buffer.write(shape_bytes)
        buffer.write(struct.pack(">I", len(tensor_bf16)))
        buffer.write(tensor_bf16)
    return buffer.getvalue()

def decode_bf16_gradient(raw_bytes: bytes) -> dict:
    buffer = io.BytesIO(raw_bytes)
    pseudo_grad = {}
    while True:
        len_bytes = buffer.read(4)
        if not len_bytes:
            break
        name_len = struct.unpack(">I", len_bytes)[0]
        name = buffer.read(name_len).decode("utf-8")
        
        shape_len = struct.unpack(">I", buffer.read(4))[0]
        shape = json.loads(buffer.read(shape_len).decode("utf-8"))
        
        tensor_len = struct.unpack(">I", buffer.read(4))[0]
        tensor_bytes = buffer.read(tensor_len)
        
        tensor_np = np.frombuffer(tensor_bytes, dtype=np.int16).copy()
        tensor = torch.from_numpy(tensor_np).view(torch.bfloat16).reshape(shape).to(torch.float32)
        pseudo_grad[name] = tensor
    return pseudo_grad

def encode_bf16_model(model: torch.nn.Module) -> bytes:
    return encode_bf16_gradient({n: p.data for n, p in model.named_parameters()})
