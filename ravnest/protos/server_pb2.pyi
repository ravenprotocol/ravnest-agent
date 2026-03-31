import tensor_pb2 as _tensor_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BufferStatusReply(_message.Message):
    __slots__ = ["status"]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: str
    def __init__(self, status: _Optional[str] = ...) -> None: ...

class CheckBufferStatus(_message.Message):
    __slots__ = ["name", "type"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: str
    def __init__(self, name: _Optional[str] = ..., type: _Optional[str] = ...) -> None: ...

class CheckGatherIteration(_message.Message):
    __slots__ = ["ring_id"]
    RING_ID_FIELD_NUMBER: _ClassVar[int]
    ring_id: int
    def __init__(self, ring_id: _Optional[int] = ...) -> None: ...

class CheckReduceIteration(_message.Message):
    __slots__ = ["ring_id"]
    RING_ID_FIELD_NUMBER: _ClassVar[int]
    ring_id: int
    def __init__(self, ring_id: _Optional[int] = ...) -> None: ...

class DataChunk(_message.Message):
    __slots__ = ["buffer", "data_size", "type"]
    BUFFER_FIELD_NUMBER: _ClassVar[int]
    DATA_SIZE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    buffer: bytes
    data_size: int
    type: str
    def __init__(self, buffer: _Optional[bytes] = ..., type: _Optional[str] = ..., data_size: _Optional[int] = ...) -> None: ...

class GatherChunk(_message.Message):
    __slots__ = ["data_chunk", "ring_id"]
    DATA_CHUNK_FIELD_NUMBER: _ClassVar[int]
    RING_ID_FIELD_NUMBER: _ClassVar[int]
    data_chunk: DataChunk
    ring_id: int
    def __init__(self, ring_id: _Optional[int] = ..., data_chunk: _Optional[_Union[DataChunk, _Mapping]] = ...) -> None: ...

class GatherIterationReply(_message.Message):
    __slots__ = ["iteration"]
    ITERATION_FIELD_NUMBER: _ClassVar[int]
    iteration: int
    def __init__(self, iteration: _Optional[int] = ...) -> None: ...

class PingRequest(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: str
    def __init__(self, data: _Optional[str] = ...) -> None: ...

class PingResponse(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: str
    def __init__(self, data: _Optional[str] = ...) -> None: ...

class ReceivedChunk(_message.Message):
    __slots__ = ["reply"]
    REPLY_FIELD_NUMBER: _ClassVar[int]
    reply: bool
    def __init__(self, reply: bool = ...) -> None: ...

class ReduceChunk(_message.Message):
    __slots__ = ["data_chunk", "ring_id"]
    DATA_CHUNK_FIELD_NUMBER: _ClassVar[int]
    RING_ID_FIELD_NUMBER: _ClassVar[int]
    data_chunk: DataChunk
    ring_id: int
    def __init__(self, ring_id: _Optional[int] = ..., data_chunk: _Optional[_Union[DataChunk, _Mapping]] = ...) -> None: ...

class ReduceIterationReply(_message.Message):
    __slots__ = ["iteration"]
    ITERATION_FIELD_NUMBER: _ClassVar[int]
    iteration: int
    def __init__(self, iteration: _Optional[int] = ...) -> None: ...

class SendLatestWeights(_message.Message):
    __slots__ = ["param_names"]
    PARAM_NAMES_FIELD_NUMBER: _ClassVar[int]
    param_names: bytes
    def __init__(self, param_names: _Optional[bytes] = ...) -> None: ...

class WeightsChunk(_message.Message):
    __slots__ = ["tensor_chunk"]
    TENSOR_CHUNK_FIELD_NUMBER: _ClassVar[int]
    tensor_chunk: _tensor_pb2.TensorChunk
    def __init__(self, tensor_chunk: _Optional[_Union[_tensor_pb2.TensorChunk, _Mapping]] = ...) -> None: ...
