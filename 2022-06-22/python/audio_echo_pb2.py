# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: audio-echo.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10\x61udio-echo.proto\x12\x07\x66oo.bar\"\x15\n\x05\x43hunk\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x32<\n\tAudioEcho\x12/\n\techoAudio\x12\x0e.foo.bar.Chunk\x1a\x0e.foo.bar.Chunk(\x01\x30\x01\x42\tZ\x07\x66oo.barb\x06proto3')



_CHUNK = DESCRIPTOR.message_types_by_name['Chunk']
Chunk = _reflection.GeneratedProtocolMessageType('Chunk', (_message.Message,), {
  'DESCRIPTOR' : _CHUNK,
  '__module__' : 'audio_echo_pb2'
  # @@protoc_insertion_point(class_scope:foo.bar.Chunk)
  })
_sym_db.RegisterMessage(Chunk)

_AUDIOECHO = DESCRIPTOR.services_by_name['AudioEcho']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z\007foo.bar'
  _CHUNK._serialized_start=29
  _CHUNK._serialized_end=50
  _AUDIOECHO._serialized_start=52
  _AUDIOECHO._serialized_end=112
# @@protoc_insertion_point(module_scope)
