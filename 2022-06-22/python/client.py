import grpc
import audio_echo_pb2_grpc
import audio_echo_pb2

def run():
    channel = grpc.insecure_channel('localhost:50051')

    stub = audio_echo_pb2_grpc.AudioEchoStub(channel)

    # Bi-di Streaming
    with open("../sample-audio/python-client-audio.wav", "wb") as wf:
        chunk_iterator = send_audio_chunk()
        for chunk in stub.echoAudio(chunk_iterator):
            wf.write(chunk.data)

def send_audio_chunk(blocksize=1024):
    with open("../sample-audio/heykakao_hello.wav", "rb") as rf:
        while True:
            data = rf.read(blocksize)
            if not data:
                break
            yield audio_echo_pb2.Chunk(data=data)

run()
