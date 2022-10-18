from concurrent import futures

import grpc
import audio_echo_pb2_grpc
import audio_echo_pb2

class AudioEchoServicer(audio_echo_pb2_grpc.AudioEchoServicer):

    #Bi-di Streaming
    def echoAudio(self, request_iterator, context):
        print('Echo Audio.. ')

        with open("../sample-audio/python-server-audio.wav", "wb") as wf:
            for chunk in request_iterator:
                wf.write(chunk.data)
                yield audio_echo_pb2.Chunk(data=chunk.data)


# Creating gRPC Server
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
audio_echo_pb2_grpc.add_AudioEchoServicer_to_server(AudioEchoServicer(), server)
print('Starting server. Listening on port 50051.')
server.add_insecure_port('[::]:50051')
server.start()
server.wait_for_termination()
