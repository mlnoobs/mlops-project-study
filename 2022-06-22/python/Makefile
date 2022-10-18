.PHONY: protoc
protoc:
	python3 -m grpc_tools.protoc -I=../ --python_out=. --grpc_python_out=. ../audio-echo.proto

.PHONY: client
client:
	python3 client.py

.PHONY: server
server:
	python3 server.py
