.PHONY: fmt
fmt:
	go fmt ./...

.PHONY: imports
imports:
	find . -name \*.go -exec goimports -w {} \;

.PHONY: protoc
protoc:
	protoc -I=../ ../audio-echo.proto --go_out=plugins=grpc:./

.PHONY: client
client:
	go run client.go

.PHONY: server
server:
	go run server.go
