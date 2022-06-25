# grpc binary 양방향 통신 테스트
client에서 sample-audio/heykakao_hello.wav 파일을 stream 전송하고,<p>
server는 받은 binary를 echo하는 예제이다.

-----------------------------------------------------------------
### python 패키지 설치
```shell
$ python -m pip install grpcio
$ python -m pip install grpcio-tools
````

### python grpc binary 양방향 통신 테스트
```shell
$ cd python
$ make protoc
$ make server
$ make client
```

### go 패키지 설치
```shell
$ brew install protobuf
$ protoc --version  # Ensure compiler version is 3+

$ go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.28
$ go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.2
$ export PATH="$PATH:$(go env GOPATH)/bin"
```

### go grpc binary 양방향 통신 테스트
```shell
$ cd go
$ make protoc
$ make server
$ make client
```
