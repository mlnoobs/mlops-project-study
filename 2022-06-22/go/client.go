package main

import (
	pb "audioecho/service/foo.bar"
	"context"
	"errors"
	"fmt"
	log "github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"io"
	"os"
	"runtime"
	"strings"
	"time"
)

var logger *log.Logger

func init() {
	logger = log.New()
	logger.SetReportCaller(true)
	logger.SetFormatter(&log.JSONFormatter{
		TimestampFormat: time.RFC3339Nano,
		FieldMap:        log.FieldMap{"msg": "message"},
		CallerPrettyfier: func(f *runtime.Frame) (string, string) {
			return "", fmt.Sprintf("%s:%d", formatFilePath(f.File), f.Line)
		},
	})
}

func formatFilePath(path string) string {
	arr := strings.Split(path, "/")
	return arr[len(arr)-1]
}

const (
	address = "localhost:50051"
)

func main() {
	// Setting up a connection to the server.
	conn, err := grpc.Dial(address, grpc.WithInsecure())
	if err != nil {
		logger.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	client := pb.NewAudioEchoClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
	defer cancel()

	// // =========================================
	// Echo Audio : Bi-di streaming scenario
	streamEcho, err := client.EchoAudio(ctx)
	if err != nil {
		logger.Fatal(err)
	}

	channel := make(chan struct{})
	defer func() {
		channel <- struct{}{}
	}()
	go asyncClientRecv(streamEcho, channel)

	f, err := os.Open("../sample-audio/heykakao_hello.wav")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	data := make([]byte, 1024)

	for {
		rbytes, err := f.Read(data)
		if err != nil {
			if errors.Is(err, io.EOF) {
				if err := streamEcho.CloseSend(); err != nil {
					logger.Fatal(err)
				}
				break
			} else {
				logger.Fatal(err)
			}
		}

		chunk := &pb.Chunk{Data: data[:rbytes]}
		if err := streamEcho.Send(chunk); err != nil {
			logger.Fatalln(err)
		}
	}
}

func asyncClientRecv(streamEcho pb.AudioEcho_EchoAudioClient, c chan struct{}) {
	f, _ := os.Create("../sample-audio/go-client-audio.wav")
	defer f.Close()

	for {
		chunk, err := streamEcho.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		_, _ = f.Write(chunk.GetData())
	}
	<-c
}
