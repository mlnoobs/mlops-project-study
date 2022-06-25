package main

import (
	pb "audioecho/service/foo.bar"
	"fmt"
	log "github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"io"
	"net"
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
	port = ":50051"
)

type server struct{}

// bi-di RPC
func (s *server) EchoAudio(stream pb.AudioEcho_EchoAudioServer) error {
	f, _ := os.Create("../sample-audio/go-server-audio.wav")
	defer f.Close()

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			logger.Println(err)
			return err
		}

		_, _ = f.Write(chunk.GetData())

		wChunk := &pb.Chunk{Data: chunk.GetData()}
		if err := stream.Send(wChunk); err != nil {
			logger.Println(err)
			return err
		}
	}
}

func main() {
	lis, err := net.Listen("tcp", port)
	if err != nil {
		logger.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	pb.RegisterAudioEchoServer(s, &server{})
	// Register reflection service on gRPC server.
	// reflection.Register(s)
	if err := s.Serve(lis); err != nil {
		logger.Fatalf("failed to serve: %v", err)
	}
}
