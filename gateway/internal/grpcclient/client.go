package grpcclient

import (
	"context"
	"time"

	pb "github.com/gyx22-12/vllm-copilot/gateway/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Client 是 Go 网关到 Python gRPC 服务的客户端（复用 go-grpc-demo 的 grpc.NewClient 写法）。
type Client struct {
	conn *grpc.ClientConn
	pb   pb.CopilotClient
}

func New(addr string) (*Client, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	return &Client{conn: conn, pb: pb.NewCopilotClient(conn)}, nil
}

func (c *Client) Close() error {
	return c.conn.Close()
}

// Run 执行一次问答；agent 多步推理可能较慢，给 90s 超时。
func (c *Client) Run(ctx context.Context, query string) (*pb.AnswerReply, error) {
	ctx, cancel := context.WithTimeout(ctx, 90*time.Second)
	defer cancel()
	return c.pb.Run(ctx, &pb.QueryRequest{Query: query})
}
