package handler

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	"github.com/gyx22-12/vllm-copilot/gateway/internal/grpcclient"
	pb "github.com/gyx22-12/vllm-copilot/gateway/pb"
)

// fakeSuggestCopilot 实现 Suggestions，返回固定建议（不实现 Run，够这个测试用）。
type fakeSuggestCopilot struct {
	pb.UnimplementedCopilotServer
}

func (f *fakeSuggestCopilot) Suggestions(_ context.Context, _ *pb.SuggestionsRequest) (*pb.SuggestionsReply, error) {
	return &pb.SuggestionsReply{Suggestions: []*pb.Suggestion{
		{Category: "doc", Text: "what is vLLM?"},
		{Category: "gen", Text: "write fp8 inference"},
	}}, nil
}

func TestSuggestions(t *testing.T) {
	lis := bufconn.Listen(1 << 20)
	s := grpc.NewServer()
	pb.RegisterCopilotServer(s, &fakeSuggestCopilot{})
	go s.Serve(lis)
	t.Cleanup(s.Stop)

	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.Dial() }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("连接 bufconn: %v", err)
	}
	t.Cleanup(func() { conn.Close() })

	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET("/api/suggestions", Suggestions(grpcclient.NewWithConn(conn)))

	req := httptest.NewRequest(http.MethodGet, "/api/suggestions", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	var body struct {
		Suggestions []suggestion `json:"suggestions"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("解析响应失败: %v", err)
	}
	if len(body.Suggestions) != 2 || body.Suggestions[0].Category != "doc" || body.Suggestions[1].Text != "write fp8 inference" {
		t.Fatalf("suggestions = %+v", body.Suggestions)
	}
}
