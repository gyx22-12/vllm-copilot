package handler

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	"github.com/gyx22-12/vllm-copilot/gateway/internal/cache"
	"github.com/gyx22-12/vllm-copilot/gateway/internal/grpcclient"
	pb "github.com/gyx22-12/vllm-copilot/gateway/pb"
)

// fakeCopilot 是内存里的 Copilot 实现：Run 返回固定答案并计数调用次数。
type fakeCopilot struct {
	pb.UnimplementedCopilotServer
	calls int
}

func (f *fakeCopilot) Run(_ context.Context, req *pb.QueryRequest) (*pb.AnswerReply, error) {
	f.calls++
	return &pb.AnswerReply{Answer: "answer for " + req.GetQuery(), Contexts: []string{"ctx"}}, nil
}

// newTestHandler 起一个 bufconn 内存 gRPC + miniredis，返回可测的 handler 与 fake 服务。
func newTestHandler(t *testing.T) (*ChatHandler, *fakeCopilot) {
	t.Helper()
	svc := &fakeCopilot{}

	// bufconn：用内存 buffer 冒充网络连接，起一个真的 gRPC server（不占 TCP 端口）。
	lis := bufconn.Listen(1 << 20)
	s := grpc.NewServer()
	pb.RegisterCopilotServer(s, svc)
	go s.Serve(lis)
	t.Cleanup(s.Stop)

	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return lis.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("连接 bufconn: %v", err)
	}
	t.Cleanup(func() { conn.Close() })

	mr := miniredis.RunT(t)
	c := cache.New(mr.Addr(), "", 0)

	return NewChatHandler(grpcclient.NewWithConn(conn), c, 600), svc
}

// doChat 用 httptest 走一遍真实 HTTP → handler 的完整链路，返回状态码与解析后的响应体。
func doChat(t *testing.T, h *ChatHandler, query string) (int, chatResponse) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.POST("/api/chat", h.Chat)

	body, _ := json.Marshal(map[string]string{"query": query})
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	var resp chatResponse
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	return w.Code, resp
}

func TestChatReturnsAnswer(t *testing.T) {
	h, svc := newTestHandler(t)
	code, resp := doChat(t, h, "hi")

	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	if resp.Answer != "answer for hi" {
		t.Fatalf("answer = %q, want %q", resp.Answer, "answer for hi")
	}
	if len(resp.Contexts) != 1 || resp.Contexts[0] != "ctx" {
		t.Fatalf("contexts = %v", resp.Contexts)
	}
	if svc.calls != 1 {
		t.Fatalf("gRPC 调用次数 = %d, want 1", svc.calls)
	}
}

func TestChatCacheHitSkipsGrpc(t *testing.T) {
	h, svc := newTestHandler(t)

	// 第一次：缓存未命中 → 回源 gRPC
	if code, _ := doChat(t, h, "same question"); code != http.StatusOK {
		t.Fatalf("第一次 status = %d", code)
	}
	// 第二次：同一 query 命中缓存 → 不再调 gRPC
	if code, _ := doChat(t, h, "same question"); code != http.StatusOK {
		t.Fatalf("第二次 status = %d", code)
	}

	if svc.calls != 1 {
		t.Fatalf("gRPC 调用次数 = %d, want 1（第二次应命中缓存）", svc.calls)
	}
}

func TestChatEmptyQuery(t *testing.T) {
	h, _ := newTestHandler(t)
	if code, _ := doChat(t, h, ""); code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", code)
	}
}
