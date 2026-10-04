package handler

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/sync/singleflight"

	"github.com/gyx22-12/vllm-copilot/gateway/internal/cache"
	"github.com/gyx22-12/vllm-copilot/gateway/internal/grpcclient"
	pb "github.com/gyx22-12/vllm-copilot/gateway/pb"
)

type ChatHandler struct {
	grpc  *grpcclient.Client
	cache *cache.Redis
	ttl   time.Duration
	sf    singleflight.Group // 合并并发相同 query 的回源，防缓存击穿
}

func NewChatHandler(g *grpcclient.Client, c *cache.Redis, ttlSeconds int) *ChatHandler {
	return &ChatHandler{grpc: g, cache: c, ttl: time.Duration(ttlSeconds) * time.Second}
}

type chatResponse struct {
	Answer   string   `json:"answer"`
	Contexts []string `json:"contexts"`
}

// Chat 处理 POST /api/chat：query → Redis 缓存 → gRPC Run → 回写缓存 → 返回。
// 缓存用 cache-aside（先查、命中返回、未命中回源再写），与 ExchangeApp 文章缓存同款。
// 字段同时兼容 question（前端 app.js 用）与 query（curl 用），二者取非空者。
func (h *ChatHandler) Chat(c *gin.Context) {
	var req struct {
		Query    string `json:"query"`
		Question string `json:"question"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "参数错误"})
		return
	}
	q := req.Query
	if q == "" {
		q = req.Question
	}
	if q == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "问题不能为空"})
		return
	}
	if len(q) > 2000 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "问题过长（限 2000 字符）"})
		return
	}

	// 用请求自身的 ctx：客户端断开即取消缓存读取与回源，不再 context.Background 拖满 90s。
	ctx := c.Request.Context()
	cacheKey := "copilot:" + q
	if cached, err := h.cache.Get(ctx, cacheKey); err == nil {
		var resp chatResponse
		if json.Unmarshal([]byte(cached), &resp) == nil {
			c.JSON(http.StatusOK, resp)
			return
		}
	}

	reply, err := h.runDeduped(ctx, q, cacheKey)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": fmt.Sprintf("gRPC 调用失败：%v", err)})
		return
	}
	resp := chatResponse{Answer: reply.GetAnswer(), Contexts: reply.GetContexts()}

	if data, err := json.Marshal(resp); err == nil {
		_ = h.cache.Set(ctx, cacheKey, string(data), h.ttl)
	}

	c.JSON(http.StatusOK, resp)
}

// runDeduped 用 singleflight 合并并发相同 query：缓存过期瞬间大量并发打同一条问题，
// 只回源一次、其余请求共享结果（防缓存击穿）。用 DoChan + select 而非 Do：
// 共享那次 gRPC 调用挂独立 ctx（不被第一个请求的断开连带取消），
// 但每个请求在自身 ctx 取消（客户端断开）时能提前退出等待。
func (h *ChatHandler) runDeduped(ctx context.Context, q, cacheKey string) (*pb.AnswerReply, error) {
	ch := h.sf.DoChan(cacheKey, func() (interface{}, error) {
		return h.grpc.Run(context.Background(), q)
	})
	select {
	case res := <-ch:
		if res.Err != nil {
			return nil, res.Err
		}
		reply, ok := res.Val.(*pb.AnswerReply)
		if !ok {
			return nil, fmt.Errorf("singleflight 返回类型异常")
		}
		return reply, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}
