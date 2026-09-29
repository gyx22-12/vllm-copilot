package handler

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/gyx22-12/vllm-copilot/gateway/internal/cache"
	"github.com/gyx22-12/vllm-copilot/gateway/internal/grpcclient"
)

type ChatHandler struct {
	grpc  *grpcclient.Client
	cache *cache.Redis
	ttl   time.Duration
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
func (h *ChatHandler) Chat(c *gin.Context) {
	var req struct {
		Query string `json:"query"`
	}
	if err := c.ShouldBindJSON(&req); err != nil || req.Query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "query 不能为空"})
		return
	}

	ctx := context.Background()
	cacheKey := "copilot:" + req.Query
	if cached, err := h.cache.Get(ctx, cacheKey); err == nil {
		var resp chatResponse
		if json.Unmarshal([]byte(cached), &resp) == nil {
			c.JSON(http.StatusOK, resp)
			return
		}
	}

	reply, err := h.grpc.Run(ctx, req.Query)
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
