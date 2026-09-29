package handler

import (
	"context"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/gyx22-12/vllm-copilot/gateway/internal/grpcclient"
)

type suggestion struct {
	Category string `json:"category"`
	Text     string `json:"text"`
}

// Suggestions 处理 GET /api/suggestions：转发 gRPC 的 Suggestions，返回建议问题列表。
// 前端 app.js 期望 {suggestions: [{category, text}]}。
func Suggestions(g *grpcclient.Client) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		reply, err := g.Suggestions(ctx)
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": "获取建议失败"})
			return
		}
		out := make([]suggestion, 0, len(reply.GetSuggestions()))
		for _, s := range reply.GetSuggestions() {
			out = append(out, suggestion{Category: s.GetCategory(), Text: s.GetText()})
		}
		c.JSON(http.StatusOK, gin.H{"suggestions": out})
	}
}
