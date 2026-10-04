package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/gyx22-12/vllm-copilot/gateway/config"
	"github.com/gyx22-12/vllm-copilot/gateway/internal/auth"
	"github.com/gyx22-12/vllm-copilot/gateway/internal/cache"
	"github.com/gyx22-12/vllm-copilot/gateway/internal/grpcclient"
	"github.com/gyx22-12/vllm-copilot/gateway/internal/handler"
)

// staticDir 前端静态资源目录（与 webapp.py 共用项目根 static/，不重复维护）。
// 本地 go run . 默认相对 gateway/ 的 ../static；容器内由 STATIC_DIR 环境变量覆盖（compose 注入 /app/static）。
func staticDir() string {
	if d := os.Getenv("STATIC_DIR"); d != "" {
		return d
	}
	return "../static"
}

func main() {
	config.Load("config/config.yml")

	grpcClient, err := grpcclient.New(config.AppConfig.Grpc.Addr)
	if err != nil {
		log.Fatalf("连接 gRPC 服务失败: %v", err)
	}
	defer grpcClient.Close()

	redisClient := cache.New(config.AppConfig.Redis.Addr, config.AppConfig.Redis.Password, config.AppConfig.Redis.DB)
	if err := redisClient.Ping(context.Background()); err != nil {
		log.Printf("警告：Redis 连接失败（缓存/限流降级，不影响正确性）: %v", err)
	}

	chatHandler := handler.NewChatHandler(grpcClient, redisClient, config.AppConfig.Cache.TTL)

	r := gin.Default()

	// 前端页面：/ 返回 index.html，/static/ 托管 css/js（与 webapp.py 同一份文件）。
	dir := staticDir()
	r.Static("/static", dir)
	r.GET("/", func(c *gin.Context) {
		c.File(dir + "/index.html")
	})

	r.GET("/healthz", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	// /api/token：用配置里的 demo 凭据签发 JWT（演示网关鉴权流程）。
	r.POST("/api/token", rateLimit("rate:token:", redisClient, 5, time.Minute), bodyLimit(64<<10), func(c *gin.Context) {
		var req struct {
			Username string `json:"username"`
			Password string `json:"password"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "参数错误"})
			return
		}
		if req.Username != config.AppConfig.Auth.Username || req.Password != config.AppConfig.Auth.Password {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "用户名或密码错误"})
			return
		}
		token, err := auth.GenerateToken(req.Username, config.AppConfig.JWT.Secret, config.AppConfig.JWT.TTL)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "签发失败"})
			return
		}
		c.JSON(http.StatusOK, gin.H{"token": token})
	})

	// /api/*：限流 → JWT 鉴权（可配置开关）→ 业务。
	api := r.Group("/api")
	api.Use(bodyLimit(64 << 10))
	api.Use(rateLimit("rate:", redisClient, config.AppConfig.Rate.Limit, time.Duration(config.AppConfig.Rate.Window)*time.Second))
	api.Use(auth.Middleware(config.AppConfig.Auth.Enabled, config.AppConfig.JWT.Secret))
	api.GET("/suggestions", handler.Suggestions(grpcClient))
	api.POST("/chat", chatHandler.Chat)

	log.Printf("Go 网关已启动：%s（gRPC → %s，前端 http://localhost%s/）", config.AppConfig.Server.Port, config.AppConfig.Grpc.Addr, config.AppConfig.Server.Port)
	if err := r.Run(config.AppConfig.Server.Port); err != nil {
		log.Fatalf("启动失败: %v", err)
	}
}

// rateLimit 固定窗口限流中间件：按客户端 IP 计数，超限返回 429；Redis 故障时降级放行。
func rateLimit(keyPrefix string, r *cache.Redis, limit int, window time.Duration) gin.HandlerFunc {
	return func(c *gin.Context) {
		allowed, err := r.Allow(c.Request.Context(), keyPrefix+c.ClientIP(), limit, window)
		if err != nil {
			c.Next()
			return
		}
		if !allowed {
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{"error": "请求过于频繁，请稍后再试"})
			return
		}
		c.Next()
	}
}

// bodyLimit 限制请求体大小，防止超大 JSON 打爆内存（配合 handler 里的字段长度校验）。
// http.MaxBytesReader 在读取超限时让 Read 返回 error，ShouldBindJSON 随之失败返回 400。
func bodyLimit(maxBytes int64) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxBytes)
		c.Next()
	}
}
