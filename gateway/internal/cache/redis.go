package cache

import (
	"context"
	"time"

	"github.com/redis/go-redis/v9"
)

// Redis 封装：缓存 + 限流共用同一个客户端（对应 ExchangeApp 的 global.RedisDB）。
type Redis struct {
	client *redis.Client
}

func New(addr, password string, db int) *Redis {
	return &Redis{client: redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       db,
	})}
}

func (r *Redis) Ping(ctx context.Context) error {
	return r.client.Ping(ctx).Err()
}

// Get 读缓存；key 不存在时返回 redis.Nil（调用方按 err != nil 处理为未命中）。
func (r *Redis) Get(ctx context.Context, key string) (string, error) {
	return r.client.Get(ctx, key).Result()
}

func (r *Redis) Set(ctx context.Context, key, val string, ttl time.Duration) error {
	return r.client.Set(ctx, key, val, ttl).Err()
}

// Allow 固定窗口限流：INCR 计数，首次设窗口过期；n<=limit 放行。
// 窗口内请求超过 limit 返回 false；Redis 故障时把 error 返回给调用方降级放行。
func (r *Redis) Allow(ctx context.Context, key string, limit int, window time.Duration) (bool, error) {
	n, err := r.client.Incr(ctx, key).Result()
	if err != nil {
		return false, err
	}
	if n == 1 {
		_ = r.client.Expire(ctx, key, window)
	}
	return n <= int64(limit), nil
}
