package cache

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
)

func TestSetGet(t *testing.T) {
	mr := miniredis.RunT(t)
	r := New(mr.Addr(), "", 0)
	ctx := context.Background()

	if err := r.Set(ctx, "k", "v", time.Minute); err != nil {
		t.Fatalf("Set: %v", err)
	}
	got, err := r.Get(ctx, "k")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got != "v" {
		t.Fatalf("Get = %q, want %q", got, "v")
	}
}

func TestGetMissingKey(t *testing.T) {
	mr := miniredis.RunT(t)
	r := New(mr.Addr(), "", 0)

	// 不存在的 key 应返回 redis.Nil（非 nil 的 error），调用方据此判「未命中」。
	if _, err := r.Get(context.Background(), "nope"); err == nil {
		t.Fatal("missing key 应返回 error")
	}
}

func TestAllowFixedWindow(t *testing.T) {
	mr := miniredis.RunT(t)
	r := New(mr.Addr(), "", 0)
	ctx := context.Background()

	for i := 1; i <= 3; i++ {
		ok, err := r.Allow(ctx, "rate:x", 2, time.Minute)
		if err != nil {
			t.Fatalf("第 %d 次 Allow: %v", i, err)
		}
		if i <= 2 && !ok {
			t.Fatalf("第 %d 次（≤limit=2）应放行", i)
		}
		if i == 3 && ok {
			t.Fatal("第 3 次（>limit=2）应被限流")
		}
	}
}
