package auth

import (
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt"
)

// GenerateToken 签发 HS256 JWT（复用 ExchangeApp 的写法，返回裸 token，不带 Bearer 前缀）。
func GenerateToken(username, secret string, ttlHours int) (string, error) {
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"username": username,
		"exp":      time.Now().Add(time.Duration(ttlHours) * time.Hour).Unix(),
	})
	return token.SignedString([]byte(secret))
}

// ParseToken 校验并解析 Bearer token，返回 username。
func ParseToken(tokenString, secret string) (string, error) {
	if strings.HasPrefix(tokenString, "Bearer ") {
		tokenString = tokenString[7:]
	}
	token, err := jwt.Parse(tokenString, func(t *jwt.Token) (interface{}, error) {
		if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, errors.New("unexpected signing method")
		}
		return []byte(secret), nil
	})
	if err != nil {
		return "", err
	}
	if claims, ok := token.Claims.(jwt.MapClaims); ok && token.Valid {
		username, _ := claims["username"].(string)
		return username, nil
	}
	return "", errors.New("invalid token")
}

// Middleware 返回 JWT 鉴权中间件；enabled=false 时直接放行（内部服务直连场景）。
func Middleware(enabled bool, secret string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if !enabled {
			c.Next()
			return
		}
		header := c.GetHeader("Authorization")
		if header == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "缺少 Authorization 头"})
			return
		}
		if _, err := ParseToken(header, secret); err != nil {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "token 无效或已过期"})
			return
		}
		c.Next()
	}
}
