package config

import (
	"log"
	"os"
	"strconv"

	"github.com/spf13/viper"
)

// Config 网关全部配置，结构与 config.yml 对应（mapstructure 标签）。
type Config struct {
	Server struct {
		Port string `mapstructure:"port"`
	} `mapstructure:"server"`
	Grpc struct {
		Addr string `mapstructure:"addr"`
	} `mapstructure:"grpc"`
	Redis struct {
		Addr     string `mapstructure:"addr"`
		Password string `mapstructure:"password"`
		DB       int    `mapstructure:"db"`
	} `mapstructure:"redis"`
	JWT struct {
		Secret string `mapstructure:"secret"`
		TTL    int    `mapstructure:"ttl"` // 单位：小时
	} `mapstructure:"jwt"`
	Cache struct {
		TTL int `mapstructure:"ttl"` // 单位：秒
	} `mapstructure:"cache"`
	Auth struct {
		Enabled  bool   `mapstructure:"enabled"`
		Username string `mapstructure:"username"`
		Password string `mapstructure:"password"`
	} `mapstructure:"auth"`
	Rate struct {
		Limit  int `mapstructure:"limit"`  // 每窗口最大请求数
		Window int `mapstructure:"window"` // 窗口秒数
	} `mapstructure:"rate"`
}

var AppConfig Config

func Load(path string) {
	viper.SetConfigFile(path)
	viper.AutomaticEnv()
	if err := viper.ReadInConfig(); err != nil {
		log.Fatalf("读取配置失败: %v", err)
	}
	if err := viper.Unmarshal(&AppConfig); err != nil {
		log.Fatalf("解析配置失败: %v", err)
	}
	// 环境变量覆盖（与 ExchangeApp 一致：env > config.yml）
	if s := os.Getenv("JWT_SECRET"); s != "" {
		AppConfig.JWT.Secret = s
	}
	// 容器编排（docker-compose）里用服务名取代 localhost，这里用环境变量覆盖。
	if s := os.Getenv("GRPC_ADDR"); s != "" {
		AppConfig.Grpc.Addr = s
	}
	if s := os.Getenv("REDIS_ADDR"); s != "" {
		AppConfig.Redis.Addr = s
	}
	// 压测时抬高限流阈值（默认 60 次/分钟会让 k6 全打 429），同样走环境变量覆盖。
	if s := os.Getenv("RATE_LIMIT"); s != "" {
		if v, err := strconv.Atoi(s); err == nil {
			AppConfig.Rate.Limit = v
		}
	}
	if s := os.Getenv("RATE_WINDOW"); s != "" {
		if v, err := strconv.Atoi(s); err == nil {
			AppConfig.Rate.Window = v
		}
	}
	// 鉴权开关可用环境变量覆盖（compose 里 AUTH_ENABLED=1 开启），防生产误留默认关鉴权。
	if s := os.Getenv("AUTH_ENABLED"); s != "" {
		if v, err := strconv.ParseBool(s); err == nil {
			AppConfig.Auth.Enabled = v
		}
	}
}
