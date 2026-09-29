package config

import (
	"log"
	"os"

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
}
