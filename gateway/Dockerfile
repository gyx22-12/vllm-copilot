# 多阶段构建：golang:1.26 编译 → alpine 运行（build context = 项目根）。
# static/ 在项目根、gateway/ 在子目录，故 context 用项目根、dockerfile 放 gateway/：
#   docker build -f gateway/Dockerfile -t vllm-copilot-gateway .
FROM golang:1.26 AS build
WORKDIR /app
# 国内直连 proxy.golang.org 会被墙，走 goproxy.cn 镜像（与本地 WSL 的 GOPROXY 一致）。
ENV GOPROXY=https://goproxy.cn,direct
COPY gateway/go.mod gateway/go.sum ./
RUN go mod download
COPY gateway/ ./
RUN CGO_ENABLED=0 go build -o /copilot-gateway .

FROM alpine:3.20
WORKDIR /app
COPY --from=build /copilot-gateway /copilot-gateway
COPY static/ /app/static/
# config.yml 已 gitignore，容器内用 config.yml.example 兜底（地址由环境变量覆盖）。
COPY gateway/config/config.yml.example /app/config/config.yml
ENV STATIC_DIR=/app/static
EXPOSE 8080
ENTRYPOINT ["/copilot-gateway"]
