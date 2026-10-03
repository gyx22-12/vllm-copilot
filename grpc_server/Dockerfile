# Python gRPC 服务镜像：CPU torch 底座 + 全部依赖 + 项目代码。
# 模型（~2.9GB）/语料（156MB）不进镜像，运行时从宿主机 volume 挂载（见 docker-compose.yml）。
FROM python:3.10-slim
WORKDIR /app

# 国内直连 pypi.org 的 files.pythonhosted.org 会卡死（numpy 等稍大 wheel 下载中断），
# 统一走阿里云 PyPI 镜像（HTTPS 全量镜像；清华已不可用）。
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

# 先装 CPU 版 torch（避免 pip 默认拉 CUDA 版，数 GB）。装法分两步，规避两个坑：
#   1) 先把 torch 的纯 Python 依赖从阿里云镜像装好（包小、网络稳）；
#   2) 再装本地预下载的 CPU wheel，--no-deps 跳过依赖解析，完全离线。
# 为什么第 2 步不继续用 --index-url 在线拉：download.pytorch.org 会 302 到 Cloudflare R2，
# 国内直连常把 196MB 的 wheel 截断/污染，导致 sha256 与索引页对不上（build 反复失败）。
# 所以把「脆弱的下载」挪到宿主机：curl 带重试下好、sha256 校验通过后放这里，构建只管 COPY+安装。
# 换机器重新生成（下完 sha256 应为 4ce2a6ae...，见 pytorch 索引页）：
#   curl -L -o grpc_server/torch-2.14.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl \
#     'https://download.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl'
RUN pip install --no-cache-dir fsspec filelock typing-extensions sympy networkx jinja2 mpmath
COPY grpc_server/torch-2.14.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl /tmp/
RUN pip install --no-cache-dir --no-deps /tmp/torch-2.14.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl

# 核心依赖 + gRPC 依赖（server.py 需要 grpcio/protobuf，拆在 requirements-grpc.txt）。
COPY requirements.txt grpc_server/requirements-grpc.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-grpc.txt

COPY . .
CMD ["python3", "grpc_server/server.py"]
