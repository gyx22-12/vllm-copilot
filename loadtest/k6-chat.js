// k6 压测：Go 网关 /api/chat 的缓存命中路径（网关 → 鉴权 → 限流 → Redis 缓存）。
//
// 前置：
//   1. 三服务已起：docker compose up -d（等 grpc-server 加载完索引：docker compose logs -f grpc-server）
//   2. 压测前抬高限流阈值，否则默认 60 次/分钟会让请求全打 429：
//        实测缓存命中吞吐 ~6600 req/s、50s 约 33 万请求，故阈值需 10^7 量级（10 万会被限流拦掉）。
//        在项目根 .env 加一行 RATE_LIMIT=10000000，再 docker compose up -d --force-recreate gateway
//   3. setup() 先跑一次真实问答把结果写进 Redis，之后 default() 全命中缓存，
//      不触达 gRPC / DeepSeek（省钱、避开 6s+ 慢路径）。
//
// 跑法：
//   k6 run loadtest/k6-chat.js                          # 终端出汇总（QPS / p95 / 错误率）
//   k6 run --out json=results.json loadtest/k6-chat.js  # 出 JSON 报告

import http from 'k6/http';
import { check } from 'k6';

export const options = {
  stages: [
    { duration: '10s', target: 10 },  // 预热：爬升到 10 VU
    { duration: '30s', target: 50 },  // 加压：爬到 50 VU
    { duration: '10s', target: 0 },   // 收尾：降到 0
  ],
};

const BASE = __ENV.BASE_URL || 'http://localhost:8080';
const Q = 'What is vLLM?';

export function setup() {
  // 鉴权默认开启：先取 token（setup 只跑一次，不触发 /api/token 的 5 次/分钟限流）。
  const t = http.post(`${BASE}/api/token`, JSON.stringify({ username: 'admin', password: 'admin123' }), {
    headers: { 'Content-Type': 'application/json' },
  });
  check(t, { '取 token 成功': (x) => x.status === 200 });
  const token = t.json('token');

  // 预热：真实问答一次，把 Q 的答案写进 Redis 缓存（带 token）。
  const r = http.post(`${BASE}/api/chat`, JSON.stringify({ query: Q }), {
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
  });
  console.log(`预热缓存完成，status=${r.status}`);
  return { token };
}

export default function (data) {
  const r = http.post(`${BASE}/api/chat`, JSON.stringify({ query: Q }), {
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${data.token}` },
  });
  check(r, {
    'status 200': (x) => x.status === 200,
    '含 answer 字段': (x) => x.json('answer') !== undefined,
  });
}
