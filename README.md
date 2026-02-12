# 每日技术情报简报（Python + GitHub Actions）

自动化流程：
1. 从 Gist API 获取 OPML 文件并解析 Feed 源。
2. 拉取每个 Feed，筛选过去 24 小时内更新的文章（UTC 计算，展示按 `Asia/Shanghai`）。
3. 优先抓取文章全文，失败回退 Feed 摘要。
4. 调用 OpenAI 兼容接口（默认 MiniMax）生成中文专业摘要。
5. 生成 HTML 简报并通过 SMTP（Gmail/SendGrid）发送到收件人。

## 文件说明
- `daily_digest.py`：主脚本。
- `requirements.txt`：依赖。
- `.github/workflows/daily_digest.yml`：每日 08:00（北京时间）自动运行。
- `tests/test_daily_digest.py`：核心测试。

## 本地运行
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest -v
python daily_digest.py --dry-run --timezone Asia/Shanghai
```

默认会输出 `daily_digest.html`（`--output-file` 可覆盖）。

## 环境变量
必填（生产）：
- `LLM_API_KEY`
- `MAIL_FROM`
- `DIGEST_RECIPIENTS`（逗号分隔）
- `SMTP_PROVIDER`（`gmail` 或 `sendgrid`）
- `SMTP_USERNAME`
- `SMTP_PASSWORD`

常用可选：
- `LLM_BASE_URL`（默认 `https://api.minimax.chat/v1`）
- `LLM_MODEL`（默认 `MiniMax-Text-01`）
- `LLM_TEMPERATURE`（默认 `0.2`）
- `LLM_MAX_INPUT_CHARS`（默认 `12000`）
- `SMTP_HOST` / `SMTP_PORT` / `SMTP_USE_TLS`
- `MAIL_SUBJECT_PREFIX`（默认 `[Daily Digest]`）
- `REQUEST_TIMEOUT_SEC`（默认 `20`）
- `HTTP_RETRY_COUNT`（默认 `3`）
- `LLM_RETRY_COUNT`（默认 `3`）

## SMTP 说明
### Gmail
- `SMTP_PROVIDER=gmail`
- 默认 `SMTP_HOST=smtp.gmail.com`
- 默认 `SMTP_PORT=587`
- `SMTP_USERNAME` 为 Gmail 地址
- `SMTP_PASSWORD` 为应用专用密码

### SendGrid
- `SMTP_PROVIDER=sendgrid`
- 默认 `SMTP_HOST=smtp.sendgrid.net`
- 默认 `SMTP_PORT=587`
- 默认 `SMTP_USERNAME=apikey`
- `SMTP_PASSWORD` 为 SendGrid API Key

## GitHub Actions Secrets 建议
在仓库 `Settings -> Secrets and variables -> Actions` 中配置：
- 必需：`LLM_API_KEY`, `MAIL_FROM`, `DIGEST_RECIPIENTS`, `SMTP_PROVIDER`, `SMTP_USERNAME`, `SMTP_PASSWORD`
- 建议：`LLM_BASE_URL`, `LLM_MODEL`, `HTTP_RETRY_COUNT`, `LLM_RETRY_COUNT`

## 安全指引
- 不要在代码、日志、Issue、PR 中粘贴任何密钥。
- 所有密钥仅放入 GitHub Secrets。
- 如密钥疑似泄露，立即轮换并废弃旧密钥。
- 使用最小权限原则创建专用发信账号与 API Key。

## 故障排查
- 报错 `LLM_API_KEY is required`：未配置模型密钥。
- 报错 `SMTP credentials are incomplete`：SMTP 参数缺失。
- 报错 `No recipients provided`：`DIGEST_RECIPIENTS` 为空。
- 大量摘要失败：提高 `LLM_RETRY_COUNT`，或降低 `--max-items`。
