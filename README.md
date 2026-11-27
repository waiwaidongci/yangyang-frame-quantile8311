# 定格动画擦线工时排班

苇岸偶动画工作室的制作助理记录镜头帧数、遮挡、角色交叠和支架痕迹。后端按场次隔离历史与留出镜头，用梯度提升回归估计擦线 P50 和 P90 工时，再按审片时限和单工位班次生成交接工时单。台账口径、模型和交班剧本见 [业务说明](BUSINESS.md)。

## 本地启动

```sh
cp .env.example .env
docker compose build backend
docker compose up -d --no-build --wait
docker compose exec -T backend python -m scripts.seed_demo
```

API 默认绑定 `127.0.0.1:19434`，PostgreSQL 16 绑定 `127.0.0.1:17934`。运行 `curl -fsS http://127.0.0.1:19434/api/health` 检查 API。接口描述见 `/api/openapi.json`。`backend` 提供 FastAPI，`worker` 从数据库领取计算任务，`db` 保存台账、运行和复核，`redis` 唤醒 worker。修改领域代码后重建应用镜像。`docker compose down` 保留具名数据卷。

`.env.example` 提供 `COMPOSE_PROJECT_NAME`、`BACKEND_PORT`、`DB_PORT`、`SEED_DEMO` 与本地演示 `JWT_SECRET`。启用 `SEED_DEMO` 时 API 启动会初始化演示账号和镜头台账。再次运行 seed 按账号和台账名查找已有记录。

| 账号 | 演示口令 | 可执行操作 |
|---|---|---|
| `admin@demo.local` | `Demo-2611!` | 导入、提交运行、查阅计算、审批、读取审计、导出获批报告 |
| `analyst@demo.local` | `Demo-2611!` | 导入、提交运行、查阅计算、导出获批报告 |
| `viewer@demo.local` | `Demo-2611!` | 查看台账、查阅获批运行、导出获批报告 |

## API 工作流

| 请求 | 请求内容或返回内容 |
|---|---|
| `POST /api/auth/login` | JSON `email`、`password`，返回 `access_token`。后续鉴权请求使用 `Authorization: Bearer <token>` |
| `GET /api/spec` | CSV 列顺序、运行参数默认值与范围 |
| `GET /api/datasets` | 台账列表及记录数、校验和 |
| `POST /api/datasets` | JSON `name` 和 `csv`，其中 `csv` 是完整 UTF-8 CSV 内容 |
| `POST /api/runs` | JSON `dataset_id`、`label`、`settings`，返回 `202` 和运行 ID。`settings: {}` 采用默认参数 |
| `GET /api/runs/{id}` | 查询 worker 状态及结果。完成后返回 `result.summary`、`metrics`、`tables`、`warnings` |
| `POST /api/runs/{id}/review` | 负责人在计算完成后提交 `decision` 和 `note`，通过取值 `approved` |
| `GET /api/runs/{id}/export/{key}` | 下载对应的 UTF-8 CSV，需运行已获批准 |
| `POST /api/runs/{id}/confirmations` | JSON `shot_id`、`version`、`occlusion_low`、`occlusion_high`、`rig_low`、`rig_high`。复核范围须落在台账范围内且只收窄；乱序到达按最高 `version` 生效，返回重排后的工时单 |
| `GET /api/runs/{id}/confirmations` | 该运行的复核记录列表 |
| `POST /api/runs/{id}/locks` | JSON `shot_id`、`start_minute`、`finish_minute`。锁定工位时段：不与其他锁定重叠、须赶上该镜头交接并在下班前收工，返回重排后的工时单 |
| `GET /api/runs/{id}/locks` | 该运行的锁定列表 |
| `GET /api/runs/{id}/schedule` | 按最新复核重排的交接工时单：已锁定时段原样交接，未锁定镜头按生效范围重排 |

| 可导出表 `key` | 内容 |
|---|---|
| `shift_queue` | 待排镜头的范围上下限、P50/P90、时限、预留分钟、预排起止和顺延原因 |
| `holdout_metrics` | 全部留出和分场次的模型及常量 Pinball、覆盖率 |
| `holdout_shots` | 留出镜头的输入、P50/P90 和实测工时 |
| `scene_manifest` | 历史、留出、待排场次及各自镜头数 |
| `model_profile` | 本次固定的提升参数、班次容量和训练常量分位 |

待排镜头的 `occlusion_pct`、`rig_marks` 在量准前可在台账中填 `下限-上限` 范围（如 `40-60`），工时单按范围内模型能给出的最高 P90 向上取整预留。复核登记只收窄范围，乱序到达的确认按最高版本生效后重排未锁定镜头；已锁定时段原样交接。复核与锁定按运行串行，同一工位时段只排一个镜头，排完仍须赶上各自交接并在下班前收工。

`backend/src/domain/` 提供确定性的 Python 模型、校验和报告。路由位于 `backend/src/routes/`，工作流位于 `backend/src/services/`。JWT 用户角色、运行状态与审批决定由 `backend/src/constants/workflow.py` 定义，分别用于鉴权、worker 领取与轮询、结果可见性和导出。领域模型使用 Python 标准库。
