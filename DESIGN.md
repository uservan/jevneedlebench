# JEV 合成决策评测：实验设计

目标：判断 JEV 的退化主要来自候选增多、检索距离变长，还是历史更新难以整合。
定位是受控的检索与状态跟踪评测，不宣称测到真实环境中的多步执行能力。

## 0. 已确认的接口事实

- 端点 `POST https://api.typesafe.ai/v1/systemone`，`Authorization: Bearer <TYPESAFE key>`。
- 模型固定为 `jev-1.13.0`；请求 `{"model", "state", "questions": {"decision": {"type": "choice", "instructions", "criteria": {option_id: text}}}}`。
- 返回 `answers.decision.{choice, probabilities, confidence}` 和 `usage.{input_tokens, output_tokens}`。
- 上限：state + 最长 question 32k tokens（实测 32,984 通过，约 33.4k 报 `max_tokens_exceeded`）。本轮最长上下文 16k。
- 价格 $0.042 / 百万输入 tokens，输出免费；延迟约 0.5 s，基本不随长度变化。

## 1. 通用约定

| 项目 | 决定 |
|---|---|
| 长度构造 | tiktoken `o200k_base` 计数，只用于构造；同时记录 API 返回的 `input_tokens` |
| `target_context_tokens` | 只计上下文，不含问题和候选。填充到不超过目标值，记录实际值 |
| 语言 | 全部英文：记录、问题、候选、instructions。与原版 NIAH 和 jevbench 一致 |
| 干草堆 `haystack_type` | `kv`：全部为同格式记录；`essay`：Paul Graham 文章段落 + 少量插入记录（目标 + 错误候选对应的 user，共 4～8 条），段落边界处插入 |
| ID 空间 | `user_XXXXX`、`code_XXXXX` 5 位随机数，上下文内全局唯一，ID 与访问码无映射关系 |
| 相似 ID 控制 | 排除与目标 ID 编辑距离 ≤ 1 的 ID（改一位数字或相邻两位交换） |
| 候选顺序 | 每次重复独立随机打乱；同一场景的 R 次重复里正确位置按随机置换轮转，保证均衡 |
| 候选嵌套 | 同一 base 的候选 **集合** 嵌套（4 ⊂ 8 ⊂ 16 …），顺序不嵌套 |
| 上下文嵌套 | 同一场景的短上下文是长上下文的前缀（填充部分），关键记录按相同的相对位置插入 |
| 不传给模型 | `answer_id`、状态模拟结果、`metadata` |
| 状态码 | `ok` / `api_error` / `unsupported`（HTTP 400 `max_tokens_exceeded`）/ `invalid`（choice 不在候选内或概率无效）；只有 `ok` 且选错才算答错 |

记录格式：

```text
The access code of user_28417 is code_90311.       # 任务 A
Day 12: user_28417 joined project_C.               # 任务 B
Day 15: user_28417 left project_A.
```

约 15 tokens/条：4k 上下文约 270 条，16k 约 1,050 条。
essay 语料：Paul Graham 文章（原版 NIAH 所用），按段落切分，随机拼接到目标长度；语料下载后固定在 `corpus/` 并记录 sha256。

## 2. 三张热力图（横轴都是上下文长度 1k / 2k / 4k / 8k / 16k）

| 图 | 纵轴 | 固定 | 每次重复改变什么 | 回答 |
|---|---|---|---|---|
| ① 检索 | 目标位置 10% / 50% / 90%；kv 与 essay 各一张 | 4 个候选 | 只洗候选顺序 | 多长、在哪、什么干草堆时找不到 |
| ② 选择 | 候选数 2 / 4 / 8 / 16 / 32 / 64 | kv 堆 | 重新散布 64 条关键记录 + 洗候选顺序 | 多长、多少候选时选错 |
| ③ 状态 | 目标更新次数 1 / 2 / 4 / 8 / 16；事件查找与最终状态两道题 | 4 个候选 | 只洗候选顺序 | 多长、几次更新时整合不了 |

每格 = N 个场景 × R 次重复（默认 10 × 3）。同一场景在所有格子里复用，上下文嵌套（短是长的前缀），
候选集合嵌套（2 ⊂ 4 ⊂ … ⊂ 64），所以格子之间只差被测因素。R 次重复不是独立样本，置信区间只按场景重采样；
R 的用处是消掉顺序偏差并给出顺序敏感度。

### 任务 A：静态 KV 查找（图 ①②）

- 记录 `The access code of user_28417 is code_90311.`，每个 user 一条，顺序随机。
- 问题 `According to the context, what is the access code of user_28417?`；候选是上下文中出现过的访问码。
- 图 ②：64 条关键记录（目标 + 63 个候选 user）总在上下文里，无论本格用几个候选，因此同一长度下上下文对所有候选数相同。
- essay 干草堆：Paul Graham 文章重新分句成 40–120 tokens 的块，关键记录以整句插在块之间。

### 任务 B：历史状态跟踪（图 ③）

- 日志 `Day 12: user_28417 joined project_C.` / `left`，按天排序；每个 user 每天最多一条；只生成合法操作。
- **初始状态：所有 user 为空集合。** 最终集合大小与更新次数奇偶相同，记入 `metadata.final_set_size` 作为协变量。
- 长度通过增加其他 user 的事件扩展（事件列表前缀，按 user 分组，前缀仍合法）；目标事件的天数固定在 1–300 天内均匀分布，不随长度变。
- 更新次数 k 取同一个 16 步合法序列的前 k 步。
- 事件查找：问第 ⌊k/2⌋ 次事件，`Which project did user_X join on day N?`，候选 = 正确项目 + 日志中出现过的 3 个其他项目。
- 最终状态：规则写在问题里；候选 = 正确集合 + 3 个错误集合，来源按优先级：目标旧状态 → 漏一次更新 → 其他 user 最终状态 → 增删一个项目兜底；`(none)` 表示空集。
- 附加指标 `P(最终状态正确 | 事件查找正确)`，同场景同重复配对。

## 3. 后续可加的条件（本轮不跑）

- 相似 ID 干扰：用编辑距离 1 的 ID 替换普通 ID（0 / 3 / 20 个），长度 × 干扰数第四张图。
- 图 ② 等总输入长度对照；错误候选取自相似 ID 的 user。
- 日志嵌入 essay；非法操作和冲突记录；16k–32k 区间。

## 4. 数据格式

每条 JSONL：

```json
{
  "id": "h1_b07_kv_L4096_p0.5_r0",
  "base_id": "h1_b07",
  "task": "kv_lookup | log_event_lookup | log_final_state",
  "condition": {
    "haystack_type": "kv",
    "target_context_tokens": 4096,
    "num_options": 4,
    "target_position": 0.5,
    "distractor_level": "none",
    "target_updates": 0
  },
  "context": "...",
  "question": "...",
  "options": [{"id": "option_0", "text": "..."}],
  "answer_id": "option_1",
  "metadata": {
    "seed": 42,
    "actual_context_tokens": 4092,
    "options_tokens": 60,
    "total_input_tokens": 4180,
    "correct_index": 1,
    "target_user": "user_28417",
    "final_set_size": 2,
    "distractor_sources": ["previous_state", "skip_update_2", "other_user"]
  }
}
```

发给模型的只有 `context`、`question`、`options`。数据集生成后写入 `manifest.json`（每个文件的 sha256 和 order-independent hash），评测前校验。

## 5. 评测接口与记录

```python
class Model:
    name: str
    def choose(self, context: str, question: str, options: list[Option]) -> Result
# Result: status, choice_id, probabilities | None, usage, latency_ms, raw_error
```

- `JevModel`：context 放入 `state`，question 放 `instructions`，options 放 `criteria`。不重试（429 除外，指数退避最多 3 次）、不回退、不修补答案。
- `MockModel`：随机或「永远选第一个」，用于跑通流程。
- `ChatModel`：预留 OpenAI 兼容接口，JSON 输出 option id，无概率。
- 遇 401/403 或连续 3 次网络错误停止；未跑的题记为 `unattempted`，不算错。
- 禁止截断上下文：超限的题原样发送，按 `unsupported` 记录。
- 逐题结果 JSONL 保存请求摘要（不含 context 全文）、响应原文、时间戳。

## 6. 指标

按条件汇总：

- 准确率；随机修正准确率 `(acc − 1/k) / (1 − 1/k)`
- 正确选项平均概率；答错时的置信度（最高概率）；Brier；top-label ECE
- 延迟 p50 / p95
- 构造 tokens 与 API `input_tokens` 的均值和差值
- `api_error` / `unsupported` / `invalid` 比例，分母为全部尝试
- B 任务额外：`P(final_state 正确 | event_lookup 正确)`，同一 base 配对

置信区间：按 `base_id` 重采样 1000 次的 95% bootstrap 区间。

输出：`data/h*.jsonl`、`results/<model>/h*.jsonl`、`results/<model>/report/summary.csv` 与热力图 PNG。

## 7. 规模

| 阶段 | 每格 | 请求数 | JEV tokens | 费用 |
|---|---|---|---|---|
| 试跑（已跑） | 10 场景 × 3 次 | 3,300 | 2,900 万 | 约 1.2 美元 |
| 正式 | 100 场景 × 3 次 | 33,000 | 2.9 亿 | 约 12 美元 |

## 8. 结论判读

| 观察 | 解释 |
|---|---|
| 图 ② 随候选数下降，图 ① 平稳 | 选择困难 |
| 图 ① 随长度或位置下降；kv 掉、essay 不掉 | 检索困难，且来自同类记录中的精确匹配 |
| 图 ③ 事件查找平稳但最终状态随长度或更新次数下降 | 状态跟踪困难 |
| 事件查找答对时最终状态仍大量答错 | 整合困难独立于检索 |

## 9. 试跑结果（2026-09-23，jev-1.13.0，每格 10 场景 × 3 次）

- 图 ①（检索）：所有 30 格 100%，正确选项概率 1.0；kv 与 essay、三个位置、1k–16k 无差别。
- 图 ②（选择）：所有 30 格 100%，包括 16k + 64 候选。
- 图 ③ 事件查找：所有 25 格 100%。
- 图 ③ 最终状态：1 次更新 100%；2 次以上随长度下降，16k 时 63–77%；1k 时 16 次更新 87%。
  750 题错 79 题，其中 70 题选的是目标的**旧状态**，错选集合几乎都比正确集合小（漏掉后面的加入）。
- 3,300 次请求全部 `ok`，无失败、无超限、无无效答案；延迟 p50 173 ms、p95 302 ms；JEV 计数约为 o200k 的 1.5 倍；费用约 1.2 美元。

结论（试跑，每格只有 10 个独立场景，区间较宽）：JEV 的退化来自历史更新的整合，不来自候选数或检索距离。
