# SDFT 与 SDPO 防遗忘机制分析 & SIRA 可借鉴点

## 一、SDFT 防遗忘机制

### 1.1 Loss 计算
- **纯 Forward KL**：`KL(teacher || student)`，在所有 completion tokens 上计算
- 代码位置：[`_compute_loss()`](sdft/Self-Distillation/distil_trainer.py:1683)
- `alpha=0` → Forward KL；`alpha=1` → Reverse KL；中间值 → Generalized JSD
- 你的 baseline 用 `alpha=0.0`（纯 Forward KL）

```python
kl_loss = kl_div(all_logps, teacher_all_logps, reduction="none", log_target=True)
per_token_loss = kl_loss.sum(-1)  # sum over vocab
loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1)).mean()
```

**关键点**：SDFT 没有 CE loss，整个训练目标就是让 student 模仿 teacher 的分布。

### 1.2 EMA Teacher（核心防遗忘机制）
- 代码位置：[`MemoryEfficientSyncRefModelCallback`](sdft/Self-Distillation/distil_trainer.py:119)
- **每步更新**：`ref_model_sync_steps=1`
- **EMA 公式**：`ref_param = (1 - α) * ref_param + α * student_param`，`α=0.01`
- 效果：teacher 缓慢跟踪 student，形成"移动锚点"

### 1.3 Teacher Prompt 增强
- 代码位置：[`_build_teacher_prompt()`](scripts/train/sdft_baseline.py:161)
- Teacher 看到 prompt + 参考答案，生成更高质量的分布
- Student 只看到 prompt，学习模仿 teacher 的输出分布
- **这意味着 teacher 分布本身就是高确定性的**

### 1.4 防遗忘总结
| 机制 | 作用 |
|------|------|
| 纯 KL loss（无 CE） | 不强制学习特定 token，只匹配分布 |
| EMA teacher | 避免 teacher 与 student 差距过大导致梯度爆炸 |
| 在线生成 | 每步用当前模型生成新数据，避免 distribution shift |

---

## 二、SDPO 防遗忘机制

### 2.1 Loss 计算
- 代码位置：[`compute_self_distillation_loss()`](sdpo/SDPO/verl/trainer/ppo/core_algos.py:1085)
- **也是 KL 蒸馏**，配置 `alpha=0.5` → Generalized JSD（混合 Forward + Reverse KL）

```python
# alpha=0.5 时：
mixture_log_probs = logsumexp([student + log(0.5), teacher + log(0.5)])
kl_teacher = KL(mixture || teacher)
kl_student = KL(mixture || student)
kl_loss = 0.5 * kl_teacher + 0.5 * kl_student  # 对称 JSD
```

- 加了 **IS clipping**（`is_clip=2.0`）：防止 off-policy 比率过大

### 2.2 EMA Teacher
- 代码位置：[`_update_teacher()`](sdpo/SDPO/verl/workers/actor/dp_actor.py:132)
- `teacher_update_rate=0.05`（比 SDFT 的 0.01 更激进）
- 同样的 EMA 公式：`teacher = (1-rate)*teacher + rate*student`

### 2.3 Reprompt 机制（独特设计）
- 从 rollout 中找到成功样本，用其作为 teacher 的"参考答案"
- Teacher 看到 prompt + 成功解答 → 生成高质量分布
- 只对有成功样本的 batch 做蒸馏（`self_distillation_mask`）

### 2.4 PPO + 蒸馏结合
- SDPO 的 `loss_mode=sdpo` 时，**完全替换 PPO loss 为蒸馏 loss**
- 不是 PPO + KL 的加权组合，而是纯蒸馏

### 2.5 防遗忘总结
| 机制 | 作用 |
|------|------|
| JSD（α=0.5） | 比纯 Forward KL 更稳定，双向约束 |
| EMA teacher（rate=0.05） | 快速跟踪 student，保持 teacher 相关性 |
| IS clipping | 防止 off-policy 梯度过大 |
| 在线 rollout | 用当前策略生成数据，避免 distribution shift |

---

## 三、SIRA 当前问题 vs SDFT/SDPO

| 维度 | SDFT | SDPO | SIRA（当前） |
|------|------|------|-------------|
| Teacher | EMA 动态更新 | EMA 动态更新 | **固定 base model** |
| 数据 | 在线生成 | 在线 rollout | **离线静态数据** |
| Loss 类型 | 纯 KL | 纯 JSD | CE + KL 混合 |
| 分段处理 | 无（全 completion） | 无（全 response） | hint/answer/anchor 分段 |
| KL 方向 | Forward | JSD | Forward |

---

## 四、可借鉴的改进建议

### 建议 1：引入 EMA Reference Model（最重要）

你当前的 ref 是固定的 base model（`disable_adapter()`）。随着训练进行，student 与 ref 的差距越来越大，KL 梯度会变得不稳定。

**实现方案**：维护一个 EMA 版本的 LoRA 参数作为 ref：

```python
# 在 SequentialTrainer.__init__ 中：
self.ema_lora_state = None  # 初始化为 None

# 在每步结束后：
def _update_ema_ref(self, alpha=0.01):
    with torch.no_grad():
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                if self.ema_lora_state is None:
                    self.ema_lora_state = {}
                if name not in self.ema_lora_state:
                    self.ema_lora_state[name] = param.data.clone()
                else:
                    self.ema_lora_state[name].mul_(1 - alpha).add_(param.data, alpha=alpha)
```

计算 ref logits 时，临时加载 EMA 参数而非 disable adapter。

### 建议 2：用 JSD 替代 Forward KL

SDPO 用 `alpha=0.5` 的 JSD，比纯 Forward KL 更稳定：
- Forward KL 在 teacher 高概率但 student 低概率时梯度很大（mode-covering）
- JSD 是对称的，梯度更平滑

```python
# 替换 _per_token_forward_kl 为 JSD：
def _per_token_jsd(logits_s, logits_t, alpha=0.5):
    log_p_s = F.log_softmax(logits_s, dim=-1)
    log_p_t = F.log_softmax(logits_t, dim=-1)
    log_m = torch.logsumexp(
        torch.stack([log_p_s + math.log(1-alpha), log_p_t + math.log(alpha)]), dim=0
    )
    kl_s = F.kl_div(log_m, log_p_s, reduction="none", log_target=True).sum(-1)
    kl_t = F.kl_div(log_m, log_p_t, reduction="none", log_target=True).sum(-1)
    return (1-alpha) * kl_s + alpha * kl_t
```

### 建议 3：降低学习率

- SDPO 用 `lr=1e-5`（你用 5e-5）
- 蒸馏任务本身梯度信号弱，但 CE on hint 梯度信号强
- 高 lr + CE 会快速改变 LoRA 参数，破坏 answer/anchor 分布
- **建议**：降到 `2e-5` 或 `1e-5`

### 建议 4：去掉 Gate 机制

SDFT 和 SDPO 都没有 gate。你的 gate 在 hint CE 高时抑制 answer KL，但这恰恰是最需要保护 answer 的时候（模型正在大幅学习 hint）。

**建议**：直接去掉 gate，让 answer KL 始终生效：
```python
l_gen = hint_loss + answer_kl_loss  # 去掉 gate
```

### 建议 5：考虑在线生成替代离线数据

SDFT/SDPO 的核心优势是**在线生成**——每步用当前模型生成 completion，然后蒸馏。这避免了 distribution shift。

你的离线数据（模型之前生成的正确推理路径）随着训练进行会越来越 off-policy。虽然实现在线生成成本高，但可以考虑：
- 每 N 步重新生成一批 answer 数据
- 或者用更大的 `answer_kl_beta` 来补偿 off-policy 问题

---

## 五、优先级排序

1. **去掉 Gate**（简单，立即可做）
2. **降低 lr 到 1e-5 ~ 2e-5**（简单）
3. **引入 EMA ref**（中等复杂度，效果显著）
4. **用 JSD 替代 Forward KL**（中等）
5. **定期重新生成数据**（复杂，长期优化）
