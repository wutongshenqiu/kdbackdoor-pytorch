# 代码流程

## 加载dataloader

## 构造模型
- `teacher model`
- `student model`
- `backdoor(shadow) model`

## 构造优化器
- `teacher`
    - lr(PiecewiseConstantDecay) + SGD + momentum(0.9)
- `student`
    - lr(PiecewiseConstantDecay) + SGD + momentum(0.9)
- `backdoor`
    - lr(constant) + SGD + momentum(0.9)

## 预训练 `teacher model`
- lr(constant) + SGD + momentum(0.9) + loss(CE)

## 攻击(每个epoch)
- 训练 `backdoor`(对应论文中的 `optimize trigger`)
    - 根据 `mask` 将 `trigger` 覆盖到输入
    - 对**加入 `trigger` 后**的输入进行数据增强和正则化
    - 根据下述公式进行优化(<font color="red">**这里似乎与论文中稍有区别**</font>)
    $$
        \begin{align}
        & \underset{m, \delta}{\mathrm{argmin}} \mathcal{L}_{m, \delta} (x^*, k) \\
        & \mathcal{L}_{m, \delta} = \mathcal{L}(F_t(x^*), k) + \mathcal{L}(F_s(x^*), k) + \mu \cdot \left \|  m \odot \delta \right \|_2
        \end{align}
    $$
- 训练 `teacher`(对应论文中的 `train teacher`)
    - 对输入数据进行数据增强和正则化
    - 根据下述公式进行优化(<font color="red">**这里似乎与论文中稍有区别**</font>)
    $$
        \begin{align}
        & \underset{\theta_t}{\mathrm{argmin}} \mathcal{L}_t(x, y, x^*, k, \theta_t) \\
        & \mathcal{L}_t = (1 - \beta) \mathcal{L}(F_t(x), y) + \beta \cdot \mathcal{L}(F_t(x^*), k)
        \end{align}
    $$
    - **注意这里 `teacher model` 使用的 logits 需要进行降温处理**
        > 这也是论文中提到的与正常模型蒸馏的区别，即在蒸馏的时候 `teacher model` 仍然处于可训练的状态
- 训练 `student`(对应论文中的 `shadow model`)
    - 对输入数据进行增强和正则化
    - 根据下述公式进行优化
    $$
        \begin{align}
        & \underset{\theta_s}{\mathrm{argmin}} \mathcal{L}_s(x, y, h, \theta_s) \\
        & \mathcal{L}_s = \alpha \cdot \mathcal{L}_{KD}(\theta_s, h) + (1 - \alpha) \cdot \mathcal{L} (F_s(x), y)
        \end{align}
    $$

## 模型评估
- 对 `teacher` 和 `student` 的准确率、loss、和后门攻击成功率进行评估

## 正常模型蒸馏
- 重置 `student` 模型并进行正常蒸馏，测试此时后门的攻击成功率