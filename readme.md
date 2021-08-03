## Todo List

* [X] 损失函数不收敛
* [X] asr达不到预期
* [ ] 加backdoor后对图片的影响太大

## How to run

- prerequisite
  - python=3.9.5
  - pytorch=1.9.0
  - pytorch-lightning=1.4.0
- run

```bash
python -m src.trainer.kdbackdoor
```

- tensorboard日志位于lightning_logs

## 和原代码不同的地方

- 先transform再添加trigger
- 训练 ` student model` 的时候用  `KL Divergence` 代替 `softmax_cross_entropy_with_logits`
- 可能一些其他的错误
