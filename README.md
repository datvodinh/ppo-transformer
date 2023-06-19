# ppo-transformer

<a href="https://colab.research.google.com/github/datvodinh10/ppo-transformer/blob/main/main.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

```
ENV/model
│   agent.py
│   gru.py
│   memory.py
│   model.py
│   rma.py
│   rollout_buffer.py
│   trainer.py
│   transformer.py
│   writer.py
│
```

```
(...) mean inherit from.
Model: seq length = 1 x
Model_v2: seq length > 1 (=memory length) x
Model_v3: separate network (model v2) x
Model_v4: separate network (model) x
Model_v5: transformer policy only (model v3) x
Model_v6: replace relu with gelu,  (model v5)  
Model_v7: like model 6 but (model)
Model_v8: like model 6 but (model v2)
Model_v9: sequence_length = max_eps_length, add dropout, many optimize
model_v10: fix padding and normalization (modelv9)
```