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
Model: seq length = 1
Model_v2: seq length > 1 (=memory length)
Model_v3: separate network (model v2)
Model_v4: separate network (model)
Model_v5: transformer policy only (model v3)
Model_v6: retain memory between batch. 
```