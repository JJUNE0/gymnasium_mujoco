from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# 예시 데이터: 확률 분포
import torch
import numpy as np

for epoch in range(100):
    # 확률 분포 예시 (예: softmax)
    probabilities = torch.softmax(torch.randn(5), dim=0)  # 5개 액션에 대한 확률 분포

    # 확률 분포를 히스토그램으로 기록
    writer.add_histogram('Action Distribution', probabilities, epoch)

writer.close()