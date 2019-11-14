from data import RapDataset
from model import DATModel
from utils import eta
import torch
import time

if __name__ == '__main__':
    ### initialize
    epochs = 50
    batch_size = 6
    n_layers = 12

    model = DATModel(512, 4, [3, 4]).cuda()
    rd = RapDataset(batch_size=batch_size)
    num_batches = len(rd)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=16000)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    total_iters = 0

    for epoch in range(epochs):
        it = iter(rd)

        b = 0
        times = []
        for batch in it:
            b += 1
            start = time.time()
            opt.zero_grad()
            total_loss = 0.
            for i, subbatch in enumerate(batch):
                subbatch = [torch.from_numpy(x).cuda() for x in subbatch]
                subbatch[0] = subbatch[0].long()
                out = model(*subbatch, first_part=not i)
                out = out.permute(0, 2, 1)
                loss = criterion(out, subbatch[0])
                total_loss += loss.item()
                loss.backward()
            end = time.time()
            times.append(end - start)
            remaining = eta(times, num_batches - b)
            print(epoch + 1, b, '/', num_batches, remaining, total_loss / (i + 1))
            opt.step()

            total_iters += 1
            if total_iters % 1000 == 0:
                torch.save(model.state_dict(), '../models/' + str(total_iters) + '.pth')
