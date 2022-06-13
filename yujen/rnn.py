import torch

class RNN2(torch.nn.Module):
    def __init__(self, dim_in, hidden_size):
        super().__init__()
        self.dim_in = dim_in
        self.hidden_size = hidden_size
        self.cell = torch.nn.LSTMCell(dim_in, hidden_size=hidden_size)
        self.cell2 = torch.nn.LSTMCell(hidden_size, hidden_size=hidden_size)
        self.output = torch.nn.Linear(hidden_size, 1)

    def forward(self, data):
        if not self.training:
            return self.predict(data)
        
        # assume unbatched
        features = data['features'] # (L, dim_in)
        y = data['targets'] # (L)
        L = y.shape[0]

        assert L == features.shape[0]
        
        h1 = torch.zeros((self.hidden_size), dtype=torch.float32).cuda()
        c2 = h2 = c1 = h1
        out = [torch.tensor([y[0]]).cuda()]
        
        for i in range(L-1):
            p = out[i]
            x = torch.cat([features[i,:], p])
            assert self.dim_in == x.shape[0]
            h1, c1 = self.cell(x, (h1, c1))
            h2, c2 = self.cell2(h1, (h2,c2))
            predict = self.output(h2) + p
            out.append(predict)
        
        out = torch.squeeze(torch.stack(out),1)
        assert(out.shape == (L,)), (out.shape, y.shape, (L,))
        loss = torch.sqrt(torch.nn.functional.mse_loss(out, y))
        
        return loss
    
    def predict(self, data):
        features = data['features'] # (L, dim_in)
        y = data['targets'] # (L') where L' < L
        L = features.shape[0]
        Lp = y.shape[0]
        
        c2 = h2 = c1 = h1 = torch.zeros((self.hidden_size), dtype=torch.float32).cuda()
        out = [torch.tensor([y[0]]).cuda()]
        
        for i in range(Lp):
            p = torch.FloatTensor([y[i]]).cuda()
            x = torch.cat([features[i,:], p])
            h1, c1 = self.cell(x, (h1, c1))
            h2, c2 = self.cell2(h1, (h2, c2))
            predict = self.output(h2) + p
            out.append(predict)
        
        for i in range(Lp-1,L-2):
            p = out[i].cuda()
            x = torch.cat([features[i,:], p])
            h1, c1 = self.cell(x, (h1, c1))
            h2, c2 = self.cell2(h1, (h2, c2))
            predict = self.output(h2) + p
            out.append(predict)
        
        return torch.stack(out)