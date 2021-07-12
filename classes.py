import torch
import os
import pandas as pd
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
    
class TextTransform:
    import torch
    import os
    import pandas as pd
    import torchaudio
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        ا 2
        ب 3
        پ 4
        ت 5
        ٹ 6
        ث 7
        ج 8
        چ 9
        ح 10
        خ 11
        د 12
        ڈ 13
        ذ 14
        ر 15
        ڑ 16
        ز 17
        ژ 18
        س 19
        ش 20
        ص 21
        ض 22
        ط 23
        ظ 24
        ع 25
        غ 26
        ف 27
        ق 28
        ک 29
        گ 30
        ل 31
        م 32
        ن 33
        ں 34
        و 35
        ہ 36
        ھ 37
        ی 38
        ے 39
        ئ 40
        ء 41
        آ 42
        اً 43
        ؤ 44
         ّ 45
         ٔ 46
         ً 47
         ُ 48
         َ 49
         ِ 50
         \u200c 51
        """
        self.char_map = {}
        self.index_map = {}
        
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')

class CNNLayerNorm(nn.Module):
    import torch
    import os
    import pandas as pd
    import torchaudio
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 

class ResidualCNN(nn.Module):
    import torch
    import os
    import pandas as pd
    import torchaudio
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)

class BidirectionalGRU(nn.Module):
    import torch
    import os
    import pandas as pd
    import torchaudio
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()
        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x

class SpeechRecognitionModel(nn.Module):
    print('====================================iuytrew=============================')
    import torch
    import os
    import pandas as pd
    import torchaudio
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

class IterMeter(object):
    import torch
    import os
    import pandas as pd
    import torchaudio
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    """keeps track of total iterations"""
    def __init__(self):
        print('Text-13')
        self.val = 0

    def step(self):
        print('Text-14')
        self.val += 1

    def get(self):
        print('Text-15')
        return self.val
