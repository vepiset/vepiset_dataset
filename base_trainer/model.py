import sys
import timm
import random
import torchaudio
import torch.nn as nn
import torch

sys.path.append('.')



class AUG(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        bs = x.size(0)
        for i in range(bs):
            if random.uniform(0, 1) < 0.5:
                x[i, ...] = self.pitch_shift_spectrogram(x[i, ...])
            if random.uniform(0, 1) < 0.0:
                x[i, ...] = self.time_shift_spectrogram(x[i, ...])

        return x

    def do_cut_out(self, x):

        h = 128
        w = 128
        line_width = random.randint(1, 8)

        if random.uniform(0, 1) < 0.5:

            start = random.randint(0, w - line_width)
            x[:, :, start:start + line_width] = 0
        else:
            start = random.randint(0, h - line_width)
            x[:, start:start + line_width, :] = 0

        return x

    def pitch_shift_spectrogram(self, spectrogram):
        """ Shift a spectrogram along the frequency axis in the spectral-domain at
        random
        """
        nb_cols = spectrogram.size(1)
        max_shifts = nb_cols // 50  # around 5% shift
        nb_shifts = random.randint(-max_shifts, max_shifts)

        return torch.roll(spectrogram, nb_shifts, dims=[1])

    def time_shift_spectrogram(self, spectrogram):
        """ Shift a spectrogram along the frequency axis in the spectral-domain at
        random
        """
        nb_cols = spectrogram.size(2)
        max_shifts = nb_cols // 2  # around 100% shift
        nb_shifts = random.randint(-max_shifts, max_shifts)

        return torch.roll(spectrogram, nb_shifts, dims=[2])


class Transform(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(n_fft=256, hop_length=16, power=1, pad_mode='reflect')

    def forward(self, x):
        image = self.wave_transform(x)
        return image


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Net(nn.Module):
    def __init__(self, num_classes=1, add_channel=0):
        super().__init__()

        self.preprocess = Transform()

        self.model = timm.create_model('vgg16',
                                       pretrained=True,
                                       in_chans=45)

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes, bias=True)
        self.add_sleep_feature = MLP(1)
        weight_init(self.fc)

    def forward(self, x):

        # do preprocess
        bs = x.size(0)
        x = self.preprocess(x)
        x = self.model.forward_features(x)
        fm = self.avg_pooling(x)
        fm = fm.view(bs, -1)
        feature = self.dropout(fm)
        x = self.fc(feature)

        return x


if __name__ == '__main__':

    dummy_input = torch.randn(1, 74, 2000, device='cpu')
    model = Net()

    x = model(dummy_input)
