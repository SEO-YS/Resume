import os, argparse, traceback, glob, random, itertools, time, torch, threading, queue
import numpy as np
import torch.optim as optim
from models.tacotron import post_CBHG
from torch.nn import L1Loss
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from util.hparams import *


data_dir = 'text_data'
mel_list = sorted(glob.glob(os.path.join(data_dir + '/mel', '*.npy')))
spec_list = sorted(glob.glob(os.path.join(data_dir + '/spec', '*.npy')))
mel_len = np.load(os.path.join(data_dir + '/mel_len.npy'))


def DataGenerator():
    while True:
        idx_list = np.random.choice(len(mel_list), batch_group,
                                    replace=False)  # mel_list의 길이에 해당하는 인덱스 중에서 batch_group 개수만큼 랜덤하게 선택합니다.
        idx_list = sorted(idx_list)
        idx_list = [idx_list[i: i + batch_size] for i in
                    range(0, len(idx_list), batch_size)]  # 인덱스들을 배치 크기인 batch_size 만큼 나누어 리스트로 만듭니다.
        random.shuffle(idx_list)

        for idx in idx_list:  # 배치들에 대해서 반복합니다.
            random.shuffle(idx)

            mel = [torch.from_numpy(np.load(mel_list[mel_len[i][1]])) for i in
                   idx]  # 해당 배치의 mel_list에서 인덱스에 해당하는 mel 스펙트로그램 데이터를 로드하고, Pytorch 텐서로 변환하여 mel 리스트에 저장합니다.
            spec = [torch.from_numpy(np.load(spec_list[mel_len[i][1]])) for i in
                    idx]  # 해당 배치의 spec_list에서 인덱스에 해당하는 Spectrogram스펙트로그램 데이터를 로드하고, pytorch 텐서로 변환하여 spec리스트에 저장합니다.

            mel = pad_sequence(mel,
                               batch_first=True)  # mel리스트에 있는 Mel 스펙트로그램 데이터들을 패딩하여 배치 크기에 맞춥니다. batch_frist =True로 설정하여 배치 차원을 첫 번째 자원으로 설정합니다.
            spec = pad_sequence(spec, batch_first=True)  # spec리스트에 있는 Spectrogram 스펙트로그램 데이터들을 패딩하여 배치 크기에 맟춥니다.

            yield [mel, spec]  # Mel 스펙트로그램과 Spectrongram 스펙트로그램을 반환하니다. yield 키워드는 제너레이터를 만들 때 사용되며, 데이터를 생성하는 데 사용됩니다.


class Generator(threading.Thread):
    def __init__(self, generator):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(
            8)  # 크기가 8인 큐(queue.Queue(8))를 생성하여 self.queue라는 인스턴스 변수에 저장합니다. 이 큐는 최대 8개의 항목을 담을 수 있습니다
        self.generator = generator
        self.start()

    def run(self):
        for item in self.generator:  # self.generator에서 생성된 데이터를 하나씩 가져옵니다.
            self.queue.put(item)  # 가져온 데이터를 큐에 넣습니다.
        self.queue.put(None)  # 모든 데이터를 큐에 넣은 후에 None을 큐에 추가하여 데이터 생성이 끝났음을 알립니다.

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item


def train(args):
    train_loader = Generator(DataGenerator())  # 데이터를 생성하는 DataGenerator()를 이용하여 학습 데이터 로더를 생성

    model = post_CBHG(K=8, conv_dim=[256, mel_dim]).cuda()  # post_CBHG 모델을 생성

    optimizer = optim.Adam(model.parameters())

    step, epochs = 0, 0
    if args.checkpoint is not None:  # 만약 이전에 저정한 체크포인트 파일이 존재한다면
        ckpt = torch.load(args.checkpoint)  # 체크포인트 파일로부터 저장된 모델과 옵티마아저의 상태를 로드합니다.
        model.load_state_dict(ckpt['model'])  # 모델의 상태를 로드한 값으로 업데이트합니다.
        optimizer.load_state_dict(ckpt['optimizer'])  # 옵티마이저의 상태를 로드한 값으로 업데이트합니다.
        step = ckpt['step'],  # 로드한 체크포인트에서 학습 스텝을 가져옵니다.
        step = step[0]
        epoch = ckpt['epoch']  # 로드한 체크포인트에서 에폭을 가져옵니다
        print('Load Status: Epoch %d, Step %d' % (epoch, step))

    torch.backends.cudnn.benchmark = True

    try:
        for epoch in range(300):  # 무한 루프를 이용하여 에폭을 증가시키며 학습을 진행합니다.
            for _ in range(batch_group):  # batch_group 만큼 반복하면서 배치 단위로 학습을 진행합니다.
                start = time.time()
                mel, target = train_loader.next()  # 데이터 로더에서 학습에 필요한 mel과 target을 가져옵니다.
                mel = mel.float().cuda()
                target = target.float().cuda()

                pred = model(mel)  # 모델을 사용하여 mel로 부터 예측값pred을 얻습니다.
                loss = L1Loss()(pred, target)

                model.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1
                print(
                    'epoch: {} ,step: {}, loss: {:.5f}, {:.3f} sec/step'.format(epoch, step, loss, time.time() - start))

                if step % checkpoint_step == 0:  # 일정 주기마다 체크포인트를 저장하기 위해 checkpoint_step으로 나눈 나머지를 검사합니다.
                    save_dir = 'ckpt/' + args.name + '/2'
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'epoch': epoch
                    }, os.path.join(save_dir, 'ckpt-{}.pt'.format(
                        step)))  # 체크포인트 파일을 저장합니다. 모델 상태, 옵티마이저 상태, 학습 스텝, 에폭 정보가 저장됩니다.

    except Exception as e:
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', default="")
    parser.add_argument('--name', '-n', required=True)
    args = parser.parse_args()
    save_dir = os.path.join('ckpt/' + args.name, '2')
    os.makedirs(save_dir, exist_ok=True)
    train(args)
