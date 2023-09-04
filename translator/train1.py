import os, argparse, traceback, glob, random, itertools, time, torch, threading, queue
import numpy as np
import torch.optim as optim
from models.tacotron import Tacotron
from torch.nn import L1Loss
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from util.text import sequence_to_text
from util.plot_alignment import plot_alignment
from util.hparams import *

data_dir = 'text_data'
text_list = sorted(glob.glob(os.path.join(data_dir + '/text', '*.npy')))
mel_list = sorted(glob.glob(os.path.join(data_dir + '/mel', '*.npy')))
dec_list = sorted(glob.glob(os.path.join(data_dir + '/dec', '*.npy')))

fn = os.path.join(data_dir + '/mel_len.npy')
if not os.path.isfile(fn):  # 만약 fn 경로에 해당하는 파일이 존재하지 않는다면 다음을 수행합니다.
    mel_len_list = []
    for i in range(len(mel_list)):  # 멜 스펙트로그램 파일 리스트 'mel_list'의 각 요소에 대해 루프를 실행합니다.
        mel_length = np.load(mel_list[i]).shape[
            0]  # mel_list[i]에 해당하는 멜 스펙트로그램 파일을 로드하고, 해당 스펙트로그램의 시간 축 길이를 mel_length변 수에 저장합니다.
        mel_len_list.append([mel_length, i])  # 각 멜 스펙트로그램 파일의 길이 정보와 인덱스가 저장됩니다.
    mel_len = sorted(mel_len_list)
    np.save(os.path.join(data_dir + '/mel_len.npy'),
            np.array(mel_len))  # 정렬된 mel_len을 NumPy 배열로 저장하고,해당 배열을 저장합니다. 이로써 멜 스펙트로그램 파일의 길이 정보가 저장되었습니다.

text_len = np.load(os.path.join(data_dir + '/text_len.npy'))
mel_len = np.load(os.path.join(data_dir + '/mel_len.npy'))


def DataGenerator():
    while True:
        idx_list = np.random.choice(len(mel_list), batch_group,
                                    replace=True)  # mel_list의 인덱스 중에서 batch_group 개수만큼 랜덤하게 인덱스를 선택하여 idx_list에 저장합니다. replace=False옵션으로 중복 선택을 허용하지 않습니다.
        idx_list = sorted(idx_list)
        idx_list = [idx_list[i: i + batch_size] for i in
                    range(0, len(idx_list), batch_size)]  # idx_list를 batch_size크기의 작은 그룹으로 나눕니다.
        random.shuffle(idx_list)  # 작은 그룹들의 순서를 무작위로 섞습니다.

        for idx in idx_list:
            random.shuffle(idx)  # 각 작은 그룹 내에서 해당 그룹의 인덱스들의 순서를 무작위로 섞습니다.

            text = [torch.from_numpy(np.load(text_list[mel_len[i][1]])) for i in
                    idx]  # idx에 해당하는 인덱스를 사용하여 텍스트 데이터를 로드하고,Pytorch tensor로 변환하여 text리스트에 저장
            dec = [torch.from_numpy(np.load(dec_list[mel_len[i][1]])) for i in
                   idx]  # idx에 해당하는 인덱스를 사용하여 디코더 입력 데이터를 로드하고,Pytorch tensor로 변환하여 dec리스트에 저장합니다.
            mel = [torch.from_numpy(np.load(mel_list[mel_len[i][1]])) for i in
                   idx]  # idx에 해당하는 인덱스를 멜 스펙트로그램 데이터를 로드하고, Pytorch tensor로 변환하여 mel 리스트에 저장합니다.

            text_length = torch.tensor([text_len[mel_len[i][1]] for i in idx],
                                       dtype=torch.int32)  # idx에 해당하는 인덱스를 사용하여 텍스트 데이터의 길이 정로를 가져와 text_length tensor를 생성합니다.
            text_length, _ = text_length.sort(descending=True)  # text_lenght tensor를 길이 기준으로 내림차순으로 정렬합니다.

            text = pad_sequence(text, batch_first=True)  # text리스트의 tensor들을 패딩하여(배치 크기, 최대 시퀀스 길이)형태의 tensor로 만듭니다.
            dec = pad_sequence(dec, batch_first=True)  # dec리스트의 tensor들을 패딩하여(배치 크기, 최대 시퀀스 길이)형태의 tensor로 만듭니다.
            mel = pad_sequence(mel, batch_first=True)  # mel리스트의 tensor들을 패딩하여(배치 크기, 최대 시퀀스 길이)형태의 tensor로 만듭니다.

            yield [text, dec, mel,
                   text_length]  # 이제까지 처리한 데이터들을 리스트로 묶어서 제너레이터가 반환하는 값으로 지정합니다.즉, 각 배치마다'text','dec','mel','text_length'데이터를 반환합니다.


class Generator(threading.Thread):
    def __init__(self, generator):
        threading.Thread.__init__(self)  # 초기화 메서드
        self.queue = queue.Queue(8)  # 크기가 8인 큐 (queue.Queue(8))를 생성한다
        self.generator = generator
        self.start()  # __init__메서드가 호출되면 self.start()를 사용하여 스레드를 시작합니다.

    def run(self):  # 스레드가 시작되면 실행되는 메서드인 run()을 정의 합니다.
        for item in self.generator:  # self.generator에서 생성된 데이터를 하나씩 가져옵니다.
            self.queue.put(item)  # 가져온 데이터를 큐에 넣습니다.
        self.queue.put(None)  # 모든 데이터를 큐에 넣은 후에 None을 큐에 추가하여 데이터 생성이 끝났음을 알린다.

    def next(self):  # 다음 데이터를 가져오는 메서드인 next()를 정의합니다.
        next_item = self.queue.get()  # 큐에서 다음 항목을 가져옵니다.
        if next_item is None:
            raise StopIteration  # 예외를 발생시킵니다. 이는 데이터가 모두 소진되었음을 나타내는 예외
        return next_item


def train(args):
    train_loader = Generator(
        DataGenerator())  # 학습 데이터를 생성하는 DataGenerator()를 이용하여 데이터 로더 를 생성 (Generator 클래스를 사용하여 스레드를 통해 데이터를 생성하고 메인 스레드와 통신하도록 구현한 것으로 보인다)

    model = Tacotron(K=16,
                     conv_dim=[128, 128]).cuda()  # K=16과 conv_dim=[128,128]를 인자로 전달하여 Tacotron 모델을 생성하고 모델을 GPU에 올린다.

    optimizer = optim.Adam(model.parameters())

    step, epochs = 0, 0
    if args.checkpoint is not None:  # 만약 이전에 저장한 체크포인트 파일이 존재한다면
        ckpt = torch.load(args.checkpoint)  # 체크포인트 파일로부터 저장된 모델과 옵티마이저의 상태를 로드합니다.
        model.load_state_dict(ckpt['model'])  # 모델의 상태를 로드한 값으로 업데이트 합니다.
        optimizer.load_state_dict(ckpt['optimizer'])  # 옵티마이저의 상태를 로드한 값으로 업데이트합니다.
        step = ckpt['step'],  # 로드한 체크포인트에 학습 스탭을 가져옵니다.
        step = step[0]  # 로드한 체크포인트에서 에폭을 가져옵니다.
        epoch = ckpt['epoch']  # 로드한 체크포인트에서 에폭을 가져옵니다.
        print('Load Status: Epoch %d, Step %d' % (epoch, step))

    torch.backends.cudnn.benchmark = True  # CUDA 연산 최적화를 위해 CuDNN라이브러리를 사용하도록 설정합니다.

    try:
        for epoch in range(300):  # 무한 루프를 이용하여 에폭을 증가 시키며 학습을 진행합니다.itertools.count(epochs)
            for _ in range(batch_group):  # batch_group만큼 반복하면서 배치 단위로 학습을 진행합니다.
                start = time.time()
                text, dec, target, text_length = train_loader.next()  # 데이터 로더에서 학습에 필요한 text,dec,target,text_length를 가져옵니다
                text = text.cuda()
                dec = dec.float().cuda()
                target = target.float().cuda()

                pred, alignment = model(text, text_length, dec, is_training=True,
                                        mode='train')  # 모델을 사용하여 예측값pred와 어텐션 alignment을 얻습니다.
                loss = L1Loss()(pred, target)  # 에측값 pred와 실제값 target 사이의 L1 손실을 계산합니다.

                model.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1
                print(
                    'epoch: {} ,step: {}, loss: {:.5f}, {:.3f} sec/step'.format(epoch, step, loss, time.time() - start))

                if step % checkpoint_step == 0:  # 일정 주기마다 체크포인트를 저장하기 위해 checkpoint_step으로 나눈 나머지를 검사합니다.
                    save_dir ="ckpt/" + args.name + '/1'

                    input_seq = sequence_to_text(text[0].cpu().numpy())  # 텍스트 시퀀스를 생성합니다.
                    input_seq = input_seq[:text_length[0].cpu().numpy()]  # 텍스트 시퀀스를 실제 길이만큼 잘라냅니다.
                    alignment_dir = os.path.join(save_dir,
                                                 'step-{}-align.png'.format(step))  # 어텐션 그래프를 저장할 파일 경로를 생성합니다.
                    plot_alignment(alignment[0].detach().cpu().numpy(), alignment_dir,
                                   input_seq)  # 어텐션 그래프를 저장할 파일 경로를 생성합니다.
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'epoch': epoch
                    }, os.path.join(save_dir,
                                    'ckpt-{}.pt'.format(step)))  # 체크포인트 파일을 저장합니다. 모델 상태, 옵티마이저상태, 학습 스텝,에폭 정보가 저장됩니다.

    except Exception as e:
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 명령행 인자를 파싱하기 위한 argparse객체를 생성
    parser.add_argument('--checkpoint', '-c',
                        default=None)  # checkpoint또는 -c 옵션으로 체크포인트 파일 경로를 지정할 수 있도록 설정하고, 기본값은 None으로 설정합니다.
    parser.add_argument('--name', '-n', required=True)  # 모델의 이름을 지정할 수 있도록 설정하고 반드시 입력해야 합니다.
    args = parser.parse_args()  # 명령행 인자를 파싱하여 args 객체에 저장합니다.
    save_dir = os.path.join('ckpt/' + args.name, '1')  # 체크포인트 파일을 저장할 디렉토리 경로를 생성합니다.
    os.makedirs(save_dir, exist_ok=True)
    train(args)  # train 함수를 호출하여 Tacotron2모델을 학습합니다.
