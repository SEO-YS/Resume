import os, argparse, glob, torch, scipy, librosa
import numpy as np
from jamo import hangul_to_jamo
import soundfile as sf
from tacotron_models.tacotron import Tacotron
from util.text import text_to_sequence, sequence_to_text
from util.plot_alignment import plot_alignment
from util.hparams import *
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from transformer_model.transformer_model import *
from tacotron_models.tacotron import post_CBHG
from tacotron_models.modules import griffin_lim
from util.plot_alignment import plot_alignment


transformer_model = torch.load("/transformer_model/transformer_jeju_model3.pt")
mel_load_dir = "/translator/result"

os.makedirs(mel_load_dir, exist_ok=True)

title = input('제목을 입력해주세요 = ')
texts = input('문장을 입력해주세요 = ')
translation, attention = translate_sentence(texts, SRC, TRG, transformer_model, device, logging=True)
text_list = []
for text in translation:
    if text in ["<unk>", "?", "!"]:
        pass
    else:
        text_list.append(text)
texts = " ".join(text_list)
sentences = [texts]
if "?" in translation:
    print("제주도 번역 = ", texts + "?")
    texts = texts + "?"
elif "!" in translation:
    print("제주도 번역 = ", texts + "!")
    texts = texts + "!"
else:
    print("제주도 번역 = ", texts)


def inference(text, idx):
    seq = text_to_sequence(text)  # text는 생성하고자 하는 음성에 해당하는 텍스트이며, 'idx'는 인덱스로 결과 파일의 이름을 지정할 때 사용됩니다.
    enc_input = torch.tensor(seq, dtype=torch.int64).unsqueeze(
        0)  # 입력 시퀀스 seq를 정수형으로 변환하여 enc_input에 저장하고, 배치 차원을 추가합니다.
    enc_input = enc_input
    sequence_length = torch.tensor([len(seq)],
                                   dtype=torch.int32)  # 시퀀스의 길이를 구하여 sequenece_length에 저장합니다. 이 값은 모델에서 인코더에 입력할 때 사용됩니다.
    dec_input = torch.from_numpy(
        np.zeros((1, mel_dim), dtype=np.float32))  # 디코더 입력을 빈 Mel스펙트로그램으로 초기화합니다.디코더는 이 입력을 기반으로 음성을 생성합니다.

    # pred, alignment = model(enc_input, sequence_length, dec_input, is_training=False,
    #                         mode='inference')  # 모델을 사용하여 입력으로부터 Mel 스펙트로그램 pred와 정렬 정보 alignment을 추론합니다
    pred, alignment = tacotron_model(enc_input, sequence_length, dec_input, is_training=False,
                            mode='inference')
    pred = pred.squeeze().detach().numpy()  # 추론된 Mel스펙트로그램 pred에서 배치 차원을 제거하고, 그래디언트 추적을 비활성화하여 Numpy 배열로 변환합니다.
    alignment = np.squeeze(alignment.detach().numpy(),
                           axis=0)  # 추론된 정렬 정보 alignment 에서 배치 차원을 제거합니다. 정렬 정보는 시각화를 위해 사용 됩니다.

    np.save(os.path.join(mel_load_dir, 'mel-{}'.format(title)), pred, allow_pickle=False)

    mel_list = glob.glob(os.path.join(mel_load_dir, '*.npy'))

    mel_title_name = fr"\translator\result\mel-{title}.npy"

    for i, mel_data in enumerate(mel_list):  # mel_list에 있는 Mel스펙트로그램 파일들에 대해 반복합니다.
        mel_data = os.path.normpath(mel_data)
        if mel_data == mel_title_name:
            mel = np.load(mel_data)
            mel_inference(mel, i)

    input_seq = sequence_to_text(seq)  # 시퀀스를 다시 텍스트로 변환하여 input_seq에 저장합니다. 이는 시각화를 위해 사용됩니다.
    alignment_dir = os.path.join(mel_load_dir, 'align-{}.png'.format(title))  # 정렬 정보를 시각화한 이미지 파일을 저장할 경로를 생성합니다.
    plot_alignment(alignment, alignment_dir, input_seq)  # input_seq 를 사용하여 정렬 정보를 시각화하고, 그 결과를 이미지 파일로 저장합니다.


def mel_inference(text, idx):  # text는 생성하고자 하는 음성에 해당하는 Mel스펙트로그램입니다.
    idx = 0
    mel = torch.from_numpy(text).unsqueeze(0)  # 주어진 Mel스펙트로그램 데이터를 Numpy 배열에서 Pytorch 텐서로 변환합니다. 배치 차원을 추가합니다.
    # mel = torch.from_numpy(text)
    pred = post_CBHG_model(mel)  # 모델에 Mel 스펙트로그램 데이터를 입력하여 음성 파리미터를 예측합니다.
    pred = pred.squeeze().detach().numpy()  # 예측된 파라미터에서 배치 차원을 제거하고, 그래디언트 추적을 비활화성화 하여 NumPy배열로 변환합니다.
    pred = np.transpose(pred)  # 파라미터 데이터의 축 순서를 변경하여 다시 시간 순서로 변환합니다.

    pred = (np.clip(pred, 0, 1) * max_db) - max_db + ref_db  # 파리미터를 원래 스케일로 변환합니다.
    pred = np.power(10.0, pred * 0.05)  # 파리미터를 원래의 스펙트로그램 형태로 변환합니다.
    wav = griffin_lim(pred ** 1.5)  # 그리핀-림 알고리즘을 사용하여 파라미터에서 음성 데이터를 생성합니다.
    wav = scipy.signal.lfilter([1], [1, -preemphasis], wav)  # 선형 필터를 적용하여 미리 강조된 프리-엠페시스를 취소합니다.
    wav = librosa.effects.trim(wav, top_db=60,frame_length=2048, hop_length=512)[0]  # 스펙트로그램에서 생성된 음성의 무음 부분을 자릅니다.
    wav = wav.astype(np.float32)  # 음성 데이터를 32비트 부동 소수점 형태로 변환합니다.
    sf.write(os.path.join(mel_load_dir, '{}.wav'.format(title)), wav, sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint1', '-c1', default=None)  # 체크포인트 파일 경로를 명령줄 인자로 받을 수 있도록 설정합니다.
    parser.add_argument('--checkpoint2', '-c2', default=None)  # 체크포인트 경로를 명령줄 인자로 받을 수 있도록 설정합니다.
    args = parser.parse_args()


    tacotron_model = Tacotron(K=16, conv_dim=[128, 128])  # Tacotron 모델을 생성합니다.
    tacotron_ckpt = torch.load(args.checkpoint1)  # 지정된 체크포인트 파일에서 모델의 가중치를 로드합니다.
    tacotron_model.load_state_dict(tacotron_ckpt['model'])

    post_CBHG_model = post_CBHG(K=8, conv_dim=[256, mel_dim])
    post_ckpt = torch.load(args.checkpoint2)  # 지정된 체크포인트 파일에서 모델의 가중치를 로드합니다.
    post_CBHG_model.load_state_dict(post_ckpt['model'])

    for i, text in enumerate(sentences):
        jamo = ''.join(list(hangul_to_jamo(text)))
        inference(jamo, i)