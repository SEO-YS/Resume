import pandas as pd
import numpy as np
import os, librosa, re, glob, scipy
from tqdm import tqdm
from util.hparams import *
from util.text import text_to_sequence
from matplotlib import rcParams

path2 ="/mnt/srv/home/test1_kh/3_YS_Tacotron2/"
# path1 = "/mnt/srv/home/test1_kh/2_YS_Hifi_GAN/Guide_Voice/"

# text_dir = path2+"Gudie/txt/gudie.txt"
text_dir = path2+"Child_Voice/text_clear/child.txt"
filters = '([.,!?])'

metadata = pd.read_csv(text_dir, dtype='object', sep='|', header=None)
# print(metadata.head())
wav_dir = metadata[0].values
print(wav_dir)
text = metadata[1].values
# print(text)

out_dir = path2+'Child_Voice/text_data'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_dir + '/text', exist_ok=True)
os.makedirs(out_dir + '/mel', exist_ok=True)
os.makedirs(out_dir + '/dec', exist_ok=True)
os.makedirs(out_dir + '/spec', exist_ok=True)

text
print('Load Text')
text_len = []
for idx, s in enumerate(tqdm(text)):
    sentence = re.sub(re.compile(filters), '', s) #정규 표현식 패턴을 사용하여 "filters"에 해당하는 문자열을 ""(빈 문장열)로 대체합니다. 이 과정은 텍스트에서 불필요한 문자를 제거하는 역할을 합니다.
    sentence = text_to_sequence(sentence) #"text_to_sequence" 함수를 사용하여 sentence  문자열을 처리합니다.
    text_len.append(len(sentence)) #현재 문장의 길이를 text_len 리스트에 추가합니다. 이렇게 하면 각 문장의 길이 정보가 저장됩니다.
    text_name = 'guide-text-%05d.npy' % idx
    np.save(os.path.join(out_dir + '/text', text_name), sentence, allow_pickle=False) #sentence를 NumPy배열로 저장하고, 해당 배열을 저장, "allow_pickle=False"옵션은 배열을 저장할 때 Pickle 형식을 사용하지 않도록 설정
np.save(os.path.join(out_dir + '/text_len.npy'), np.array(text_len))#모든 문장의 길이 정보를 NumPy 배열로 저장하고 파일에 저장
print('Text Done')

# audio
print('Load Audio')
mel_len_list = []
for idx, fn in enumerate(tqdm(wav_dir)):
    file_dir = path2+'Child_Voice/audio_clear/'+fn+".wav"
    wav, _ = librosa.load(file_dir, sr=sample_rate) #"librosa.load()" 함수를 사용하여 오디오 파일을 로드
    wav, _ = librosa.effects.trim(wav) #오디오에서 묵음 부분을 자르기 위한 "librosa.effects.trim 함수를 사용
    wav = scipy.signal.lfilter([1, -preemphasis], [1], wav) #전처리를 위해 scipy.signal.lfilter()함수를 사용하여 오디오 데이터에 프리에멘시스(preemphasis)필터를 적용
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length) #오디오의 짧은 시간 간격 별로 고속 푸리에 변환(Short-TimeFourier Transform)을 수행합니다.
    stft = np.abs(stft) # 절대값으로 변환하여 복소수 형태의 데이터를 강도로 변환
    mel_filter = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=mel_dim) #멜 스펙트로그램 생성을 위해 멜 필터를 생성
    mel_spec = np.dot(mel_filter, stft) #멜 필터를 STFT결과에 적용하여 멜스펙트로그램을 얻습니다.

    mel_spec = 20 * np.log10(np.maximum(1e-5, mel_spec)) #멜 스펙트로그램을 로그 스케일로 변환
    stft = 20 * np.log10(np.maximum(1e-5, stft)) #STFT결과도 로그 스케일로 변환

    mel_spec = np.clip((mel_spec - ref_db + max_db) / max_db, 1e-8, 1) #멜 스펙트로그램을 정규화하여 특정 범위 내로 클리핑합니다.
    stft = np.clip((stft - ref_db + max_db) / max_db, 1e-8, 1) # STFT 결과도 정규화하여 특정 범위 내로 클리핑합니다.

    mel_spec = mel_spec.T.astype(np.float32) # 멜 스펙트로그램을 전치사(transpose)하여 형태를 변경하고, 데이터 타입을 32비트 부동소수점형으로 변환합니다.
    stft = stft.T.astype(np.float32) # STFT 결과도 똑같은 행동을 한다.
    mel_len_list.append([mel_spec.shape[0], idx]) # 멜 스펙트로그램의 시간 축 길이와 인덱스 정보를 'mel_len_list' 에 추가합니다. 이렇게 하면 각 오디오 파일의 멜 스펙트로그램 정보와 길이가 저장됩니다.

    # padding
    remainder = mel_spec.shape[0] % reduction # 'mel_spec'의 시간 축 길이를 reduction으로 나눈 나머지를 remainder변수에 저장합니다.reduction은 멜 스펙트로그램의 시간 축을 줄이기 위해 사용되는 정수 값입니다.
    if remainder != 0:
        mel_spec = np.pad(mel_spec, [[0, reduction - remainder], [0, 0]], mode='constant')  #mel_spec을 reduction - remainder 만큼 상하로 패딩합니다. 이렇게 하면 멜 스펙트로그램의 시간 축 길이를 reduction의 배수로 맞출수 입습니다.
        stft = np.pad(stft, [[0, reduction - remainder], [0, 0]], mode='constant') # STFT 결과인 stft도 같은 방식으로 패딩하여 시간 축 길이를 조정합니다.

    mel_name = 'child-mel-%05d.npy' % idx
    np.save(os.path.join(out_dir + '/mel', mel_name), mel_spec, allow_pickle=False)

    stft_name = 'child-spec-%05d.npy' % idx
    np.save(os.path.join(out_dir + '/spec', stft_name), stft, allow_pickle=False)

    # Decoder Input
    mel_spec = mel_spec.reshape((-1, mel_dim * reduction)) #맬 스펙트로그램 mel_spec의 형태를 변경하여 시간 축을 줄입니다. 새로운 형태는 2차원 배열로, 행은 mel_spec * reduction길이를 가지도록 하고 열은 자유롭게 조정됩니다.
    dec_input = np.concatenate((np.zeros_like(mel_spec[:1, :]), mel_spec[:-1, :]), axis=0) # 디코더 입력인 dec_input을 생성합니다.dec_input은 멜 스펙트로그램 mel_spec의 이전 타임스텝과 현재 타임스텝을 결합하여 만들어 집니다.첫 번째 행은 0으로 채워지며, 나머지 행은 이전 타임스텝의 값으로 채워집니다.
    dec_input = dec_input[:, -mel_dim:] #dec_input 의 열은 mel_dim 길이로 자릅니다. 이로써 디코더 입력의 길이가 멜 스펙트로그램의 차원과 일치하게 됩니다.
    dec_name = 'child-dec-%05d.npy' % idx
    np.save(os.path.join(out_dir + '/dec', dec_name), dec_input, allow_pickle=False)

mel_len = sorted(mel_len_list) # mel_len_list를 오름차순으로 정렬한다
np.save(os.path.join(out_dir + '/mel_len.npy'), np.array(mel_len)) # 정렬된 mel_len 을 Numpy 배열로 저장한다.
print('Audio Done')
