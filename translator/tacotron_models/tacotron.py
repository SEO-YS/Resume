import numpy as np
from torch.nn import Module, Embedding, Linear, GRUCell
from tacotron_models.modules import *
from util.hparams import *


class Encoder(Module):
    def __init__(self, K, conv_dim):
        super(Encoder, self).__init__()
        self.embedding = Embedding(symbol_length, embedding_dim)
        self.prenet = prenet(embedding_dim) # 임베딩 차원의 데이터를 받아서 음성 합성에 도움이 되는 정보를 추출하는 역할을 합니다.
        self.cbhg = CBHG(K, conv_dim) #CBHG 모듈은 Convolutional Bank, Highway Network, Bidirectional GRU를 통합하여 특성을 추출하는 역할을 수행합니다.
        
    def forward(self, enc_input, sequence_length, is_training):
        x = self.embedding(enc_input)
        x = self.prenet(x, is_training=is_training)
        x = x.transpose(1, 2)
        x = self.cbhg(x, sequence_length)
        return x

    
class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.prenet = prenet(mel_dim) #음성 스펙트로그램 입력을 전처리하는 역할
        self.attention_rnn = GRUCell(encoder_dim, decoder_dim) # 어텐션 기능을 수행하는 RNN
        self.attention = LuongAttention()
        self.proj1 = Linear(decoder_dim * 2, decoder_dim)
        self.dec_rnn1 = GRUCell(decoder_dim, decoder_dim)
        self.dec_rnn2 = GRUCell(decoder_dim, decoder_dim)
        self.proj2 = Linear(decoder_dim, mel_dim * reduction) # RNN 출력을 음성 스펙트로그램 차원으로 변환하는 데 사용
       
        
        
    def forward(self, batch, dec_input, enc_output, mode,sequence_length):
        if mode == 'train':
            # print('하기전 dec_input===',dec_input.shape)
            dec_input = dec_input.transpose(0, 1)
            # print('하고나서 dec_input===',dec_input.shape)
            attn_rnn_state = torch.zeros(batch, decoder_dim).cuda()
            dec_rnn_state1 = torch.zeros(batch, decoder_dim).cuda()
            dec_rnn_state2 = torch.zeros(batch, decoder_dim).cuda()
        else:
            # dec_input = dec_input.transpose(0, 1)
            attn_rnn_state = torch.zeros(batch, decoder_dim)
            dec_rnn_state1 = torch.zeros(batch, decoder_dim)
            dec_rnn_state2 = torch.zeros(batch, decoder_dim)
            
            
        if mode == 'train':
            iters = dec_input.shape[0]
        if mode == 'inference':
            if 0 <= sequence_length <=30:
                iters = 30
            elif 31<= sequence_length <=50:
                iters = 40
            elif 51 <= sequence_length <= 70:
                iters = 55
            elif 71 <= sequence_length <= 90:
                iters = 70
            else:
                iters = 90
       
        
        # iters = dec_input.shape[0] if mode == 'train' else dec_input.shape[1]
#         print("iters========",iters)
        
        
        for i in range(iters): # train 일 경우는 디코더 입력의 시퀀스 길이를, test 일 경우 최대 반복 횟수
            inp = dec_input[i] if mode == 'train' else dec_input
            x = self.prenet(inp, is_training=True) # inp을 전처리하는 과정 prenet은 입력 데이터의 차원을 조정하고 정보를 추출하는 역할
            attn_rnn_state = self.attention_rnn(x, attn_rnn_state) #어텐션 RNN에 x와 이전 스텝의 어텐션 상태 attn_rnn_state를 입력으로 넣어서 어텐션 RNN의 다음 상태를 계산합니다.
            attn_rnn_state = attn_rnn_state.unsqueeze(1)
            context, align = self.attention(attn_rnn_state, enc_output)

            dec_rnn_input = self.proj1(context)
            dec_rnn_input = dec_rnn_input.squeeze(1)

            dec_rnn_state1 = self.dec_rnn1(dec_rnn_input, dec_rnn_state1)
            dec_rnn_input = dec_rnn_input + dec_rnn_state1
            dec_rnn_state2 = self.dec_rnn2(dec_rnn_input, dec_rnn_state2)
            dec_rnn_output = dec_rnn_input + dec_rnn_state2

            dec_out = self.proj2(dec_rnn_output)

            dec_out = dec_out.unsqueeze(1)
            attn_rnn_state = attn_rnn_state.squeeze(1)

            if i == 0:
                mel_out = torch.reshape(dec_out, [batch, -1, mel_dim])
                alignment = align
            else:
                mel_out = torch.cat([mel_out, torch.reshape(dec_out, [batch, -1, mel_dim])], dim=1)
                alignment = torch.cat([alignment, align], dim=-1)
                
            if mode == 'inference':
                # print('mel_out =====',mel_out.shape)
                # dec_input = mel_out[:, reduction * (i+1) - 1, :]
                dec_input = mel_out[:, -1, :]
                # dec_input = mel_out.transpose(0, 1)
                # print("dec_input===",dec_input.shape) 

        return mel_out, alignment
    
    
class Tacotron(Module):
    def __init__(self, K, conv_dim):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(K, conv_dim)
        self.decoder = Decoder()
        
    def forward(self, enc_input, sequence_length, dec_input, is_training, mode):
        batch = dec_input.shape[0]
        x = self.encoder(enc_input, sequence_length, is_training)
        x = self.decoder(batch, dec_input, x, mode,sequence_length)
        return x
    

class post_CBHG(Module):
    def __init__(self, K, conv_dim):
        super(post_CBHG, self).__init__()
        self.cbhg = CBHG(K, conv_dim)
        self.fc = Linear(256, n_fft // 2 + 1)
        
    def forward(self, mel_input):
        x = self.cbhg(mel_input.transpose(1, 2), None)
        x = self.fc(x)
        return x
    