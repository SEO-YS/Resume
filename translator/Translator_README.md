# Translator

1. **제주도 사투리 데이터 다운로드**

    *[제주도](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=realm&dataSetSn=121)*

```
   Translator
     |- data_propercess
         |- text_data
             *.json(Json 파일만 저장) 
         |- audio_data
   ```

2. **데이터 전처리**

   *jeju_data_proprecess.ipynb를 통해서 데이터 전처리를 할 수있다*

3. **Transformer_model 생성**

   *transformer.ipynb를 통해서 모델을 생성할 수 있다.*
   (생성을 하는데 시간이 오래 걸린다(20만 data로 학습하는데 일주일 넘거 걸렸다))

```
   Translator
     |- transformer_model
         |- data
         |- transformer_jeju_model3.pt
         |- transformer_model.py            
   ```
*transformer_jeju_model3.pt를 Fine-tuning을 해도 됩니다.*(약 200000_data로 학습시킨 model)

4. **한국어 음성 데이터 다운로드**

    *[KSS](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)*

```
   Translator
     |- data_propercess
         |- text_data
             *.json(Json 파일만 저장) 
         |- audio_data
            |-archive
               |- kss
                  |-1
                  |-2
                  |-3
                  |-4
               |- transcript.v.1.x.txt  
   ```

5.  **Preprocess**

(Terminal에서 python voice_preprocess.py을 입력해서 학습에 필요한 파일을 생성한다.)

6. **Train**
   ```
   python train1.py -n <name>
   python train2.py -n <name>
   ```
   * train1 부터 순서대로 실행을 한다.
   * 만약 학습을 이어서 하고 싶으면 python train1.py -n <name> -c ckpt/<name>/1/ckpt-<step>.pt 식으로 입력을 하면 된다.


7. **Test**
    ```
   python test.py
   ```
   
8. **Result**

![20230904_180545.png](image%2F20230904_180545.png)
   



