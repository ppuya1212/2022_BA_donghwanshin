# Python Tutorial by 2021010668 신동환

## 1. Dimensionality Reduction
<details>
    <summary> View Contents </summary>

  * 차원축소의 목적: 불필요한 데이터를 제거하여 복잡도를 낮추면서 성능을 유지, computation power 절감

  * 차원축소 기법의 종류
  ![image](https://user-images.githubusercontent.com/77199749/195138151-b06862d6-5887-42c5-b660-7b9d6816f127.png)

  * Genetic Algorithm
  ![image](https://user-images.githubusercontent.com/77199749/195138246-bb82f29a-2246-4db7-a469-3672852b8b72.png)

  * PCA vs. MDS
  ![image](https://user-images.githubusercontent.com/77199749/195138319-51de7065-8de3-4616-9dbb-e8a9b7548cc9.png)

</details>

## 2. SVM&SVR
<details>
    <summary> View Contents </summary>
    
  * Support Vector Machine(SVM): 마진을 극대화하는 초평면활용 분류기
  ![image](https://user-images.githubusercontent.com/77199749/195419556-907a0536-c7e5-4352-aecd-09477dc94630.png)

  * SVM 4 Cases
  ![image](https://user-images.githubusercontent.com/77199749/195419669-c1fcbe7a-acff-405e-923c-e78a1c3cc051.png)

  * SVM Case1: Linear&Hard Margin
  ![image](https://user-images.githubusercontent.com/77199749/195419851-e0627d18-2c57-452e-9665-a03dd14653a9.png)

  * Support Vector Regression: "𝝐-tube"를 가진 회귀 추정식
  ![image](https://user-images.githubusercontent.com/77199749/195420076-533c14b2-9dbb-4476-a847-50e36ca8ff15.png)

</details>


## 3. Anomaly Detection
<details>
    <summary> View Contents </summary>
    
  * Anomaly Detection: "이상치" 탐지 기법
  
  * Novel Data: 이상치 데이터이지만, 긍정적인 의미로 사용됨. 데이터의 본질적 특성은 같지만, 유형이 다른 관측치(돌연변이 등)
  * Anomaly Data: 이상치 데이터이지만, 부정적인 의미로 사용됨.  대부분의 데이터와 특성이 다른 관측치(불량 등)
  * Outlier Data: 이상치 데이터, 대부분의 데이터와 본질적인 특성이 다른 관측치(미완성제품 등)

  * Anomaly Detection vs Classification
    : 아래 그림처럼, 이상치 탐지는 데이터의 불균형이 심하고 minority class의 개수가 정량적으로 매우 적을 때 사용한다.
  ![image](https://user-images.githubusercontent.com/77199749/201830267-fb474e19-cad8-43e5-baef-ef2061d4bcc4.png)

  * Density-based Anomaly Detection: 주어진 데이터를 바탕으로 각 객체들이 생성될 확률을 추정하여 새로운 데이터가 생성될 확률이 낮을 경우 이상치로 판단함.

  * 아래의 4가지 밀도기반 이상치 탐지기법들을 소개함
      1. Single Gassusian -> parametric기법으로,

          * 모든 데이터가 '하나'의 가우시안(정규)분포로부터 생성됨을 가정함.
          * 학습은 정상데이터들의 가우시안분포의 평균(mu)과 공분산(var)행렬을 추정하며 이루어짐
          * 테스트는 새로운 데이터가 생성확률이 낮으면 이상치, 높으면 정상으로 구분함

          ![image](https://user-images.githubusercontent.com/77199749/201831498-e5f60dac-f6dd-48ca-8dfc-a87052a7b745.png)
          
          * 결과해석: Gaussian 기법은 추정이 간단하며 학습기간이 짧고, 적절한 threshold(epsilon)를 분포로부터정할 수 있다. 위 그림에서 보이는 것처럼 epsilon값을 줄일 수록 이상치의 개수(빨간색)가 증가하고, epsilon값을 늘릴 수록 이상치의 개수(빨간색)가 줄어드는 것을 알 수 있었다.


      2. Gaussian Mixture -> parametric기법

          * 여러가지의 가우시안(정규)분포로부터 데이터가 생성됬음을 가정함
          * Expectation-Maximization(EM)알고리즘을 통해, 각 가우시안분포의 중심점(조건확률)이 끝날때까지 반복함

          ![image](https://user-images.githubusercontent.com/77199749/201831111-2db4a24c-176a-4217-a45d-0235391c3cad.png)
          
          * 결과해석: Mixture of Gaussian 기법은 Single Gaussian기법을 보완하여, multi Gaussian분포를 추정해, 좀 더 유연하고 복잡한 boundary를 만들 수 있게 된다. 위 그림처럼, 5개의 클러스터가 있을 때에도 다수군집의 확률분포를 구분할 수 있도록 만든다. 

      3. Parzen Window Density Estimation -> non-parametric기법
          * 위 가우시안기법들은 데이터가 특정분포를 따른다고 가정하지만, 파젠 윈도우 기법에선 데이터가 특정분포를 따른다고 가정하지 않음 ==> non-parametric기법
          * 확률밀도를 추정할 때, 데이터의 수는 고정되어 있으므로 적절한 V(부피)를 찾는 문제로 귀결되지만, 파젠 윈도우 기법에선 V(부피)를 고정하고, "k(V안에 포함되는 데이터 수)"를 찾음 ==> 커널 밀도 추정

          ![image](https://user-images.githubusercontent.com/77199749/201831516-db5a7a33-4329-45b0-a5c7-875c601b031d.png)

          * 결과해석: 커널을 활용한 파젠윈도우 기법은 parametric한 가우시안기법들과 다르게, 특정분포를 따른다고 가정하지 않아 더욱 유연하고 섬세한 boundary를 생성 할 수 있다. 특별히, smoothing (window width) parameter h를 사용해 밀도 분포를 조정할 수 있다.


      4. Local Outlier Factor(LoF) -> non-parametric기법
          * 이상치 스코어를 산출 할때, "주변부"의 밀도도 같이 고려하여 이상여부를 나타내고자함
          * 특정 데이터 주변 밀도가 작지만, 주변부 데이터들은 밀도가 높을때 --> 이상치라고 판단함

          ![image](https://user-images.githubusercontent.com/77199749/201831541-1e7cd125-89aa-4297-b7d6-4c9a65e60f43.png)
          
          * 결과해석: LoF는 위 설명드린 3가지 기법과 다르게 특정 데이터의 '주변부 밀도'를 고려한다는점에서 큰 차이점이 있다. 위 그림에서 볼 수 있듯이, 가장자리에 위치하여 주변부밀도가 작은 데이터포인트들을 이상치라고 판단하는것을 알 수 있다. 특정 이상치를 정제하는데에 있어선 좋은 detection방안이 될 수 있을것이다.


  * Model-based Anomaly Detection: 주어진 정상데이터만으로 각 모델들을 학습하여, 새로운 데이터가 들어왔을 때 각 모델의 기준에 부합하지 않는 데이터들을 이상치라 판단함.
  
  * 아래의 3가지 모델기반 이상치 탐지기법들을 소개함
    1. Auto-Encoder
        * 이미지 데이터(예시)를 넣었을 때, 똑같은 이미지를 복원해내는 NN모델
        * 이 때, 정상치만을 학습시켜 weight값을 저장하고 새로운 이상치가 들어왔을때 복원error값이 높아지므로, 복원이 잘 안될 수록 이상치로 판단
        
        ![image](https://user-images.githubusercontent.com/77199749/201831412-ec22f679-70cc-4d8b-8317-4211a7a14235.png)
        
        * 결과해석: Auto-encoder는 인코더단에서 압축한 latent vector를 decoder단에서 생성해내는 모델이다. 그림'outlier-score'그림을 통해 극히 outlier-score가 높은 데이터들을 통해 이상치를 탐지할 수 있었다. 또한 'Combination by average'그림을 통해 3개의 모델을 동시에 사용하고 이를 정규화함으로써 조금 더 정제된 Anomaly_score를 얻을 수 있다.
    
    
    2. One Class Support Vector Machine(OCSVM)
        * Threshold(임계치)가 아닌 "boundary"를 사용하여 이상치 여부를 판단함
        * OCSVM은 원점으로부터의 거리들을 사용하여 "초평면 boundary"를 만들고, 이를 기준으로 이상치 여부를 판단
        (참고)
        * SVDD과 OCSVM의 공통점은, 모두 threshold(임계치)가 아닌 boundary를 생성하여 이상치여부를 구분하는 것임
        * 차이점은, "boundary의 형태와 중심점"이 다름
        * OCSVM에선 초평면을 활용하였다면, SVDD에선 "초구 boundary"를 사용하며 "원점이 중심이 아니어도 무방"함

        ![image](https://user-images.githubusercontent.com/77199749/201831461-7e44eced-05fe-4b6f-b4de-3f99e07a17b7.png)
        
        * 결과해석: 일반적인 선형분류기인 SVM과 다르게 비선형성을 더한 OCSVM으로 원점으로부터 떨어진 거리로 초평면을 만들고 데이터들을 정상/이상치로 분류하는 성능이 그림에서처럼 눈에 띠게좋았다. 여러 데이터들 속에 포함되어 있는 데이터가 이상치일땐, 분류하기 어렵지만 위 그림처럼 경계면에서 발생하는 이상치는 threshold를 정하여 분류됨을 볼 수 있다.

    3. Isolation Forest
        * Forest라는 단어에서 알 수 있듯이, 분기를 하면서 이상치여부를 따지는 tree구조의 모델임
        * 이 때, 소수 범주(이상치)는 개체수가 적을 것이므로 적은 분기만으로 고립이 가능하다는 가정
        * 따라서, 고립시키는데에 많은 분기가 필요하면 정상, 적은 분기가 필요하다면 이상치데이터로 판단함

        ![image](https://user-images.githubusercontent.com/77199749/201831471-541147c9-d165-422a-bfd3-cdffad30a401.png)

        * 결과해석: Isolation Forest는 Forest의 장점을 이상치 탐지에 잘 녹여내었다고 볼 수 있다. 위 그림에서 처럼, 분기 수가 많이 필요없는 이상치 데이터들은 전부 이상치로 잘 분류되어지고, 분기가 많이 필요한, 즉 정상 데이터들은 정상으로 분류됨을 볼 수 있다. 여러개의 밀집된 군집이 형성되어 있을 때 사용하면 효과적인 이상치 탐지 성능을 볼 수 있을 것이다.

  
</details>

## 4. Ensemble Learning
<details>
    <summary> View Contents </summary>
    
  * Ensemble: "이상치" 탐지 기법
  
  * 


  * Bagging : asdf

  * 아래 Bagging기반 앙상블 기법을 소개함
      1. Random Forest -> parametric기법으로,

          * 모든 데이터가 '하나'의 가우시안(정규)분포로부터 생성됨을 가정함.
          * 학습은 정상데이터들의 가우시안분포의 평균(mu)과 공분산(var)행렬을 추정하며 이루어짐
          * 테스트는 새로운 데이터가 생성확률이 낮으면 이상치, 높으면 정상으로 구분함

          ![image](https://user-images.githubusercontent.com/77199749/201831498-e5f60dac-f6dd-48ca-8dfc-a87052a7b745.png)
          
          * 결과해석: Gaussian 기법은 추정이 간단하며 학습기간이 짧고, 적절한 threshold(epsilon)를 분포로부터정할 수 있다. 위 그림에서 보이는 것처럼 epsilon값을 줄일 수록 이상치의 개수(빨간색)가 증가하고, epsilon값을 늘릴 수록 이상치의 개수(빨간색)가 줄어드는 것을 알 수 있었다.


  * Boosting: asdf
  
  * 아래의 4가지 Boosting기반 앙상블 모델들을 소개함
    1. Auto-Encoder
        * 이미지 데이터(예시)를 넣었을 때, 똑같은 이미지를 복원해내는 NN모델
        * 이 때, 정상치만을 학습시켜 weight값을 저장하고 새로운 이상치가 들어왔을때 복원error값이 높아지므로, 복원이 잘 안될 수록 이상치로 판단
        
        ![image](https://user-images.githubusercontent.com/77199749/201831412-ec22f679-70cc-4d8b-8317-4211a7a14235.png)
        
        * 결과해석: Auto-encoder는 인코더단에서 압축한 latent vector를 decoder단에서 생성해내는 모델이다. 그림'outlier-score'그림을 통해 극히 outlier-score가 높은 데이터들을 통해 이상치를 탐지할 수 있었다. 또한 'Combination by average'그림을 통해 3개의 모델을 동시에 사용하고 이를 정규화함으로써 조금 더 정제된 Anomaly_score를 얻을 수 있다.
    
    
    2. One Class Support Vector Machine(OCSVM)
        * Threshold(임계치)가 아닌 "boundary"를 사용하여 이상치 여부를 판단함
        * OCSVM은 원점으로부터의 거리들을 사용하여 "초평면 boundary"를 만들고, 이를 기준으로 이상치 여부를 판단
        (참고)
        * SVDD과 OCSVM의 공통점은, 모두 threshold(임계치)가 아닌 boundary를 생성하여 이상치여부를 구분하는 것임
        * 차이점은, "boundary의 형태와 중심점"이 다름
        * OCSVM에선 초평면을 활용하였다면, SVDD에선 "초구 boundary"를 사용하며 "원점이 중심이 아니어도 무방"함

        ![image](https://user-images.githubusercontent.com/77199749/201831461-7e44eced-05fe-4b6f-b4de-3f99e07a17b7.png)
        
        * 결과해석: 일반적인 선형분류기인 SVM과 다르게 비선형성을 더한 OCSVM으로 원점으로부터 떨어진 거리로 초평면을 만들고 데이터들을 정상/이상치로 분류하는 성능이 그림에서처럼 눈에 띠게좋았다. 여러 데이터들 속에 포함되어 있는 데이터가 이상치일땐, 분류하기 어렵지만 위 그림처럼 경계면에서 발생하는 이상치는 threshold를 정하여 분류됨을 볼 수 있다.

    3. Isolation Forest
        * Forest라는 단어에서 알 수 있듯이, 분기를 하면서 이상치여부를 따지는 tree구조의 모델임
        * 이 때, 소수 범주(이상치)는 개체수가 적을 것이므로 적은 분기만으로 고립이 가능하다는 가정
        * 따라서, 고립시키는데에 많은 분기가 필요하면 정상, 적은 분기가 필요하다면 이상치데이터로 판단함

        ![image](https://user-images.githubusercontent.com/77199749/201831471-541147c9-d165-422a-bfd3-cdffad30a401.png)

        * 결과해석: Isolation Forest는 Forest의 장점을 이상치 탐지에 잘 녹여내었다고 볼 수 있다. 위 그림에서 처럼, 분기 수가 많이 필요없는 이상치 데이터들은 전부 이상치로 잘 분류되어지고, 분기가 많이 필요한, 즉 정상 데이터들은 정상으로 분류됨을 볼 수 있다. 여러개의 밀집된 군집이 형성되어 있을 때 사용하면 효과적인 이상치 탐지 성능을 볼 수 있을 것이다.
    
    4. Isolation Forest
        * Forest라는 단어에서 알 수 있듯이, 분기를 하면서 이상치여부를 따지는 tree구조의 모델임
        * 이 때, 소수 범주(이상치)는 개체수가 적을 것이므로 적은 분기만으로 고립이 가능하다는 가정
        * 따라서, 고립시키는데에 많은 분기가 필요하면 정상, 적은 분기가 필요하다면 이상치데이터로 판단함

        ![image](https://user-images.githubusercontent.com/77199749/201831471-541147c9-d165-422a-bfd3-cdffad30a401.png)

        * 결과해석: Isolation Forest는 Forest의 장점을 이상치 탐지에 잘 녹여내었다고 볼 수 있다. 위 그림에서 처럼, 분기 수가 많이 필요없는 이상치 데이터들은 전부 이상치로 잘 분류되어지고, 분기가 많이 필요한, 즉 정상 데이터들은 정상으로 분류됨을 볼 수 있다. 여러개의 밀집된 군집이 형성되어 있을 때 사용하면 효과적인 이상치 탐지 성능을 볼 수 있을 것이다.

    5. Isolation Forest
        * Forest라는 단어에서 알 수 있듯이, 분기를 하면서 이상치여부를 따지는 tree구조의 모델임
        * 이 때, 소수 범주(이상치)는 개체수가 적을 것이므로 적은 분기만으로 고립이 가능하다는 가정
        * 따라서, 고립시키는데에 많은 분기가 필요하면 정상, 적은 분기가 필요하다면 이상치데이터로 판단함

        ![image](https://user-images.githubusercontent.com/77199749/201831471-541147c9-d165-422a-bfd3-cdffad30a401.png)

        * 결과해석: Isolation Forest는 Forest의 장점을 이상치 탐지에 잘 녹여내었다고 볼 수 있다. 위 그림에서 처럼, 분기 수가 많이 필요없는 이상치 데이터들은 전부 이상치로 잘 분류되어지고, 분기가 많이 필요한, 즉 정상 데이터들은 정상으로 분류됨을 볼 수 있다. 여러개의 밀집된 군집이 형성되어 있을 때 사용하면 효과적인 이상치 탐지 성능을 볼 수 있을 것이다.

  
</details>
    
==========================================================================
## Reference
- https://sustaining-starflower-aff.notion.site/2022-2-0e068bff3023401fa9fa13e96c0269d7 (강필성 교수님 ba수업자료)
