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
    
  * Ensemble: 개별 알고리즘들을 적당히 잘 섞어 놓으면 어떨까? 라는 motivation에서 출발
  
  * 모델학습 및 오류
    * 모델이란, 주어진 하나의 샘플 집합으로부터 데이터 생성 함수를 추론하는 것임
    * 모델에 의한 오류는 편향(Bias)와 분산(Variance)로 구분 될 수 있음
    * 편향(Bias): 평균적으로 얼마나 “정확한“ 추정이 가능한지에 대한 측정 지표
    * 분산(Variance): 개별 추정이 얼마나 “차이”가 크게 나타나는지에 대한 측정 지표
    
  * 모델 구분
    * 분산이 낮고, 편향이 높음 --> Boosting
    * 편향이 낮고, 분산이 높음 --> Bagging
    
    ![image](https://user-images.githubusercontent.com/77199749/204252620-980730d0-54b9-4714-8cbc-3caa5f279e79.png)




  * Bagging : Bootstrap Aggregating의 약자로, 각  모델에 다른 학습데이터 셋을 이용해 결과값을 취합하는 앙상블 기법

  * 아래 Bagging기반 앙상블 기법을 소개함
      1. Random Forest

          * Random Forests란, 다수의 decision tree모델에
          * 동일하지 않은 데이터셋(크기는 동일, 중복가능)을 '독립적'으로 학습 시킨 후
          * 결과를 majority voting과 같이 취합하는 모델임
          * bagging의 대표적인 모델이라 할 수 있음
    
            ![image](https://user-images.githubusercontent.com/77199749/204252497-deeca3ea-1cbd-4974-8a6e-71cd70b96e96.png)


          
          * 결과해석: 하나의 decision tree로는 overfitting이 쉽게 일어날 수 있지만, 다수의 tree모델을 사용하여 각기 다른 데이터셋으로 학습을 시켜 일반화된 tree를 만들었음. 앙상블 모델로, 여러개의 형성된 tree모델에 새로운 데이터를 통과시키며, 각 트리가 분류한 결과에서 voting을 실시하여 가장 많이 득표한 결과를 최종 분류 결과로 선택함. 또한, Information Gain(IG)과 Gini impurity를 기준으로 split을 진행하여 비교한 결과, 분류 정확도와 error카운트수가 동일한것으로 보아, 붓꽃 데이터와 같이 단순데이터에 대해선 큰 차이가 없음을 확인하였다.


  * Boosting: Boosting은 오분류된 데이터에 집중해 더 많은 가중치를 주는 ensemble 기법으로, 특징은, 각 모델이 서로에게 dependent(model guided)한 성질을 지녀 다양성 측면에서 explicit함
  
  * 아래의 4가지 Boosting기반 앙상블 모델들을 소개함
    1. Adaptive Boosting(AdaBoost)
        * AdsBoost에선 이전단계에서의 단점을 데이터분포에 반영시키면서 모델의 성능을 향상시킴.
        * 가장 기본적인 AdabBoost의 base learner(estimator)는 depth가 1인 DecisionTree임
        * 분기횟수(depth)를 증가시키고,
        * 추정횟수(n_estimators)를 증가시키면서 비교 실험하였음
        
        ![image](https://user-images.githubusercontent.com/77199749/204252879-498089dc-f8f5-4e19-9223-7ab93108c6be.png)
        
        * 결과해석: AdaBoost를 통해 boosting의 기본적인 개념은 이전 단계에서 맞추지 못한 데이터들의 영향력을 키워 다음 단계에서 더 잘 맞추게 유도하는 것임을 알았다. Tree계열의 모델을 base model로 할때에는 주로 depth(leaf노드까지의 분기수라고 이해하면 편함)를 깊게하고 추정횟수를 더 많이 하는것이 성능이 제일 좋은 것을 확인하였다.
    
    
    2. Gradient Boosting Machine(GBM)
        * AdaBoost와 비슷한 boosting 계열의 모델이지만,
        * GBM에선 가중치 업데이트를 Gradient Descent를 이용해 회귀모형의 잔차를 예측하는 모델임
        * AdaBoost에선 전 단계의 단점이 '데이터의 선택확률'에 반영되지만,
        * GBM에선 '잔차'를 구하는 '손실함수의 gradient'에 반영됨

        ![image](https://user-images.githubusercontent.com/77199749/204253044-f7194157-9d87-4f5d-876f-61097fed4654.png)
        
        * 결과해석: AdaBoost와 동일한 조건(max_depth = 10, n_estimator=100)으로 진행하였을때보다 성능이 '9%'이상 향상된 것을 볼 수 있다. 단순히 데이터의 분포를 바꾸어가며 boosting을 하는 것 보단, 손실함수의 gradient에 반영하여 모델을 훈련시키는 것이 해당 데이터에선 보다 효과적임을 알았다. 그러나, 시간적인 측면을 보자면 AdaBoost에 비해 '8.8배 오래'걸리는 것은 충분히 고려해야할 사항이다. 추가로, GBM내에서 성능을 향상시키고자 tree의 depth를 2배로 증가시켜 20을 만들고 반복횟수를 100에서 50회로 낮추어 진행을 한 결과, 성능이 오히려 감소되었다. 특정데이터에서 tree의 depth를 높게 잡는 것이 더 정확한 분류가 가능한 것이라 예상하였으나, 과적합으로 인한 정확도 감소를 보였다.

    3. XGBoost
        * GBM은 위 실험에서 알 수 있듯이, Adaboost보다 100배의 시간이 걸리고, 과적합의 이슈가 있었음
        * 이러한 단점을 보완하기 위해 XGBoost가 탄생함
        * 전역을 탐색한 분기점이 아니라, 'locally 최적split을 찾으므로' GBM 보다 빠르고
        * 과적합을 방지하기 위해 규제를 포함시킴

        ![image](https://user-images.githubusercontent.com/77199749/204253112-8b2834b8-9fb3-4a73-bd41-3843537f5f17.png)

        * AdaBoost와 동일한 조건(max_depth = 10, n_estimator=100)으로 진행하였을때보다 성능이 '7%' 향상된 반면 GBM보단 '2% 낮은 성능'을 보인다. GBM처럼 gradient의 값을 반영시켜 잔차를 줄이는 것은 동일하나,전역을 탐색하여 분기점을 찾지 않고, 분할된 데이터셋에서 최적의 분기점을 찾는 것이 차이점이다. 이를 통해 XGBoost는 GBM에 비해 '200배 이상 빠르게' 훈련 및 테스트 된다.    

    
    4. LightGBM
        * GBM에서 출발한 알고리즘으로, 두가지의 관점을 제시한다
        * 1.Gradient-based One-slide Sampling(GOSS): 정보량이 적은것은 제외하고 학습을 진행하자
        * 2.Exclusive Feature Bundling(EFB): 변수들을 합치자
        * leaf-wise tree growth로, tree를 수평이 아닌 '수직'으로 확장하여 leaf node의 개수를 정함
        * LightGBM의 리프 중심 트리 분할 방식은 트리의 균형을 맞추지 않고, 최대 손실 값(max delta loss)을 가지는
        * 리프 노드를 지속적으로 분할하면서 트리의 깊이가 깊어지고 비대칭적인 규칙 트리가 생성됨

        ![image](https://user-images.githubusercontent.com/77199749/204253228-215111e1-80da-4773-8399-75b1ea42bfc7.png)

        * 결과해석: AdaBoost와 동일한 조건(max_depth = 10, n_estimator=100)으로 진행하였을때보다 성능이 '9%'향상되었으며, GBM보단 '0.2% 낮은 성능'으로 거의 비슷해 보인다. GBM과 gradient의 값을 반영시켜 잔차를 줄이는 것은 동일하나, light라는 이름이 붙은 만큼 피쳐를 병합해 데이터크기를 작게 만들어 메모리를 적게 차지하고 이를 통해 LightGBM은\는 GBM에 비해 '170배 이상 빠르게' 훈련 및 테스트 된다. 추가로, learning_rate를 0.1에서 0.01로 감소한 결과를 보면 gradient의 감소가 천천히 일어나 성능이 더 떨어지는 것을 볼 수 있다. local에 빠질 우려가 있어 낮추었으나, 0.1이 이번데이터엔 더 적절한 learning rate값임을 알 수 있었다.

    5. CatBoost
        * GBM에서 출발한 알고리즘으로, 'Categorical'데이터의 boosting을 수행하는 모델임
        * Issue 1.Predcition Shift: 후반으로 갈수록 초반에 만들어진 모델에 영향을 많이 받는 문제가 발생함
            --> 그 이유는, 동일한 데이터가 여러번 중복학습되기 때문임.
            (Solution): Ordered Boosting: oblivious tree(대칭)구축하고, 제한적인 데이터를 학습에 사용하게 함
                이떄, 각 데이터들에 대해 leaf value와 gradient에 대한 코사인 유사도를 계산해 loss로 사용함!!
        * Issue 2.Target Leakage: 2번째 문제는 카테고리변수들은 one-hot encoding을 통해 변환이 되는데 y값은 그대로 유지되는 문제가 발생함
            (Solution): Ordered Target Statistics: 사전확률값을 도입한 ordered target statistics를 사용하며,
                rare한 noisy categories들의 부정적인 효과를 제거함

        ![image](https://user-images.githubusercontent.com/77199749/204253276-c0af524e-a05a-45a8-8ccb-f079e9ba6096.png)

       * 결과해석: AdaBoost와 동일한 조건(max_depth = 10, n_estimator=100)으로 진행하였을때보다 성능이 '6%' 향상되었으며, GBM보단 '3% 낮은 성능'을 보인다. GBM의 target leakage와 prediction shift 문제점을 보완한 모델이 바로 이 CatBoost이다. 해당 데이터에선 값들이 모두 0과1로만 이루어져 있기 때문에 categorical한 데이터임에도 성능이 더 낮게 나왔음을 볼 수 있다. 데이터의 수가 많고 binary값이 아닌  데이터를 실험한 결과에선 CatBoost의 성능이 더 좋게 나을 수 있다고 기대해본다.  마지막으로, CatBoost는 GBM에 비해 '15배 이상 빠르게' 훈련 및 테스트 되는 것을 볼 수 있다.
    
    * 최종실험결과비교(정확도 높은순, base_learner: tree, max_depth=10, n_estimator=100)
    
        1.GBM 정확도: 81.5988%
        
        2.LightGBM 정확도: 81.3995%
        
        3.XGBoost 정확도: 79.5626%
        
        4.CatBoost 정확도: 78.5876%
        
        5.AdaBoost 정확도: 72.4790%
        

  
</details>


## 5. Semi-supervised Learning
<details>
    <summary> View Contents </summary>
    
  * Semi-supervised Learning: asdf
  
  * Consistency Regularization
    * asdf
    
  * Holisitic Methods
    * asdf


  * 아래 Consistency Regularization기반 기법들을 소개함
      1. 𝚷−Model

          * ㅁ


          
          * 결과해석: ㅁ
          
          
      2. Temporal Ensemble

          * ㅁ
          
          
          
          * 결과해석: ㅁ
          
         
          
      3. Mean Teacher

          * ㅁ
          
          
          
          * 결과해석: ㅁ
          
          
      4. Dual Students

          * ㅁ
          
          
          
         
          * 결과해석: ㅁ
          
          
      5. FastSWA

          * ㅁ
          
          
          
          * 결과해석: ㅁ
          


    * 최종실험결과비교(정확도 높은순, dataset: CIFAR10, batch_size = 256)
    
        1.VAT 정확도: 65.07%
        
        2.Mean Teacher 정확도: 59.29%
        
        3.Pi-Model 정확도: 59.14%
        
        
        

    * 최종실험결과비교(정확도 높은순, dataset: MNIST, batch_size = 64)
    
        1.Mean Teacher 정확도: 99.38%
        
        2.Temporal Ensemble 정확도: 95.20%
        

  
</details>
    
=========================================================================
## Reference
- https://sustaining-starflower-aff.notion.site/2022-2-0e068bff3023401fa9fa13e96c0269d7 (강필성 교수님 ba수업자료)
