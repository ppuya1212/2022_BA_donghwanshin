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
      1. Single Gassusian -> parametric기법
      
      ![image](https://user-images.githubusercontent.com/77199749/201831498-e5f60dac-f6dd-48ca-8dfc-a87052a7b745.png)


      2. Gaussian Mixture -> parametric기법
      ![image](https://user-images.githubusercontent.com/77199749/201831111-2db4a24c-176a-4217-a45d-0235391c3cad.png)

      3. Parzen Window Density Estimation -> non-parametric기법
      ![image](https://user-images.githubusercontent.com/77199749/201831516-db5a7a33-4329-45b0-a5c7-875c601b031d.png)


      4. Local Outlier Factor(LoF) -> non-parametric기법
      
      ![image](https://user-images.githubusercontent.com/77199749/201831541-1e7cd125-89aa-4297-b7d6-4c9a65e60f43.png)


  * Model-based Anomaly Detection: 주어진 정상데이터만으로 각 모델들을 학습하여, 새로운 데이터가 들어왔을 때 각 모델의 기준에 부합하지 않는 데이터들을 이상치라 판단함.
  
  * 아래의 3가지 모델기반 이상치 탐지기법들을 소개함
    1. Auto-Encoder
    ![image](https://user-images.githubusercontent.com/77199749/201831412-ec22f679-70cc-4d8b-8317-4211a7a14235.png)
    
    2. One Class Support Vector Machine(OCSVM)
    ![image](https://user-images.githubusercontent.com/77199749/201831461-7e44eced-05fe-4b6f-b4de-3f99e07a17b7.png)

    3. Isolation Forest
    
    ![image](https://user-images.githubusercontent.com/77199749/201831471-541147c9-d165-422a-bfd3-cdffad30a401.png)

  
</details>
    
==========================================================================
## Reference
- https://sustaining-starflower-aff.notion.site/2022-2-0e068bff3023401fa9fa13e96c0269d7 (강필성 교수님 ba수업자료)
