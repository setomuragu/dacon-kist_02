# dacon-kist_02
데이콘 kist 주최 생육 예측

데이콘 kist 주최 생육환경 최적화 경진대회

적상추의 생육과정을 1분 간격으로 촬영된 이미지와 그와 동일한
적상추의 생육환경를 데이터로 제공되었다.

제공된 데이터를 이용하여 적상추의 생육정도를 알 수 있는 정량지표를 발굴하고
나아가 청경채의 사진과 환경 데이터로 앞으로의 잎면적을 예측하는 알고리즘 개발이 목표였다.

먼저 환경데이터의 상관관계를 알아보고 상관관계가 낮은 데이터를 drop 하였다.
적상추가 보이지 않는 이미지 사진을 이용하여 배경과 적상추를 분리하였다.
주어진 데이터가 적어 데이터 augmentation을 하였다. (flip과 시계방향으로 회전 등)

모델은 sklearn의 Linear Regression을 이용하여 구축하였다.


이미지 데이터를 색상으로 나눠 training이 필요해 보이며, 더 큰 네트워크를 이용해야 할 것이다.
