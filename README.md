# Motion In-betweening for Hand Motion — Model Capacity Study

SILK(2025)의 single-Transformer-encoder in-betweening 레시피를 손(hand) 회전 데이터(rot6D,
MANO 기반 15관절)에 적용한 두 가지 모델 크기 실험. `sl_core.py`/`sl_model_a.py`가 공통 코어이고,
`experiments/exp01_bigger_model/`, `experiments/exp06_full_silk_size/`가 이 리포의 핵심이다.

## 왜 이 두 모델인가

베이스라인(d_model=256, 4layer, ~3.2M params)으로 시작했는데, in-betweening 관련 선행 연구
전반에서 "모델 용량보다 데이터 양이 성능을 더 좌우한다"는 주장이 반복적으로 나온다. 이게 손
동작 도메인(관절 수가 몸 전체보다 훨씬 적고, 데이터도 상대적으로 적음)에서도 성립하는지
직접 검증하기 위해 모델 크기만 바꾼 두 단계를 만들었다.

- **exp01 (d_model=512, 6layer, 8head, ~19M params)**: 베이스라인 대비 약 6배.
- **exp06 (d_model=1024, 6layer, 8head, ~76M params)**: SILK 원 논문이 실제로 쓴 크기 그대로.
  exp01에서 개선이 계속되면 원 논문 크기까지 밀어붙였을 때 어디서 한계가 오는지 보려는 목적.

모델 구조 자체(`sl_model_a.SILKHand`)는 두 실험이 완전히 동일 — 학습 스텝 수, 데이터, 손실함수
전부 고정하고 `d_model`/`n_layers`/`n_heads`/`d_ff`만 바꿔서 순수하게 용량 효과만 분리했다
(`experiments/*/model.py`의 `build_model()` 참고).

## 데이터 / 표현

- How2Sign(SignSparK 전처리, MANO rot6D) 단독 사용, 오른손 + 왼손(오른손 규약으로 미러링)
  둘 다 포함 (`preprocess_offset5/windowed_dataset.py`).
- 시작점 offset=5 간격 슬라이딩 학습 윈도우(SILK 논문 실측값), 학습 시 가림 길이 T는 5~30
  사이 매 샘플마다 균일 랜덤.
- 위치 인코딩은 절대 위치가 아니라 **목표 키프레임까지 남은 거리 기준**(`sl_model_a.py`의
  `SILKHand.forward`) — 윈도우 길이가 T마다 바뀌는 구조에서 같은 절대 인덱스가 "목표
  프레임"과 "gap 중간"을 오가며 충돌하는 문제를 피하기 위한 설계.

## 결과 (test split 전수 채점, geodesic 회전 오차)

| T | SLERP 기준선 | exp01 (19M) | exp06 (76M) |
|---|---:|---:|---:|
| 5 | 3.12° | 1.39° | **1.32°** |
| 10 | 5.78° | 3.69° | **3.63°** |
| 20 | 9.06° | 7.20° | **7.13°** |
| 30 | 11.10° | 9.45° | **9.38°** |

두 모델 다 SLERP를 확실히 이기고, 76M이 19M보다 전 구간에서 근소하게 낫다 — 이 스케일까지는
용량을 늘리는 게 여전히（작지만) 도움이 됐다. `results/`에 오차 곡선(`error_curves_final.png`)과
GT/SLERP/6개 모델을 나란히 비교한 3D mesh 애니메이션(`.gif`, context/gap 구간을 색으로 구분)이
있다.

## 재현

```bash
# 필요한 것: SignSparK How2Sign LMDB (경로는 windowed_dataset.py 상단 DATA_ROOT에서 설정)
cd experiments/exp01_bigger_model   # 또는 exp06_full_silk_size
python train_offset5.py --steps 127560 --batch_size 64
```
