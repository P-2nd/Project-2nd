# EDA (Exploratory Data Analysis)

## 1. 데이터셋 정보

- **데이터명** : Synthetic Gym Membership Churn Dataset (1M Rows)
- **출처** : Kaggle (Emin Karlıtepe)
- **데이터 개수** : 1,000,000건 (Rows)
- **컬럼 수** : 19개 (Columns)
- **목표 변수(Target)** : Churn (0: 유지, 1: 이탈)
---

## 2. 컬럼 정보

| 컬럼명                           | 설명                | Type               |
| ----------------------------- | ----------------- |---------------- |
| `Member_ID`                   | 고유 회원 식별자         | int64 |
| `Age`                         | 고객 연령             | int64 |
| `Gender`                      | 성별 범주             | str|
| `Membership_Type`             | 헬스장 회원권 플랜        |str|
| `Membership_Start_Date`       | 멤버십 시작일           |str|
| `Monthly_Fee`                 | 월간 구독료            |float64|
| `Monthly_Visits`              | 월간 방문 횟수          |int64 |
| `Avg_Workout_Duration_Min`    | 평균 운동 시간(분)       |int64 |
| `Peak_Hour_Preference`        | 선호하는 헬스장 이용 시간    |str|
| `Cardio_Preference`           | 선호하는 유산소 운동 기구    |str|
| `Treadmill_Avg_Speed_Kmh`     | 러닝머신 평균 속도(km/h)  |float64|
| `Treadmill_Avg_Incline_Pct`   | 러닝머신 평균 경사도(%)    |float64|
| `Group_Class_Attendance`      | 그룹 수업 참여 횟수       |int64 |
| `PT_Session_Count`            | 개인 트레이너(PT) 세션 횟수 |int64 |
| `Supplement_Usage`            | 보충제 사용 여부/범주      |str|
| `Avg_Equipment_Wait_Time_Min` | 평균 장비 대기 시간(분)    |float64|
| `Late_Payment_Count`          | 연체 건수             |int64|
| `Profile_Type`                | 시뮬레이션된 고객 프로필_    |str|
|`Churn`| 이탈(0/1)|int64|


### 전처리 전 데이터 타입 확인

| Type     | 개수 |
|----------|----|
| int64    | 8  |
| float    | 4  |
| str      | 7  |

---

## 3. 결측치 확인

### 결측치 개수

| 컬럼명 | 결측치    | 처리 방법                          | 처리 이유                           |
|--------|--------|--------------------------------|---------------------------------|
|Cardio_Preference | 226685 | None -> No Preference          | 유산소 운동을 선호 하지 않는 경우를 고려         |
|Treadmill_Avg_Speed_Kmh | 662402 |                    Median            | 연속형 수치 데이터이며 결측치가 많기 때문에 제거 불가능 |
|Treadmill_Avg_Incline_Pct | 662402 |                 Median               |  연속형 수치 데이터이며 결측치가 많기 때문에 제거 불가능          |
|Supplement_Usage | 419573 | None -> No Protein Supplements | 프로틴을 사용하지 않는 경우를 고려             |

---

## 4. 중복 데이터 확인

- **중복 데이터 개수** : 0
- **처리 방법** : 없음
- **처리 이유** : 없음

---

## 5. 이상치 확인

### 확인 방법

- Box Plot

<img alt="boxplot.png" src="data/eda/boxplot.png"/>

- IQR

<img alt="iqr_result.png" src="data/eda/iqr_result.png"/>


### 처리 결과

처리 결과
- Age, Monthly_Visits, PT_Session_Count에서 일부 이상치가 확인되었으나 실제 회원의 다양한 이용 패턴을 반영하는 값으로 판단하여 제거하지 않았다.
- Treadmill_Avg_Speed_Kmh와 Treadmill_Avg_Incline_Pct는 결측치 비율이 매우 높아 중앙값으로 대체하였으며, 이로 인해 IQR 기준 이상치가 증가한 것으로 판단되어 추가적인 이상치 제거는 수행하지 않았다.

---

## 6. 데이터 분포

### 수치형 변수

- Histogram
- Box Plot

### 범주형 변수

- Count Plot
- Value Counts

---

## 7. 상관관계 분석

### 분석 방법

- Correlation Matrix
- Heatmap

### 주요 결과

| 변수1 | 변수2 | 상관계수 |
|--------|--------|---------|
| | | |

---

## 8. 타겟 변수 분석

- 클래스 분포
- 클래스 불균형 여부

---

## 9. 데이터 전처리

### 9.1 컬럼 삭제

| 컬럼명 | 삭제 이유 |
|--------|-----------|
| | |

### 9.2 결측치 처리

| 컬럼명 | 처리 방법 | 처리 이유 |
|--------|----------|-----------|
| | | |

### 9.3 이상치 처리

| 컬럼명 | 처리 방법 | 처리 이유 |
|--------|----------|-----------|
| | | |

### 9.4 feature 엔지니어링

| 컬럼명 | 처리 방법     | 처리 이유 |
|--------|-----------|-----------|
| | 데이트 타입 변환 | |

### 9.5 인코딩

| 컬럼명 | 인코딩 방법 | 선택 이유 |
|--------|------------|-----------|
| | | |

### 9.6 스케일링

| 컬럼명 | 스케일링 방법 | 선택 이유 |
|--------|--------------|-----------|
| | | |

### 9.7 Feature Selection

| Feature | 선택 여부 | 선택 이유 |
|----------|----------|-----------|
| | | |

### 전처리 후 최종 데이터 확인

| 항목 | 값 |
|------|----|
| 샘플 수 | |
| 컬럼 수 | |
| 수치형 | |
| 범주형 | |

| Type | 개수 |
|------|------|
| int | |
| float | |
| object | |
| category | |
| datetime | |


### 9.7 데이터 분할

## Feature Importance

(이미지)

| Feature | Importance |
|----------|-----------|
| | |


| 데이터     | 비율       | 파일명 |
|---------|----------|--------|
| 전체      | 80%, 20% | `churn_100.csv` |
| 전체의 50% | 80%, 20% | `churn_50.csv` |



---

## 10. 최종 데이터셋

| 항목 | 값 |
|------|----|
| 샘플 수 | |
| 컬럼 수 | |
| 수치형 컬럼 수 | |
| 범주형 컬럼 수 | |

---

## 11. 저장 파일

| 파일명 | 설명 |
|--------|------|
| `churn_100.csv` | 전체 데이터 |
| `churn_50.csv` | 50% 데이터 |

---

## 12. 결론

- 주요 데이터 특징
- 수행한 전처리 요약
- 전처리 이유
- 모델 학습 시 고려사항
- 데이터의 한계 및 개선 방향 (선택)
