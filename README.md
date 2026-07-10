## Git 협업 규칙

### 브랜치 규칙

* `main` 브랜치에는 직접 Push하지 않습니다.
* 기능별 브랜치를 생성하여 작업합니다.

```git
branch

main
└── 모델명-이니셜

예시
├── RandomForest-OHM
├── XGBoost-JSH
├── LightGBM-KSY
└── CatBoost-LHJ
```
### 커밋 내용 규칙: 모델명 - 내용 (커밋 내용은 무조건 작성해주시기 바랍니다.)
```text
* 'RandomForest - 모델 생성'
* 'XGBoost - 모델 하이퍼파라미터 수정'
* 'XGBoost - 100% 데이터 저장'
```

### 작업 순서

1. `git switch -c 브랜치명` (브랜치 생성 및 이동)
2. `git status`
3. `git add .` 또는 `git add '작업 파일'`
4. `git commit -m "작업 내용"`
5. `git push origin 브랜치명`
6. Pull Request(PR) 생성
7. 리뷰 후 Merge


### Pull Request 규칙

* 본인 브랜치에서 `main`으로만 PR을 생성합니다.
* 다른 팀원의 브랜치로 PR을 생성하지 않습니다.
* 리뷰 없이 직접 Merge하지 않습니다.

---------------------


## 모델 및 데이터 파일명 규칙

| 구분         | 위치              | 파일명 규칙                                                                              | 예시                                                 |
|------------|-----------------|-------------------------------------------------------------------------------------|----------------------------------------------------|
| 분류 데이터     | `../data/`      | `churn_100'.csv`<br/>`churn_60'.csv`<br/>`churn_30'.csv`                            |                                                    |
| 모델 파일      | `./models/`     | `모델명.py`                                                                            | `DecisionTransformer.py`                           |
| 평가 결과      | `../data/eval/` | `모델명_100_eval.json`<br/>`모델명_60_eval.json`<br/>`모델명_30_eval.json`                   | `DecisionTransformer_100_eval.json`                |
| AUC-ROC 결과 | `../data/eval/` | `모델명_100_auc_roc.parquet`<br/>`모델명_60_auc_roc.parquet`<br/>`모델명_30_auc_roc.parquet` | `DecisionTransformer_100_auc_roc.parquet` |

#### 파일명 형식

* `모델명`
* `모델명_'100/60/30/2d'_eval.json`
* `모델명_'100/60/30/2d'_auc_roc.parquet`

  * `100` : 데이터 100%
  * `60` : 데이터 60%
  * `30` : 데이터 30%

#### 경로 규칙

* 모델 파일은 `./models/`에 생성합니다.
* 모델에서 데이터를 사용할 경우 `../data` 경로를 사용합니다.
* 평가 결과(`eval.json`)와 성능 결과(`auc_roc.parquet`)는 모두 `../data/eval`에 저장합니다.

### 데이터 저장
* 실행 시간 (`total_time_sec`)
* 평균 지연 시간 (`avg_latency_ms`)
* 샘플 수 (`total_samples`)
* Confusion Matrix
* AUC Score

#### Parquet 저장 내용

* `y_true`
* `y_score`

#### 저장 예시

```python
# 속도 계산
total_time
avg_latency_ms = (total_time / len(y_true)) * 1000

# 컨퓨전 매트릭스 계산 (임계값 0.5 기준)
y_pred = (y_scores >= 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred)
# [주의] sklearn confusion_matrix는 [[TN, FP], [FN, TP]] 순서입니다.
cm_list = cm.tolist() # JSON 저장을 위해 넘파이 배열을 파이썬 리스트로 변환

# AUC 수치 계산
auc_value = roc_auc_score(y_true, y_scores)

# 파케이는 AUC 그래프 그리기용으로 원천 데이터만 저장
df_roc = pd.DataFrame({"y_true": y_true, "y_score": y_scores})

summary_data = {
    "performance": {
        "total_time_sec": total_time,
        "avg_latency_ms": avg_latency_ms,
        "total_samples": len(y_true)
    },
    "confusion_matrix": cm.tolist(),
    "auc_score": auc_value
}

model_name = "DecisionTransformer" 
percent = "100"
save_path = "../data" 

with open(f"{save_path}/{model_name}_{percent}_eval.json", "w", encoding="utf-8") as f: 
  json.dump(summary_data, f, indent=4, ensure_ascii=False) 
  
df_roc.to_parquet( 
  f"{save_path}/{model_name}_auc_roc.parquet", index=False )
```
------------

```mermaid
flowchart TD

subgraph DATA["전체 데이터셋"]
    A["정수형 데이터 (... n개)<br/>범주형 데이터 (... n개)<br/>주관식 데이터 (... n개)"]
end

DATA -->|주관식 데이터 컬럼 삭제| B["정수형 데이터 (... n개)<br/>범주형 데이터 (... n개)"]

B --> C["정수형 데이터"]
B --> D["범주형 데이터"]
```

```text
'https://app.diagrams.net/'

추천 설정
파일 형식: .drawio (원본 저장)
README 삽입용: .svg (가장 추천) 또는 .png
저장 위치: ./images/

README에서는 이렇게 넣으면 됩니다.

## 데이터 전처리

<p align="center">
  <img src="./images/preprocessing_flow.svg" width="800">
</p>

팁
SVG로 저장하면 확대해도 깨지지 않습니다.
images/ 폴더를 따로 만들어 관리하면 README와 EDA.md에서 같은 이미지를 재사용할 수 있습니다.
도형 색상은 너무 화려하게 하기보다 흰색 배경 + 검은 테두리 + 파란색 강조 정도가 GitHub에서 가장 보기 좋습니다.

프로젝트 규모를 보면 전처리 → 모델 학습 → 평가까지 여러 다이어그램이 생길 가능성이 있으니, 
처음부터 images/ 폴더를 만들어 관리하는 것을 추천합니다.
```



































```text
Project_2/
│
├── models/
├── config/
├── data/
│      ├── eval/
│      ├── plots/
│      │   
│      ├── chrun.csv
│      ├── chrun_preprocessed.csv # 전처리와 인코딩 완료
│      ├── chrun_100.csv          # chrun_preprocessed.csv의 100% 사용 데이터
│      ├── chrun_60.csv           # chrun_preprocessed.csv의 60% 사용 데이터
│      └── chrun_30.csv           # chrun_preprocessed.csv의 30% 사용 데이터
├── images/
├── data/              
├── tests/
│
├── requirements.txt
├── README.md
├── EDA.md
├── .gitignore
└── main.py
```
