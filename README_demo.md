# 공공데이터를 활용한 서울시 버스 노선 정보 및 정류장 위치 시각화 서비스

---
## 1. 팀 소개 - 6팀
* **🧑‍🔬 팀명:** **버스킹 (BusKing) -** 거리의 음악가(Busking)가 아닌, 버스 데이터의 왕(Bus-King)이 되겠다는 포부
* **👥 팀원:** 김나은, 김유진, 박기현, 오호민, 이준희A
* **🔗 멤버 개인 깃허브 계정 연동**<br>
	- 김나은: naeuneo (nangni30@naver.com) <br>
	- 김유진: ujinkim2001 (ujinkim@naver.com)<br>
	- 박기현: HYonthePark (rlgusqkr102@gmail.com)<br>
	- 오호민: necknam (gom532454@gmail.com)<br>
	- 이준희A: june229 (mswkdh3790@gmail.com)

## 2. 프로젝트 개요
* **🎡프로젝트 명**
	: 서울시 버스 노선 및 정류장 조회 시스템 (Seoul Bus Information System)
    
* **📃프로젝트 소개**
	: 서울시 공공데이터를 활용하여 사용자가 원하는 버스 노선 정보, 정류장 위치, 최적의 이동 경로(출발-도착지 기반)를 직관적인 웹/앱 인터페이스를 통해 실시간으로 조회하고 지도에서 확인할 수 있는 통합 버스 정보 서비스입니다.
    
* **프로젝트 필요성(배경)**  
  ⓐ 서울시는 대한민국에서 가장 복잡하고 밀도 높은 대중교통망을 가지고 있습니다.<br>
  ⓑ 따라서 단순 버스 노선 확인 및 정류장 조회를 원하는 사용자에게 무겁고 복잡할 수 있습니다.<br>
  ⓒ 이에 따라 버스 노선과 정류장 데이터에 집중하여, 효율적인 '조회/검색/지도 시각화'를 제공하는 특화된 서비스의 필요성을 느껴 본 프로젝트를 기획하게 되었습니다.

* **🎯프로젝트 목표**
    - 사용자 편의성 극대화: 복잡한 서울시 버스 정보 중 사용자가 원하는 내용을 클릭 또는 입력으로 직관적으로 확인할 수 있는 UI/UX를 제공한다.<br>
    - 공공데이터의 효율적 가공: 대용량의 서울시 버스 공공데이터 API를 안정적으로 파싱하고 DB화하여, 정확하고 효율적인 조회 성능을 확보한다.

## 3. 데이터 수집 방법
    1) Open API :** T Data(서울교통빅데이터플랫폼) https://t-data.seoul.go.kr
      	 - 활용 Open API 목록
    	   ⓐ 노선정보	 	https://t-data.seoul.go.kr/dataprovide/trafficdataviewopenapi.do?data_id=1053
    	   ⓑ 정류장 정보		https://t-data.seoul.go.kr/dataprovide/trafficdataviewopenapi.do?data_id=1064
         - 수집 방법: Open API를 호출하여 JSON 데이터로 저장한 뒤, 프로젝트 DB에 테이블 형태로 저장하였습니다.
	2) CVS : T Data(서울교통빅데이터플랫폼) https://t-data.seoul.go.kr
     	   ⓒ 노선별 경유노드 	https://t-data.seoul.go.kr/dataprovide/trafficdataviewfile.do?data_id=58

## 4. DB 설계(논리/물리 ERD)

---

## 논리 모델
<img width="1242" height="792" alt="image" src="https://github.com/user-attachments/assets/51742baa-45f0-43fd-a303-315440a3c24c" />

---

### 물리 모델
<img width="1806" height="766" alt="image" src="https://github.com/user-attachments/assets/82446fba-8f1d-484f-a75d-7803215f6ec8" />

---

## 5. 주요 기능
### 1) 지도조회 (정류장 검색 및 지도 표시)
정류장명을 부분 입력하면 일치하는 모든 정류장이 실시간 자동완성 리스트로 표시되고, 선택 시 해당 위치가 지도에 표시된다.

### 2) 정류장조회 (정류장별 경유 노선 조회)
정류장을 입력하면 해당 정류장을 경유하는 모든 버스 번호를 표시한다.

### 3) 버스조회(노선별 상세정보 및 경유 정류장 조회)
버스명이나 번호를 입력하면 관련 노선번호가 뜨고, 첫차/막차 시간, 배차 간격, 요금, 해당 노선이 경유하는 모든 정류장을 기점부터 종점까지 순서대로 표시한다.

### 4) 경로조회 (출발지-도착지 기반 노선 검색)
출발 정류장과 도착 정류장을 입력하면 두 정류장을 모두 경유하는 노선과 경유 정류장 개수를 표시한다.

### 5) FAQ (질의응답 조회)
질문 키워드를 입력하면 관련 질문이 뜨고 해당 질문을 클릭하면 답변을 표시한다.
   
## 6. 실행 방법
api_run.py 실행
정류장 정보 및 노선 정보를 수집하여 데이터베이스에 저장

---

JSON 데이터 생성
JSON 생성 파일
create_station_api_json.py : 정류장 정보를 JSON 파일로 변환 및 저장
create_route_api_json.py : 노선 정보를 JSON 파일로 변환 및 저장

실행 방법
api_json_run.py 실행
데이터베이스 또는 API 데이터를 JSON 파일로 생성 및 저장

---

**실행 순서**
1. DB 초기화<br>
_01_project_init.sql<br>
_02_table_create.sql<br>
_03_insert_table.sql<br>

2. 필요 파이썬 모듈 설치<br>
pip install -r requirements.txt

3. API 데이터 적재<br>
api_run.py

4. JSON 파일 생성<br>
api_json_run.py

5. 서비스 실행<br>
app/main.py

## 7. 수행 화면 캡처
<img width="1898" height="858" alt="경로조회1" src="https://github.com/user-attachments/assets/a117fa53-ea27-4f88-a559-fa270a71b23d" />

<img width="1907" height="862" alt="경로조회2" src="https://github.com/user-attachments/assets/de7804a2-5aba-4dac-846c-3e5cba638b82" />

<img width="1906" height="865" alt="버스조회" src="https://github.com/user-attachments/assets/faa1a59f-0d55-4eff-ad6f-55278c18df60" />

<img width="1902" height="865" alt="정류장조회" src="https://github.com/user-attachments/assets/b6c34b41-b434-4fcd-ae5b-de7e3ba28174" />

<img width="1902" height="866" alt="지도조회" src="https://github.com/user-attachments/assets/659d8322-098e-4515-9898-342bf9f6cbe1" />

## 8. 회고

### 김나은 (데이터 검색, ERD 테이블 제작, 데이터 정제 및 적재, Streamlit 구성) ⭐
- 공공데이터 API의 XML 데이터를 파싱하여 DB에 적재하는 과정에서 데이터 누락 및 형식 불일치 문제를 발견했고, 정규식과 예외 처리를 통해 안정적으로 데이터를 수집할 수 있었습니다.
- ERD 설계부터 데이터 적재까지 전 과정을 경험하면서 데이터 모델링과 정규화가 실제 서비스 구현에 얼마나 중요한지 배울 수 있었습니다.
- 다양한 형태의 원천 데이터를 가공하고 정제하는 과정을 통해 데이터 엔지니어링 업무에 대한 이해도를 높일 수 있었습니다.

### 김유진 (API 연동, 데이터 저장, Streamlit 구현)
- Open API 데이터를 JSON 파일로 포맷팅하고, SQL 테이블 구조에 맞게 연결하는 과정을 통해 백엔드 환경에서의 데이터 정제 및 관리 프로세스를 이해하게 되었습니다.
- API 데이터와 데이터베이스를 연동하면서 실제 서비스에서 데이터가 활용되는 전체 흐름을 이해할 수 있었습니다.
- Streamlit을 활용해 데이터를 사용자에게 효과적으로 제공하는 화면을 구현하며 데이터 시각화와 UI 구성 역량을 키울 수 있었습니다.

### 박기현 (ERD 구성 및 테이블 제작, streamlit 구조 설계, PPT 제작, 발표)
- 출발지와 도착지를 모두 만족하는 노선을 검색하는 SQL 쿼리를 작성하면서 JOIN과 서브쿼리 활용 능력을 향상시킬 수 있었습니다.
- 인덱스(Index) 설정에 따른 조회 성능 차이를 직접 확인하며 데이터베이스 최적화의 중요성을 체감할 수 있었습니다.
- 프로젝트 발표를 준비하면서 기술적인 내용을 효과적으로 전달하는 방법과 문서화의 중요성을 배울 수 있었습니다.

### 오호민 (전체 구조 구상, ERD 구성, Streamlit 구조 설계)
- 프로젝트 전체 아키텍처를 설계하며 데이터 수집, 저장, 조회, 시각화까지의 흐름을 체계적으로 구성하는 경험을 할 수 있었습니다.
- Streamlit 구조를 모듈화하고 컴포넌트를 재사용할 수 있도록 설계하면서 유지보수성과 확장성을 고려한 개발의 중요성을 배웠습니다.
- 모바일 환경에서도 편리하게 사용할 수 있도록 UI/UX를 개선하며 사용자 중심 설계 경험을 쌓을 수 있었습니다.

### 이준희A (API 연동, 데이터 저장, Streamlit 구현)
- GitHub 브랜치 전략(Git Flow)을 활용한 협업을 경험하며 버전 관리와 코드 병합 과정을 실무처럼 경험할 수 있었습니다.
- 코드 리뷰를 통해 다양한 구현 방식을 학습하고, 더 효율적인 코드 작성 방법을 고민할 수 있었습니다.
- 데이터 저장부터 화면 출력까지의 전체 개발 과정을 경험하며 백엔드와 프론트엔드의 연계 구조를 이해할 수 있었습니다.


