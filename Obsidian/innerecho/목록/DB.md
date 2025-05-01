# user

| 컬럼명          | 기본값   | Not Null | 데이터타입        | 컬럼타입    | 키정보 | 코멘트                  |
| ------------ | ----- | -------- | ------------ | ------- | --- | -------------------- |
| user_id      | -     | YES      | INT          | INT     | PRI | 사용자 ID (Primary Key) |
| user_email   | -     | YES      | VARCHAR(254) | VARCHAR | UNI | 사용자 이메일 (아이디 역할)     |
| password     | -     | YES      | VARCHAR(50)  | VARCHAR | -   | 비밀번호                 |
| user_name    | -     | YES      | VARCHAR(50)  | VARCHAR | -   | 사용자 이름               |
| user_gender  | -     | YES      | VARCHAR(5)   | VARCHAR | -   | 사용자 성별               |
| state        | NULL  | NO       | VARCHAR(50)  | VARCHAR | -   | 사용자 상태 값             |
| phone_number | NULL  | NO       | VARCHAR(50)  | VARCHAR | -   | 전화번호                 |
| birth_date   | NULL  | NO       | DATE         | DATE    | -   | 생년월일                 |
| created_at   | NOW() | YES      | DATE         | DATE    | -   | 가입 날짜                |

---

# userEventInfo

| 컬럼 명          | 기본값   | Not Null | 데이터타입      | 컬럼타입    | 키정보 | 코멘트       |
| ------------- | ----- | -------- | ---------- | ------- | --- | --------- |
| user_event_id | -     | YES      | INT        | INT     | PRI | 유저 이벤트 ID |
| user_id       | -     | YES      | INT        | INT     | FK  | 사용자 ID    |
| event_id      | -     | YES      | INT        | INT     | FK  | 이벤트 ID    |
| plant_id      | -     | YES      | INT        | INT     | FK  | 식물 ID     |
| status        | -     | YES      | VARCHAR(5) | VARCHAR | -   | 진행 상태     |
| assigned_at   | NOW() | YES      | DATE       | DATE    | -   | 할당 날짜     |
| completed_at  | NOW() | YES      | DATE       | DATE    | -   | 완료 날짜     |

---

# eventInfo

| 컬럼명           | 기본값   | Not Null | 데이터타입   | 컬럼타입         | 키 정보 | 코멘트                  |
| ------------- | ----- | -------- | ------- | ------------ | ---- | -------------------- |
| event_id      | -     | YES      | INT     | INT          | PRI  | 이벤트 ID (Primary Key) |
| event_title   | -     | YES      | VARCHAR | VARCHAR(256) | -    | 이벤트 제목               |
| event_content | -     | YES      | VARCHAR | VARCHAR(256) | -    | 이벤트 내용               |
| update_at     | NOW() | YES      | DATE    | DATE         | -    | 생성 날짜                |

---

# plant

| 컬럼명                | 기본값   | Not Null | 데이터타입       | 컬럼타입    | 키정보 | 코멘트          |
| ------------------ | ----- | -------- | ----------- | ------- | --- | ------------ |
| plant_id           | -     | YES      | INT         | INT     | PRI | 식물 ID        |
| user_id            | -     | YES      | INT         | INT     | FK  | 유저 ID        |
| species_id         | -     | YES      | INT         | INT     | FK  | 식물 종 ID      |
| nickname           | NULL  | NO       | VARCHAR(50) | VARCHAR | -   | 식물 이름        |
| plant_level        | NULL  | NO       | INT         | INT     | -   | 식물 레벨        |
| plant_experience   | NULL  | NO       | INT         | INT     | -   | 식물 경험치       |
| plant_hogamdo      | NULL  | NO       | INT         | INT     | -   | 식물 호감도       |
| last_measured_date | NOW() | YES      | DATE        | DATE    | -   | 최신 데이터 측정 시간 |

---

# plantGrowthLog

| 컬럼명        | 기본값 | Not Null | 데이터타입 | 컬럼타입 | 키정보 | 코멘트     |
| ---------- | --- | -------- | ----- | ---- | --- | ------- |
| history_id | -   | YES      | INT   | INT  | PRI | 히스토리 ID |
| content    | -   | YES      | TEXT  | TEXT | -   | 히스토리 내용 |
| user_id    | -   | YES      | INT   | INT  | FK  | 사용자 ID  |
| plant_id   | -   | YES      | INT   | INT  | FK  | 식물 ID   |
| timestamp  | -   | YES      | DATE  | DATE | -   | 기록 시간   |

---

Speices

