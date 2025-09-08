# Reddit API 설정 가이드

## Reddit API 자격 증명 얻기

### 1. Reddit 개발자 계정 생성
1. [Reddit Apps](https://www.reddit.com/prefs/apps) 페이지 방문
2. Reddit 계정으로 로그인
3. "Create App" 또는 "Create Another App" 클릭

### 2. 앱 정보 입력
- **Name**: `MemeStockAnalysis` (또는 원하는 이름)
- **App type**: `script` 선택
- **Description**: `Meme stock analysis data collection`
- **About URL**: 비워두기
- **Redirect URI**: `http://localhost:8080` (필수)

### 3. 자격 증명 확인
앱 생성 후 다음 정보를 확인:
- **Client ID**: 앱 이름 아래에 있는 문자열 (예: `abc123def`)
- **Client Secret**: "secret" 라벨이 붙은 문자열

### 4. 설정 파일 업데이트
`config/reddit_config.json` 파일을 편집하여 실제 값으로 교체:

```json
{
    "client_id": "실제_CLIENT_ID_입력",
    "client_secret": "실제_CLIENT_SECRET_입력",
    "user_agent": "MemeStockAnalysis/1.0 (by /u/당신의_레딧_사용자명)",
    "username": "당신의_레딧_사용자명",
    "password": "당신의_레딧_비밀번호"
}
```

## 필요한 Python 패키지 설치

```bash
pip install praw pandas
```

## 사용법

### 1. 설정 파일 준비
위 가이드에 따라 `config/reddit_config.json` 파일을 설정

### 2. 스크립트 실행
```bash
python scripts/download_extended_reddit_data.py
```

### 3. 다운로드되는 데이터
- **2020년 WSB 포스트**: `wsb_2020_extended.csv`
- **2021년 WSB 포스트**: `wsb_2021_extended.csv`  
- **2022년 WSB 포스트**: `wsb_2022_extended.csv`
- **통합 데이터셋**: `wsb_2020_2022_combined.csv`
- **통계 요약**: `daily_post_counts.csv`, `monthly_post_counts.csv`, `top_100_posts.csv`

## 주의사항

### Reddit API 제한
- **Rate Limiting**: 초당 요청 수 제한
- **데이터 접근**: 공개 포스트만 접근 가능
- **인증**: 사용자 계정으로 로그인 필요

### 데이터 품질
- **키워드 기반 검색**: GME, AMC, BB 등 밈주식 관련 포스트 우선
- **시간 범위**: 2020-2022년 전체 기간
- **메타데이터**: 제목, 점수, 댓글 수, 작성 시간 등 포함

## 문제 해결

### 일반적인 오류
1. **인증 실패**: 자격 증명 확인
2. **Rate Limit**: 요청 간격 늘리기
3. **파일 권한**: 출력 디렉토리 쓰기 권한 확인

### 로그 확인
스크립트 실행 시 상세한 로그가 출력되어 진행 상황을 모니터링할 수 있습니다.
