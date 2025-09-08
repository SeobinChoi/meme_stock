# 📁 Data Directory Structure

## 🎯 **Overview**
데이터 파일들을 체계적으로 분류하여 가독성과 관리 효율성을 향상시켰습니다.

## 📂 **Raw Data Structure**

```
data/raw/
├── 📊 reddit/                    # 레딧 관련 데이터
│   ├── reddit_wsb.csv           # 일별 WSB 데이터 (1,096줄)
│   ├── raw_reddit_wsb.csv      # 원본 WSB 데이터 (43MB)
│   ├── additional_reddit_data_structure.csv
│   ├── news_data_structure.csv
│   └── options_data_structure.csv
│
├── 📈 stocks/                    # 주식 가격 데이터
│   ├── GME_extended_stock_data.csv      # GME 확장 데이터 (580KB)
│   ├── GME_extended_data_DESCRIPTION.txt
│   ├── GME_enhanced_stock_data.csv      # GME 향상 데이터
│   ├── GME_stock_data.csv               # GME 기본 데이터
│   ├── GME_stock_data_DESCRIPTION.txt
│   ├── GME_quality_metadata.json
│   ├── AMC_extended_stock_data.csv      # AMC 확장 데이터 (570KB)
│   ├── AMC_extended_data_DESCRIPTION.txt
│   ├── AMC_enhanced_stock_data.csv      # AMC 향상 데이터
│   ├── AMC_stock_data.csv               # AMC 기본 데이터
│   ├── AMC_stock_data_DESCRIPTION.txt
│   ├── AMC_quality_metadata.json
│   ├── BB_extended_stock_data.csv       # BB 확장 데이터 (584KB)
│   ├── BB_extended_data_DESCRIPTION.txt
│   ├── BB_enhanced_stock_data.csv       # BB 향상 데이터
│   ├── BB_stock_data.csv                # BB 기본 데이터
│   ├── BB_stock_data_DESCRIPTION.txt
│   ├── BB_quality_metadata.json
│   ├── TSLA_extended_stock_data.csv     # TSLA 확장 데이터
│   ├── TSLA_extended_data_DESCRIPTION.txt
│   ├── AAPL_extended_stock_data.csv     # AAPL 확장 데이터 (593KB)
│   ├── AAPL_extended_data_DESCRIPTION.txt
│   ├── SPY_extended_stock_data.csv      # SPY 확장 데이터
│   ├── SPY_extended_data_DESCRIPTION.txt
│   └── QQQ_extended_stock_data.csv      # QQQ 확장 데이터
│
├── 🪙 crypto/                    # 암호화폐 데이터
│   ├── BTC_crypto_data.csv      # 비트코인 데이터 (230KB)
│   ├── ETH_crypto_data.csv      # 이더리움 데이터 (255KB)
│   └── DOGE_crypto_data.csv     # 도지코인 데이터 (275KB)
│
├── 📊 indices/                   # 지수 데이터
│   ├── DJI_index_data.csv       # 다우존스 지수 (153KB)
│   ├── GSPC_index_data.csv      # S&P 500 지수 (166KB)
│   ├── IXIC_index_data.csv      # 나스닥 지수 (162KB)
│   ├── TNX_index_data.csv       # 10년 국채 수익률 (165KB)
│   └── VIX_index_data.csv       # 변동성 지수 (164KB)
│
├── 📚 archive/                   # 아카이브 데이터
│   ├── archive-2/               # 2021년 이전 데이터
│   │   ├── AMC.csv
│   │   ├── BlackBerry.csv
│   │   ├── GameStock.csv
│   │   └── Wish.csv
│   └── archive-3/               # 2021-2023년 월별 데이터
│       ├── 2021/                # 2021년 월별 데이터
│       │   ├── wallstreetbets_2021.csv
│       │   ├── gme_2021.csv
│       │   ├── stocks_2021.csv
│       │   ├── superstonk_2021.csv
│       │   ├── cryptocurrency_2021.csv
│       │   ├── investing_2021.csv
│       │   ├── options_2021.csv
│       │   ├── pennystocks_2021.csv
│       │   ├── spacs_2021.csv
│       │   └── stockmarket_2021.csv
│       ├── 2022/                # 2022년 월별 데이터
│       │   ├── wallstreetbets_2022.csv
│       │   ├── cryptocurrency_2022.csv
│       │   ├── investing_2022.csv
│       │   ├── options_2022.csv
│       │   ├── pennystocks_2022.csv
│       │   ├── shortsqueeze_2022.csv
│       │   ├── spacs_2022.csv
│       │   ├── stockmarket_2022.csv
│       │   └── stocks_2022.csv
│       └── 2023/                # 2023년 월별 데이터
│           ├── wallstreetbets_2023.csv
│           ├── cryptocurrency_2023.csv
│           ├── investing_2023.csv
│           ├── options_2023.csv
│           ├── pennystocks_2023.csv
│           ├── stockmarket_2023.csv
│           └── stocks_2023.csv
│
└── 📋 docs/                      # 문서 및 메타데이터
    ├── ENHANCED_DATASET_DESCRIPTION.md
    ├── extended_data_collection_plan.json
    ├── HISTORICAL_STOCK_DATA_DESCRIPTION.md
    └── quality_summary.txt
```

## 🔍 **Data Collection Summary**

### **Reddit Data**
- **WSB 일별 데이터**: 1,096줄 (2021년 1월 1일 ~ 현재)
- **Archive 데이터**: 2021-2023년 월별 서브레딧별 데이터
- **총 서브레딧**: 26개 (wallstreetbets, gme, stocks, superstonk, cryptocurrency 등)

### **Stock Data**
- **메인 주식**: GME, AMC, BB, TSLA, AAPL, SPY, QQQ
- **데이터 기간**: 2019년 1월 ~ 현재
- **기술적 지표**: SMA, EMA, RSI, MACD, Bollinger Bands, Volume 등
- **파일 크기**: 각 주식당 570-593KB, 1,511줄

### **Crypto Data**
- **메인 코인**: BTC, ETH, DOGE
- **데이터 기간**: 2019년 ~ 현재
- **파일 크기**: 230-275KB

### **Index Data**
- **주요 지수**: DJI, S&P 500, NASDAQ, TNX, VIX
- **데이터 기간**: 2019년 ~ 현재
- **파일 크기**: 153-166KB

## 📝 **Benefits of Reorganization**

1. **가독성 향상**: 관련 데이터를 카테고리별로 분류
2. **접근성 개선**: 특정 데이터 타입을 쉽게 찾을 수 있음
3. **유지보수 용이**: 각 데이터 타입별로 독립적인 관리
4. **확장성**: 새로운 데이터 타입 추가 시 적절한 카테고리에 배치 가능
5. **문서화**: 메타데이터와 설명 파일을 별도 디렉토리에 정리

## 🚀 **Next Steps**

1. **데이터 통합 파이프라인 구축**: 정리된 구조를 활용한 효율적인 데이터 처리
2. **데이터 품질 모니터링**: 각 카테고리별 데이터 품질 검증
3. **자동화 스크립트**: 새로운 데이터 수집 시 자동으로 적절한 카테고리에 배치
4. **백업 및 버전 관리**: 체계적인 데이터 백업 전략 수립
