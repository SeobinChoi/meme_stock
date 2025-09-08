Reddit 소셜미디어 데이터를 활용한 밈스톡의 Contrarian Effect 분석

김건희, 최서빈*, 황유진**
서강대학교
gunhee@sogang.ac.kr, *seobin@sogang.ac.kr, **yujin@sogang.ac.kr

Analysis of Contrarian Effects in Meme Stocks Using Reddit Social Media Data
Gunhee Kim, Seobin Choi, Yujin Hwang

요 약

본 연구는 Reddit r/wallstreetbets 데이터를 활용하여 밈스톡에서 나타나는 contrarian effect를 분석하였다. 2021-2023년 기간 동안 GME, AMC, BB 등 주요 밈스톡의 Reddit 관심도와 주가 수익률 간의 관계를 종합적으로 분석한 결과, 일관된 음의 상관관계(-0.198~-0.165)를 발견하였다. 이는 Reddit에서 관심이 급증할 때 오히려 수익률이 하락하는 contrarian effect를 의미한다. Event study, 시간적 패턴 분석, 감정 분석을 통해 이 현상의 강건성을 확인하였으며, 53,187개 Reddit 포스트의 텍스트 분석을 통해 행동재무학적 메커니즘을 실증적으로 검증하였다. 특히 확신 표현이 많을수록 실제 성과가 나쁜 'behavioral paradox'를 발견하여 기존 소셜미디어-주식 연구와 차별화된 결과를 제시하였다.

**키워드**: 밈스톡, Contrarian Effect, 소셜미디어, 행동재무학, Reddit

Ⅰ. 서 론

2021년 GameStop(GME) 사태를 필두로 한 밈스톡(meme stock) 현상은 소셜미디어가 금융시장에 미치는 영향을 극명하게 보여주었다. Reddit의 r/wallstreetbets 커뮤니티에서 시작된 개인투자자들의 집단 행동은 전통적인 시장 메커니즘을 뒤흔들며 새로운 연구 영역을 열었다.

기존 연구들은 주로 소셜미디어 감정과 주가 간의 양의 상관관계에 초점을 맞춰왔다[1,2]. Twitter 감정 분석을 통한 주가 예측[3]이나 Reddit 게시물 수와 주가 상승의 관계[4] 등이 대표적이다. 하지만 이러한 연구들은 밈스톡의 특수한 특성을 충분히 반영하지 못했으며, 소셜미디어 관심도가 항상 긍정적 영향을 미친다고 가정하였다.

본 연구는 이러한 가정에 도전하여 Reddit 관심도와 밈스톡 수익률 간의 **역상관관계(contrarian effect)**를 발견하고, 이를 행동재무학 이론으로 설명하고자 한다. 특히 과대관심(excessive attention) → 과대평가(overvaluation) → 가격조정(correction)의 메커니즘을 통해 왜 Reddit에서 화제가 될수록 오히려 수익률이 낮아지는지 규명한다.

Ⅱ. 본 론

2.1 연구방법

**데이터 수집**: Yahoo Finance API를 통해 2021년 1월부터 2023년 12월까지 GME, AMC, BB의 일간 주가 데이터를 수집하였다. Reddit API를 활용하여 r/wallstreetbets 커뮤니티의 게시글 및 댓글 데이터를 수집하였으며, 총 53,187개의 포스트와 5,409개의 일간 관측치를 분석에 사용하였다.

**주요 변수 정의**: Reddit Surprise는 예상 대비 초과 언급량을 나타내는 핵심 지표로 다음과 같이 정의된다:

RedditSurprise_{i,t} = (ActualMentions_{i,t} - ExpectedMentions_{i,t}) / ExpectedMentions_{i,t}

여기서 i는 종목, t는 시점을 나타낸다. 수익률은 일간 수익률로 정의하였다:

Returns_{i,t} = (P_{i,t} - P_{i,t-1}) / P_{i,t-1}

여기서 P_{i,t}는 종목 i의 t시점 종가이다.

**분석방법**: (1) Pearson 상관관계 분석을 통한 Reddit surprise와 수익률 간 관계 분석, (2) Event Study를 통한 극한 Reddit 스파이크 전후 수익률 패턴 분석, (3) 시간적 패턴 분석, (4) 53,187개 포스트에서 행동재무학적 편향 패턴 추출을 위한 텍스트 분석을 수행하였다.

2.2 분석결과

**핵심 발견**: 주요 밈스톡에서 일관된 음의 상관관계를 발견하였다. GME의 reddit_surprise vs returns_1d = -0.198, AMC = -0.178, BB = -0.165로 모든 종목에서 통계적으로 유의한 음의 상관관계가 나타났다. 이는 Reddit에서 관심이 급증할 때 오히려 다음날 수익률이 하락함을 의미한다.

**[여기에 그림 1 삽입: fast_correlation_analysis.png - 핵심 Contrarian Effect 상관관계]**

**Event Study 결과**: 극한 Reddit 스파이크(상위 5%) 113개 이벤트를 분석한 결과, 이벤트 당일 평균 음수 수익률(-0.005~-0.026), 다음날 부분적 회복하지만 여전히 저조, 2일 후 추가 조정이 지속되는 패턴을 관찰하였다.

**시간적 패턴**: 요일별 분석에서 화요일에 가장 강한 contrarian effect(GME: -0.374, AMC: -0.434), 금요일에 효과 역전(GME: +0.307, AMC: +0.187)이 나타났다. 계절별로는 겨울철에 최강 contrarian effect(-0.38~-0.40), 봄/여름에 효과가 약화되는 패턴을 확인하였다.

**[여기에 그림 2 삽입: advanced_analysis_comprehensive.png - 시간적 패턴 종합분석]**

**행동재무학적 메커니즘**: 53,187개 Reddit 포스트의 텍스트 분석을 통해 'Behavioral Paradox'를 발견하였다. 특히 Overconfidence Paradox에서 confidence_score vs next_day_returns가 -0.205(p=0.0002)로 확신 표현이 많을수록 다음날 수익률이 유의하게 하락하는 현상을 확인하였다. 또한 Price Anchoring Effect(price_mentions vs volatility: +0.472)와 FOMO Reversal(urgency_score vs reddit_surprise: -0.127) 효과도 관찰되었다.

**[여기에 그림 3 삽입: confidence_return_analysis.png - 확신 표현 vs 다음날 수익률 (-0.205 상관관계)]**

2.3 메커니즘 해석

발견된 contrarian effect는 다음과 같은 3단계 행동경제학적 메커니즘으로 설명된다: (1) 과대관심 단계에서 Reddit 관심 급증, (2) 과대평가 단계에서 비합리적 가격 형성, (3) 조정 단계에서 스마트머니의 차익거래가 발생한다. 

이는 Attention Theory[5]와 Overconfidence Bias[6]를 실증적으로 뒷받침하며, 효율적 시장가설(EMH)의 한계를 보여준다. 특히 텍스트 분석을 통해 발견한 behavioral paradox는 "진짜 확신할 때는 조용히 하고, 불안할 때 더 큰소리친다"는 시장 격언을 통계적으로 입증하는 결과이다.

**실무적 시사점**: (1) Reddit 관심도를 contrarian 지표로 활용 가능, (2) 화요일, 겨울철 등 효과가 강한 시점에 대한 리스크 관리 필요, (3) 텍스트 기반 감정 분석의 역설적 해석이 필요하다.

Ⅲ. 결 론

본 연구는 Reddit 데이터를 활용하여 밈스톡에서 나타나는 contrarian effect를 발견하고, 이를 행동재무학 이론으로 설명하였다. 주요 기여점은 다음과 같다: (1) 소셜미디어 관심도와 수익률의 일관된 음의 상관관계라는 새로운 현상 발견, (2) Event study부터 텍스트 분석까지 다각도 검증을 통한 종합적 분석, (3) Behavioral paradox를 통한 행동재무학 이론 확장, (4) 투자 전략 및 리스크 관리에 활용 가능한 실용적 인사이트 제공이다.

특히 확신 표현과 실제 성과 간의 역설적 관계(-0.205 상관관계)는 기존 소셜미디어-주식 연구와 차별화되는 핵심 발견으로, 개인투자자의 행동편향이 시장에 미치는 영향을 실증적으로 규명하였다. 향후 연구에서는 다른 소셜미디어 플랫폼으로의 확장, 실시간 예측 모델 개발, 규제 정책의 영향 등을 탐구할 예정이다.

ACKNOWLEDGMENT

본 연구는 서강대학교 연구비 지원을 받아 수행되었습니다.

참 고 문 헌

[1] Bollen, J., Mao, H., & Zeng, X. "Twitter mood predicts the stock market," Journal of computational science, vol. 2, no. 1, pp. 1-8, 2011.

[2] Ranco, G., Aleksovski, D., Caldarelli, G., Grčar, M., & Mozetič, I. "The effects of Twitter sentiment on stock price returns," PloS one, vol. 10, no. 9, e0138441, 2015.

[3] Mittal, A., & Goel, A. "Stock prediction using twitter sentiment analysis," Stanford University CS229 Machine Learning Final Project, pp. 1-5, 2012.

[4] Boehmer, E., Jones, C. M., Zhang, X., & Zhang, X. "Tracking retail investor activity," The Journal of Finance, vol. 76, no. 5, pp. 2249-2305, 2021.

[5] Barber, B. M., & Odean, T. "All that glitters: The effect of attention and news on the buying behavior of individual and institutional investors," The review of financial studies, vol. 21, no. 2, pp. 785-818, 2008.

[6] Daniel, K., Hirshleifer, D., & Subrahmanyam, A. "Investor psychology and security market under‐and overreactions," The journal of Finance, vol. 53, no. 6, pp. 1839-1885, 1998.
