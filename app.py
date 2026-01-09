import streamlit as st
import pandas as pd
import pymysql
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ------------------- DB 연결 -------------------
@st.cache_resource(show_spinner="DB 연결 중...")
def get_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='12341234',
        database='car_dashboard',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

conn = get_connection()

st.title("서울시 자동자 시장의 최신 트랜드")

menu = st.sidebar.radio("🏠 서울시 자동자 시장의 최신 트랜드",
    [
        "🚗 서울시 승용차의 현황",
        "🌿 서울시 친환경 승용차 현황",
        "⚡ 서울시 전기차 현황",
        "🔋 서울시 전기차 충전소 현황",
        "📹 서울시 CCTV의 현황",
        "🧐 내 자동차는, 친환경 자동차일까?"
    ]
)

# ------------------- 데이터 조회 함수 -------------------
def fetch_query(query):
    with conn.cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()
    return pd.DataFrame(results)

# ------------------- 서울시 승용차의 현황 -------------------
if menu == "🚗 서울시 승용차의 현황":

    try:
        # 규모별 데이터 로드
        df_size = fetch_query("SELECT 연도, 규모, 승용 FROM seoul_size_registration2 WHERE 시도='서울' ORDER BY 연도, 규모")

        if df_size.empty:
            st.warning("규모별 데이터를 불러오지 못했습니다.")
            st.stop()

        # 연도별 총합 계산
        total_by_year = df_size.groupby('연도')['승용'].sum().reset_index()
        latest_year = total_by_year['연도'].max()
        latest_total = total_by_year[total_by_year['연도'] == latest_year]['승용'].values[0]

        # 변화량 계산
        total_by_year['변화량'] = total_by_year['승용'].diff()

        # 최신 연도 요약
        st.subheader("🚗 서울시 승용차의 등록 현황")
        col1, col2 = st.columns(2)
        with col1:
            delta = total_by_year[total_by_year['연도'] == latest_year]['변화량'].values[0] if len(total_by_year) > 1 else 0
            st.metric("총 등록 대수", f"{int(latest_total):,}", f"{int(delta):+,}대")
        with col2:
            st.metric("데이터 기준 연도", int(latest_year))

        # 1. 연도별 전체 등록 대수 추이 막대 그래프
        fig_bar, ax_bar = plt.subplots(figsize=(11, 6))

        years = total_by_year['연도'].astype(int).tolist()  # [2022, 2023, 2024, 2025]
        values = total_by_year['승용'].values
        changes = total_by_year['변화량'].fillna(0).values

        # 증가/감소 색상 구분
        colors = ['#4CAF50' if x >= 0 else '#F44336' for x in changes]

        # 막대 너비 조정하여 연도 중앙에 맞게
        width = 0.6
        bars = ax_bar.bar(years, values, color=colors, edgecolor='black', width=width)

        # 막대 위에 숫자 + 변화량 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            change = int(changes[i]) if i > 0 else 0
            change_str = f"{change:+,}대" if i > 0 else "기준"
            ax_bar.text(bar.get_x() + bar.get_width()/2., height + 500,
                        f'{int(height):,}\n{change_str}',
                        ha='center', va='bottom', fontweight='bold', fontsize=11, color='black')

        ax_bar.set_title('서울시 승용차 등록 대수', fontsize=16, pad=20)
        ax_bar.set_xlabel('연도', fontsize=12)
        ax_bar.set_ylabel('등록 대수', fontsize=12)
        ax_bar.grid(alpha=0.3, axis='y', linestyle='--')

        # x축 연도 정확히 정수로 중앙 배치
        ax_bar.set_xticks(years)
        ax_bar.set_xticklabels(years)

        # y축 범위 고정
        ax_bar.set_ylim(2760000, 2780000)
        ax_bar.set_yticks(np.arange(2760000, 2780001, 20000))
        ax_bar.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

        st.pyplot(fig_bar)

        # 2. 규모별 파이차트 (사진처럼 안쪽에 "규모 + 퍼센트" 표시)
        st.subheader("서울 승용차 규모별 구성 비율 (2025)")

        # 2025년 데이터만 필터링
        df_2025 = df_size[df_size['연도'] == 2025]
        sizes_raw = df_2025.groupby('규모')['승용'].sum()
        total = sizes_raw.sum()

        # 3% 미만은 "소형"으로 묶기 (기타 → 소형으로 변경)
        threshold = 0.03
        small_sizes = sizes_raw[sizes_raw / total < threshold]
        large_sizes = sizes_raw[sizes_raw / total >= threshold]

        if not small_sizes.empty:
            sizes = pd.concat([large_sizes, pd.Series({'소형': small_sizes.sum()})])
        else:
            sizes = large_sizes.copy()

        sizes = sizes.sort_values(ascending=False)
        percentages = (sizes / total * 100).round(1)

        # 색상 팔레트
        colors = plt.cm.Pastel1(range(len(sizes)))

        # 파이차트 그리기
        fig_pie, ax_pie = plt.subplots(figsize=(10, 10))
        wedges, texts, autotexts = ax_pie.pie(
            sizes,
            labels=None,
            autopct='',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 14}
        )

        # 각 조각 안에 "규모 + 퍼센트" 표시
        for i, wedge in enumerate(wedges):
            ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
            y = 0.6 * np.sin(np.deg2rad(ang))
            x = 0.6 * np.cos(np.deg2rad(ang))

            # 기본 라벨: 규모 + 퍼센트
            label = f"{sizes.index[i]}\n{percentages[i]}%"

            # 1위와 2위는 굵게 + "1위:" / "2위:" 추가
            if i == 0:  # 1위
                label = f"1위: {sizes.index[i]} ({percentages[i]}%)"
                fontweight = 'extra bold'
                fontsize = 18
            elif i == 1:  # 2위
                label = f"2위: {sizes.index[i]} ({percentages[i]}%)"
                fontweight = 'bold'
                fontsize = 16
            else:
                fontweight = 'bold'
                fontsize = 15

            ax_pie.text(x, y, label,
                        ha='center', va='center',
                        fontweight=fontweight, fontsize=fontsize,
                        color='black')

        ax_pie.set_title('규모별 구성 비율 (2025년)', fontsize=18, pad=30)

        st.pyplot(fig_pie)

        # 3. 상세 테이블
        st.markdown("---")
        st.subheader("📋 2022~2025년 규모별 등록 대수 상세")
        pivot_table = df_size.pivot(index='연도', columns='규모', values='승용').fillna(0).astype(int)
        pivot_table['합계'] = pivot_table.sum(axis=1)
        pivot_table = pivot_table.sort_index(ascending=False)

        styled_table = pivot_table.style\
            .format('{:,}')\
            .set_properties(**{'text-align': 'center', 'font-size': '14px'})\
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('font-weight', 'bold')]},
            ])\
            .bar(subset=['합계'], color='#a8e6cf')

        st.dataframe(styled_table, use_container_width=True, hide_index=True)

        # 결론
        st.info("""
**서울시 자동차 시장의 최신 트랜드 Insight**

ㅇ 서울시 승용차 등록 현황 : 23년부터 매년 소폭 감소  
ㅇ 서울시 승용차 사이즈 현황 : 중형(57.8%) > 대형(34.4%) > 소형 > 경형 순으로 나타남

▷ 서울시의 승용차 트렌드를 살펴보면 차량등록은 매년 소폭 감소 추세로 보이고  
   차량 사이즈는 대부분 중형 이상 (중형+대형 = 92.2%) 의 사이즈를 선호함
""")

        st.caption("데이터 출처: 국토교통부 승용차 등록 통계 (2025년 포함 최신)")

    except Exception as e:
        st.error(f"홈 화면 로드 중 오류: {e}")
        
# ------------------- 서울시 친환경 승용차 현황 -------------------
elif menu == "🌿 서울시 친환경 승용차 현황":
    st.header("🌿 서울 친환경 자동차 등록 현황")
    st.markdown("**2022~2025년 전기차 · 하이브리드 · 수소차 보급 추이**")

    try:
        df = fetch_query("SELECT * FROM seoul_fuel_registration WHERE 시도='서울' ORDER BY 연도")

        if df.empty:
            st.warning("데이터를 불러오지 못했습니다.")
            st.stop()

        # 핵심 메트릭
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None
        st.subheader("서울 친환경 승용차 등록 현황 \n ***(2025년 11월 기준)***")
        col1, col2, col3 = st.columns(3)
        with col1:
            delta_ev = int(latest['전기_승용'] - prev['전기_승용']) if prev is not None else 0
            st.metric("전기차", f"{int(latest['전기_승용']):,}", f"+{delta_ev:,}대")
        with col2:
            delta_hybrid = int(latest['하이브리드_승용'] - prev['하이브리드_승용']) if prev is not None else 0
            st.metric("하이브리드", f"{int(latest['하이브리드_승용']):,}", f"+{delta_hybrid:,}대")
        with col3:
            delta_h2 = int(latest['수소_승용'] - prev['수소_승용']) if prev is not None else 0
            st.metric("수소차", f"{int(latest['수소_승용']):,}", f"+{delta_h2:,}대")

        # 누적 막대 그래프
        fig, ax = plt.subplots(figsize=(12, 7))

        years = df['연도'].astype(int).tolist()
        ev = df['전기_승용'].values
        hybrid = df['하이브리드_승용'].values
        h2 = df['수소_승용'].values

        width = 0.6

        bar1 = ax.bar(years, ev, width=width, label='전기차', color='#1f77b4')
        bar2 = ax.bar(years, hybrid, width=width, bottom=ev, label='하이브리드', color='#ff7f0e')
        bar3 = ax.bar(years, h2, width=width, bottom=ev + hybrid, label='수소차', color='#2ca02c')

        ax.set_title('친환경 승용차 등록 추이 \n (2022년 ~ 2025년 최근 4년)', fontsize=16, pad=20)
        ax.set_xlabel('연도', fontsize=12)
        ax.set_ylabel('누적 등록 대수', fontsize=12)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(alpha=0.3, axis='y')

        ax.set_xticks(years)
        ax.set_xticklabels(years)

        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

        for i, year in enumerate(years):
            total = ev[i] + hybrid[i] + h2[i]
            ax.text(year, total + 5000, f'총 {int(total):,}대', ha='center', va='bottom', fontweight='bold', fontsize=11)
            ax.text(year, ev[i]/2, f'{int(ev[i]):,}', ha='center', va='center', color='white', fontweight='bold')
            ax.text(year, ev[i] + hybrid[i]/2, f'{int(hybrid[i]):,}', ha='center', va='center', color='white', fontweight='bold')
            if h2[i] > 0:
                ax.text(year, ev[i] + hybrid[i] + h2[i]/2, f'{int(h2[i]):,}', ha='center', va='center', color='white', fontweight='bold')

        st.pyplot(fig)

        # 테이블
        st.markdown("**※ 서울시 친환경 승용차의 상세 데이터**")
        display_df = df[['연도', '전기_승용', '하이브리드_승용', '수소_승용']].rename(columns={
            '전기_승용': '전기차', '하이브리드_승용': '하이브리드', '수소_승용': '수소차'
        })
        display_df['연도'] = display_df['연도'].astype(int)

        st.dataframe(
            display_df.style.format({
                '연도': '{:d}',
                '전기차': '{:,}',
                '하이브리드': '{:,}',
                '수소차': '{:,}'
            }),
            use_container_width=True,
            hide_index=True
        )

        # 2025년 국내 판매 TOP3 브랜드 분석 (막대그래프)
        st.subheader("2025년 국내 판매 TOP3 브랜드 분석 (국산 vs 수입)")

        # car_sales 테이블에서 2025년 데이터 가져오기
        sales_df = fetch_query("""
            SELECT * FROM car_sales 
            WHERE 연도 = 2025 
            ORDER BY FIELD(구분, '국산', '수입'), 순위
        """)

        if not sales_df.empty:
            sales_df['친환경_비중_%'] = (sales_df['친환경'] / sales_df['전체'] * 100).round(1)

            fig_sales, ax_sales = plt.subplots(figsize=(14, 8))

            x = np.arange(len(sales_df))  # 0~5

            bar_width = 0.4

            # 전체 판매량
            ax_sales.bar(x, sales_df['전체'], width=bar_width, 
                         label='전체 판매량', color='#81d4fa', alpha=0.9)  # 밝은 하늘색

            # 친환경 판매량
            ax_sales.bar(x, sales_df['친환경'], width=bar_width, 
                         label='친환경 판매량', color='#66bb6a')  # 밝은 초록

            # 비중 %
            for i, row in sales_df.iterrows():
                ax_sales.text(i, row['전체'] + 10000, f'{row["친환경_비중_%"]}%', 
                              ha='center', va='bottom', fontsize=11, fontweight='bold',
                              color='black' if row['친환경_비중_%'] < 50 else 'white')

            # x축
            ax_sales.set_xticks(x)
            ax_sales.set_xticklabels(sales_df['브랜드'], fontsize=11, rotation=45, ha='right')

            # 국산/수입 구분선
            ax_sales.axvline(x=2.5, color='gray', linestyle='--', linewidth=1.5)
            ax_sales.text(1, max(sales_df['전체']) * 1.05, '국산', ha='center', fontsize=13, fontweight='bold', color='blue')
            ax_sales.text(4.5, max(sales_df['전체']) * 1.05, '수입', ha='center', fontsize=13, fontweight='bold', color='darkred')

            ax_sales.set_title('2025년 국내 자동차 판매 TOP3 브랜드\n전체 판매량 vs 친환경 비중 (국산·수입 구분)', fontsize=16, pad=25)
            ax_sales.set_xlabel('브랜드', fontsize=12)
            ax_sales.set_ylabel('판매량 (대)', fontsize=12)
            ax_sales.grid(axis='y', alpha=0.3, linestyle='--')
            ax_sales.legend(loc='upper right')

            ax_sales.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

            plt.tight_layout()
            st.pyplot(fig_sales)

        else:
            st.warning("car_sales 테이블에 2025년 데이터가 없습니다. DB를 확인해주세요.")

        # 인사이트
        st.info("""
        **● 서울시 자동차 시장의 최신 트랜드 Insight ●**

        ㅇ 서울시 친환경 승용차 현황 : 22년부터 매년 증가  
        ㅇ 서울시 친환경 승용차 비중 : 하이브리드 > 전기차 > 수소차 순


        ▷ 특히,  친환경 승용차 중 하이브리드는 전기차 시대로 변화하는 과도기적 산물로
             현재는 가장 많은 비중을 차지 하고 있고 전기차도 매년 큰 폭으로 증가 추세로 나타남 
        """)

    except Exception as e:
        st.error(f"오류: {e}")
        
# ------------------- 서울시 전기차 현황 -------------------
elif menu == "⚡ 서울시 전기차 현황":
    
    st.header("⚡ 서울 전기차 등록 현황")

    tab1, tab2 = st.tabs(["📊 현재 추이 분석", "🚀 미래 전기차 비중 예측 (2026~2030)"])

    with tab1:

        try:
            total_df = fetch_query("""
                SELECT 연도, SUM(승용) AS 총_승용차
                FROM seoul_size_registration
                GROUP BY 연도
                ORDER BY 연도
            """)
            ev_df = fetch_query("SELECT 연도, 전기_승용 AS 전기차 FROM seoul_fuel_registration ORDER BY 연도")
            df = pd.merge(total_df, ev_df, on='연도')
            df = df.sort_values('연도').reset_index(drop=True)
            df['총_승용차'] = df['총_승용차'].astype(int)
            df['전기차'] = df['전기차'].astype(int)

            col1, col2, col3 = st.columns(3)
            latest_year = df['연도'].iloc[-1]
            latest_ev_ratio = (df['전기차'].iloc[-1] / df['총_승용차'].iloc[-1] * 100)
            with col1:
                st.metric("📊 전체 자동차 vs 전기차 추이 비교", f"{latest_ev_ratio:.2f}%",
                          delta=f"{latest_ev_ratio - (df['전기차'].iloc[-2] / df['총_승용차'].iloc[-2] * 100):.2f}%p 증가")
            with col2:
                st.metric("⬆️ 전기차 증가량", f"{df['전기차'].iloc[-1] - df['전기차'].iloc[-2]:,}대", delta="2024→2025년")
            with col3:
                st.metric("📊 전체 자동차 변화", f"{df['총_승용차'].iloc[-1] - df['총_승용차'].iloc[-2]:+,}대", delta="2024→2025년")

            # 줄바꿈 없이 바로 그래프 붙이기
            fig, ax1 = plt.subplots(figsize=(12, 6))

            # 연도를 명시적으로 정수형으로 변환
            df['연도'] = df['연도'].astype(int)

            ax1.set_xlabel('연도', fontsize=12)
            ax1.set_ylabel('전체 자동차 대수', color='gray', fontsize=12)

            # 전체 자동차 (회색 선)
            ax1.plot(df['연도'], df['총_승용차'], 
                 marker='o', linewidth=4, markersize=10, 
                 color='gray', label='전체 자동차')

            ax1.tick_params(axis='y', labelcolor='gray')
            ax1.grid(alpha=0.3)

            # 오른쪽 축 - 전기차
            ax2 = ax1.twinx()
            ax2.set_ylabel('전기차 대수', color='green', fontsize=12)

            ax2.plot(df['연도'], df['전기차'], 
                marker='s', linewidth=5, markersize=12, 
                color='green', label='전기차')

            ax2.tick_params(axis='y', labelcolor='green')

            # 제목
            ax1.set_title('서울 전체 자동차는 거의 그대로, 전기차는 꾸준히 증가!', 
                  fontsize=16, pad=20)

            # 범례 합치기
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

            # y축 천단위 콤마
            ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
            ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

            # x축을 1년 단위 정수로 강제 지정
            ax1.set_xticks(df['연도'])                
            ax1.set_xticklabels(df['연도'])           

            st.pyplot(fig)
            
            st.subheader("📅 연도별 등록 대수 요약")
            display_df = df.copy()
            display_df['전기차 비율 (%)'] = (display_df['전기차'] / display_df['총_승용차'] * 100).round(2)
            st.dataframe(
                display_df.rename(columns={
                    '연도': '연도', '총_승용차': '전체 자동차', '전기차': '전기차', '전기차 비율 (%)': '전기차 비율 (%)'
                }).style.format({'전체 자동차': '{:,}', '전기차': '{:,}', '전기차 비율 (%)': '{:.2f}%'}),
                use_container_width=True, hide_index=True
            )

            st.info("""
            **● 서울시 자동차 시장의 최신 트랜드 Insight ●**  
            서울은 전체 자동차 수가 거의 변하지 않거나 조금 줄고 있는데,  
            **전기차만 꾸준히 늘고 있어요!**
            
            ㅇ 사람들이 새 차를 살 때 **전기차를 더 많이 선택**하고 있다는 뜻  
            ㅇ 전체 시장이 줄어도 전기차가 그 빈자리를 채우고 있음  
            ㅇ 앞으로 전기차 비율이 점점 더 높아질 가능성이 큽니다!

            🌱 전기차가 서울의 자동차 시장을 새롭게 바꾸고 있어요!
            """)
            st.caption("데이터 출처: 국토교통부 승용차 등록 통계")

        except Exception as e:
            st.error(f"현재 추이 분석 중 오류: {e}")
            
# ------------------- 전기차 비중 예측 -------------------
    with tab2:
            st.markdown("**2023~2025년 월별 데이터 기반 선형회귀 예측**")

            try:
                query = """
                SELECT ym AS 연월, total_cars AS 전체, ev_cars AS 전기차, ev_ratio AS 비중
                FROM seoul_ev_ratio_monthly
                ORDER BY ym ASC
                """
                df = fetch_query(query)
                if df.empty:
                    st.error("DB에서 데이터를 불러오지 못했습니다. 테이블(seoul_ev_ratio_monthly)을 확인하세요.")
                    st.stop()

                df['연월'] = df['연월'].astype(int)
                X = df[['전기차']].values
                y = df['비중'].values * 100

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)
                r2_test = r2_score(y_test, y_pred_test)
                mae_test = mean_absolute_error(y_test, y_pred_test)

                st.subheader("모델 성능 (LinearRegression)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² (테스트)", f"{r2_test:.4f}")
                with col2:
                    st.metric("MAE (테스트)", f"{mae_test:.2f}")
                with col3:
                    st.metric("훈련 데이터 크기", f"{len(X_train)} / {len(X)}")

                st.subheader("시나리오 설정")
                annual_ev_increase = st.slider("연간 전기차 등록 증가량 (대)", 10000, 60000, 25000, 1000)

                latest_row = df.loc[df['연월'].idxmax()]
                latest_ev = latest_row['전기차']
                latest_total = latest_row['전체']
                latest_ratio = latest_row['비중'] * 100

                future_years = np.arange(2026, 2031)
                future_ev = [latest_ev + annual_ev_increase * (yr - 2025) for yr in future_years]
                future_ratio = model.predict(np.array(future_ev).reshape(-1, 1))

                st.subheader("미래 예측 결과 (연도별)")
                pred_df = pd.DataFrame({
                    '연도': future_years,
                    '예상 전기차 등록 (대)': [f"{int(ev):,}" for ev in future_ev],
                    '예상 전기차 비중 (%)': [f"{r:.2f}" for r in future_ratio]
                })
                st.dataframe(pred_df, use_container_width=True, hide_index=True)

                st.subheader("그래프 (실제 + 예측)")
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.scatter(df['전기차'], y, color='blue', s=60, alpha=0.7, label='실제 데이터 (2023~2025)')
                x_min = df['전기차'].min()
                x_max = max(future_ev) + 20000
                x_range = np.linspace(x_min, x_max, 200)
                y_range = model.predict(x_range.reshape(-1, 1))
                ax.plot(x_range, y_range, color='red', linewidth=2.5, label='선형회귀 모델')
                ax.scatter(future_ev, future_ratio, color='green', s=150, marker='*', label='미래 예측 (2026~2030)')
                ax.set_title('서울 전기차 등록대수 vs 전기차 비중 (2023~2025 기반 선형 예측)', fontsize=14)
                ax.set_xlabel('전기차 등록대수 (대)', fontsize=12)
                ax.set_ylabel('전기차 비중 (%)', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=11)
                ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
                ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.2f}'))
                st.pyplot(fig)

                st.success(f"""
                📊 **2025년 11월 기준**  
                • 전기차 등록: {latest_ev:,}대  
                • 전체 승용차: {latest_total:,}대  
                • 전기차 비중: {latest_ratio:.2f}%  
                """)
                
                st.info("""
            **● 서울시 자동차 시장의 최신 트랜드 Insight ●**

ㅇ 서울시 전기차 비중 : 3.12% (2025년  11월 기준)  
ㅇ 모델 성능은 R2 =0.99로  매우 높게 나오며, 과대적합의 우려가 있어 그리드 서치 확인  
ㅇ  실제 데이터 : 매년  증가  
ㅇ 미래 예측 데이터 : 향후 5년간 (26~30년) 증가 추세로 보임

            """)
                
                st.markdown("전기차가 증가한다면, 서울시 충전소의 현황은 어떨까요?")
                st.markdown("왼쪽 메뉴에서 선택해 주세요~!")

            except Exception as e:
                st.error(f"미래 비중 예측 오류: {str(e)}")
            
# ------------------- 서울시 전기차 충전소 현황 -------------------        
elif menu == "🔋 서울시 전기차 충전소 현황":
    st.header("🔋 서울 전기차 등록 vs 충전기 인프라 분석")
    st.markdown("**2022~2024년 누적 데이터 기반** (충전기: 환경부, 전기차: 국토부 승용 기준)")

    try:
        # DB에서 전기차 + 충전기 데이터 한번에 가져오기
        query = """
        SELECT 
            f.연도,
            f.전기_승용 AS 누적_전기차,
            COALESCE(c.누적_충전기, 0) AS 누적_충전기
        FROM seoul_fuel_registration2024 f
        LEFT JOIN seoul_chargers c ON f.연도 = c.year
        WHERE f.시도 = '서울'
        ORDER BY f.연도
        """
        df = fetch_query(query)
        df_ev = fetch_query("""
            SELECT 연도, 전기_승용 AS 전기차_등록 
            FROM seoul_fuel_registration2024
            WHERE 시도 = '서울' 
            ORDER BY 연도
        """)

        if df_ev.empty:
            st.error("데이터를 불러오지 못했습니다. DB를 확인해주세요.")
            st.stop()

        if len(df_ev) < 3:
            st.warning("시계열 분석은 최소 3년 이상 데이터가 필요합니다.")
            st.stop()

        df_ts = df_ev.set_index('연도')['전기차_등록']

        if df.empty:
            st.warning("DB에서 데이터를 가져오지 못했습니다. 테이블과 데이터를 확인해주세요.")
            st.stop()

        df['연도'] = df['연도'].astype(int)

        # 선형 회귀 모델
        model = LinearRegression()
        X = df[['누적_충전기']]
        y = df['누적_전기차']
        model.fit(X, y)
        slope = model.coef_[0]
        r2 = model.score(X, y)

        # 충전기 1기당 전기차 비율
        df['충전기1기당_전기차'] = df['누적_전기차'] / df['누적_충전기']

        col1, col2 = st.columns(2)

        with col1:
            st.metric("충전기 1기 증가 시", f"+{slope:.3f}대", "전기차 등록 증가 (평균)")
            st.metric("현재 평균 비율", f"{df['충전기1기당_전기차'].iloc[-1]:.2f}대", "충전기 1기당 전기차")

        with col2:
            st.metric("모델 설명력 (R²)", f"{r2:.6f}")
            last_year = df['연도'].iloc[-1]
            st.metric(f"{last_year}년 누적 충전기", f"{df['누적_충전기'].iloc[-1]:,}기")

        # 그래프 1: 누적 추이
        st.subheader("누적 추이")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df['연도'], df['누적_충전기'], marker='o', label='누적 충전기', linewidth=3, color='blue')
        ax1.plot(df['연도'], df['누적_전기차'], marker='s', label='누적 전기차', linewidth=3, color='green')
        ax1.set_title('서울 누적 충전기 vs 전기차 등록 추이')
        ax1.set_ylabel('대수')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax1.set_xticks(df['연도'])
        ax1.set_xticklabels(df['연도'])
        
        st.pyplot(fig1)

        # 그래프 2: 비율 추이
        st.subheader("충전기 1기당 전기차 대수 추이")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(df['연도'], df['충전기1기당_전기차'], marker='D', color='purple', linewidth=3, markersize=10)
        ax2.set_title('충전기 1기당 지원 가능한 전기차 대수 변화')
        ax2.set_ylabel('전기차 대수 / 충전기 1기')
        ax2.grid(alpha=0.3)
        for i, row in df.iterrows():
            ax2.text(row['연도'], row['충전기1기당_전기차'] + 0.01, f"{row['충전기1기당_전기차']:.2f}", 
                     ha='center', fontweight='bold')
        
        ax2.set_xticks(df['연도'])
        ax2.set_xticklabels(df['연도'])
        
        st.pyplot(fig2)

        # 회귀 산점도
        st.subheader("상관 분석 및 회귀 모델")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.scatter(df['누적_충전기'], df['누적_전기차'], s=150, color='darkblue', zorder=5)
        x_line = np.array([df['누적_충전기'].min(), df['누적_충전기'].max()])
        y_line = model.predict(x_line.reshape(-1, 1))
        ax3.plot(x_line, y_line, color='red', linewidth=3, label=f'회귀선 (기울기={slope:.3f})')
        for i, row in df.iterrows():
            ax3.text(row['누적_충전기'] + 600, row['누적_전기차'], str(row['연도']), fontsize=12, fontweight='bold')
        ax3.set_xlabel('누적 충전기 대수')
        ax3.set_ylabel('누적 전기차 등록 대수')
        ax3.set_title(f'누적 상관 분석 (R² = {r2:.6f})')
        ax3.legend()
        ax3.grid(alpha=0.3)
        st.pyplot(fig3)

        # 미래 예측
        st.subheader("🔮 2025~2027년 예측 (현재 추세 유지 가정)")
        
        last_year = df['연도'].iloc[-1]
        last_charger = df['누적_충전기'].iloc[-1]
        if len(df) >= 2:
            annual_new = last_charger - df['누적_충전기'].iloc[-2]
        else:
            annual_new = 11642  # 데이터가 1년뿐이면 기존 값 fallback

        st.caption(f"가정: 매년 충전기 약 {annual_new:,}기 증가 (최근 연간 증가량 기준)")

        pred_years = [last_year + 1, last_year + 2, last_year + 3]
        pred_list = []
        for y in pred_years:
            years_ahead = y - last_year
            pred_charger = last_charger + years_ahead * annual_new
            pred_ev = model.predict([[pred_charger]])[0]
            ratio = pred_ev / pred_charger
            pred_list.append({
                '연도': y,
                '예측 누적 충전기': f"{int(pred_charger):,}",
                '예측 누적 전기차': f"{int(pred_ev):,}",
                '예측 비율 (대/기)': f"{ratio:.2f}"
            })

        pred_df = pd.DataFrame(pred_list)
        st.table(pred_df)

        st.success(f"분석 완료! 충전 인프라가 전기차 보급을 잘 뒷받침하고 있으며, "
                   f"현재 추세로는 {last_year + 3}년 약 {pred_list[2]['예측 비율 (대/기)']}대/기 수준 예상됩니다!")
        
        st.subheader("ARIMA 시계열 예측")
        if st.button("🔮 ARIMA 모델 학습 및 예측 실행"):
            try:
                import pmdarima as pm  # 여기서 import

                with st.spinner("auto_arima 모델 학습 중..."):
                    model_fit = pm.auto_arima(
                    y=df_ts,
                    start_p=0, max_p=1,
                    start_q=0, max_q=1,
                    d=1,
                    seasonal=False,
                    trend='t',
                    stepwise=True,
                    trace=True
                )

                st.success(f"모델 학습 완료! 최적 모델: {model_fit.order}")

                # 예측
                steps = 5
                forecast = model_fit.predict(n_periods=steps, return_conf_int=True)
                
                pred_years = list(range(df_ts.index[-1] + 1, df_ts.index[-1] + steps + 1))

                # 예측 그래프
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(df_ts.index, df_ts.values, 'o-', label='실제 데이터', color='blue', linewidth=4)
                ax2.plot(pred_years, forecast[0], 's--', label='예측', color='red', linewidth=4)
                ax2.fill_between(pred_years, forecast[1][:,0], forecast[1][:,1], color='red', alpha=0.2, label='95% 신뢰구간')
                ax2.set_title('서울 전기차 등록 수 ARIMA 예측')
                ax2.set_xlabel('연도')
                ax2.set_ylabel('등록 대수')
                ax2.legend()
                ax2.grid(alpha=0.3)
                ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
                
                all_years = list(df_ts.index) + pred_years
                ax2.set_xticks(all_years)
                ax2.set_xticklabels(all_years)
                
                st.pyplot(fig2)

                # 예측 결과 테이블
                result_df = pd.DataFrame({
                    '연도': pred_years,
                    '예측 등록 대수': forecast[0].round(0).astype(int),
                    '신뢰구간 하한': forecast[1][:,0].round(0).astype(int),
                    '신뢰구간 상한': forecast[1][:,1].round(0).astype(int)
                })

                # 연도를 정수형으로 강제 변환 (쉼표 자동 제거)
                result_df['연도'] = result_df['연도'].astype(int)

                # 테이블 출력 (연도 포맷을 {:d}로 지정)
                st.table(result_df.style.format({
                    '연도': '{:d}',                     # 쉼표 없이 정수
                    '예측 등록 대수': '{:,}',
                    '신뢰구간 하한': '{:,}',
                    '신뢰구간 상한': '{:,}'
                }))

            except ImportError:
                st.error("pmdarima 라이브러리가 없습니다. 터미널에서 `pip install pmdarima` 실행 후 재시작하세요.")
            except Exception as e:
                st.error(f"ARIMA 학습 중 오류: {e}")

        else:
            st.info("↑ 버튼을 클릭하면 ARIMA 모델이 학습되고 예측 결과가 표시됩니다.")

    except Exception as e:
        st.error(f"페이지 로드 오류: {e}")
        st.info("코드에 구문 오류가 있는지 확인하세요.")
        

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
        st.info("쿼리 또는 테이블(seoul_fuel_registration, seoul_chargers)을 확인해주세요.")
        

# ------------------- 서울시 CCTV의 현황 -------------------
elif menu == "📹 서울시 CCTV의 현황":
    st.header("📹 서울 자치구 CCTV vs 교통사고 분석 (2025)")

    try:
        query = """
        SELECT 
            year AS 연도,
            gu AS 자치구,
            cctv AS CCTV,
            accidents AS 사고건수
        FROM seoul_cctv_accident
        WHERE year = 2025
        ORDER BY gu
        """
        df = fetch_query(query)

        if df.empty:
            st.error("DB에서 데이터를 불러오지 못했습니다. 테이블(seoul_cctv_accident)을 확인하세요.")
            st.stop()

        # 데이터 준비
        X = df[['사고건수']].values
        y = df['CCTV'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )

        # 그래프: 산점도
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['사고건수'], df['CCTV'], color='darkorange', s=100, alpha=0.8, label='실제 데이터 (자치구)')


        ax.set_title(f'서울 자치구별 사고건수 vs CCTV 개수 (2025년)', fontsize=14)
        ax.set_xlabel('사고건수')
        ax.set_ylabel('CCTV 개수')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.subheader("서울 자치구별 CCTV vs 교통사고 전체 추이 (2025년)")


        # 자치구 순서 강제 정렬
        df_sorted = df.sort_values('자치구').reset_index(drop=True)

        # 만약 DB에 gu 칼럼이 '강남구'처럼 '구'까지 포함되어 있다면 그대로 씁니다.
        # (필요 시 아래처럼 .str.replace('구', '') 등으로 처리할 수도 있지만 보통 그대로 씁니다)

        fig, ax = plt.subplots(figsize=(14, 7))

        # 사고건수 (주황색, 위쪽)
        ax.plot(df_sorted['자치구'], df_sorted['사고건수'],
                color='#FF5722', linewidth=2.8, marker='o', markersize=6,
                label='사고건수')

        # CCTV (파란색, 아래쪽)
        ax.plot(df_sorted['자치구'], df_sorted['CCTV'],
                color='#1976D2', linewidth=2.8, marker='o', markersize=6,
                label='cctv갯수')

        ax.set_title("차트 제목", fontsize=16, pad=20)
        ax.set_ylabel("대수", fontsize=12)
        ax.set_ylim(0, 4100)
        ax.set_yticks(range(0, 4101, 500))

        # x축 라벨 45도 회전 + 작게
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted['자치구'], rotation=45, ha='right', fontsize=9.5)

        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=11)

        # 사진처럼 오른쪽 위 화살표
        ax.text(0.98, 0.98, '→', transform=ax.transAxes,
                fontsize=28, fontweight='bold', color='blue',
                va='top', ha='right')

        # y축 천단위 콤마
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

        plt.tight_layout()
        st.pyplot(fig)

        # 전체 데이터 테이블
        st.subheader("2025년 자치구별 데이터")

        # 기본: 상위 5개 자치구만 표시
        df_display = df[['자치구', 'CCTV', '사고건수']].copy()

        # 사고건수 기준 내림차순 정렬 (사고 많은 구부터)
        df_display = df_display.sort_values('사고건수', ascending=False).reset_index(drop=True)

        # 상위 5개만 기본 표시
        st.dataframe(
            df_display.head(5).style.format({'CCTV': '{:,}', '사고건수': '{:,}'}),
            use_container_width=True,
            hide_index=True
        )

        # 전체 보기 버튼
        if st.button("📋 전체 25개 자치구 데이터 보기"):
            st.dataframe(
                df_display.style.format({'CCTV': '{:,}', '사고건수': '{:,}'}),
                use_container_width=True,
                hide_index=True
            )
            st.info("위는 사고건수 기준 내림차순 정렬된 전체 데이터입니다.")
        
        st.info("""
        **● 서울시 자동차 시장의 최신 트랜드 Insight ●**

ㅇ 서울시 CCTV와 교통사고 상관 : 상관계수가 0.6 이상으로 양의 상관  
ㅇ 훈련데이터 작지만 최적의 데이터를 찾아 미래 예측 모델 생성  

▷  CCTV와 교통사고 그래프는 사고 건수가 높을 수록 CCTV의 설치가 증가하는것으로 나타남

        """)

    except Exception as e:
        st.error(f"페이지 실행 중 오류: {str(e)}")
        st.info("DB 테이블(seoul_cctv_accident) 또는 쿼리를 확인해주세요.")


# ------------------- 전기차 분류 모델 -------------------
elif menu == "🧐 내 자동차는, 친환경 자동차일까?":

    try:
        # 1. 두 테이블 데이터 로드
        df_spec = fetch_query("SELECT displacement AS engine_cc, fuel_efficiency, vehicle_type FROM vehicle_classification")
        df_model = fetch_query("SELECT power_type, model_name FROM car_model_by_power_type")

        if df_spec.empty or df_model.empty:
            st.error("필요한 테이블 데이터가 없습니다. DB 확인해주세요.")
            st.stop()

        # 2. 학습 데이터 준비 (배기량 + 연비 → 동력유형)
        df_spec['engine_cc'] = pd.to_numeric(df_spec['engine_cc'], errors='coerce')
        df_spec['fuel_efficiency'] = pd.to_numeric(df_spec['fuel_efficiency'], errors='coerce')
        df_spec = df_spec.dropna()

        X = df_spec[['engine_cc', 'fuel_efficiency']]
        y = df_spec['vehicle_type']

        # Label Encoding
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_  # ['전기차', '일반', '하이브리드']

        # 훈련/테스트 분리 + 스케일링
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # KNN 모델 학습 (캐싱으로 속도 향상)
        @st.cache_resource(show_spinner="KNN 모델 학습 중...")
        def train_knn():
            knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
            knn.fit(X_train_scaled, y_train)
            return knn

        knn = train_knn()

        # 3. 사용자 입력: 차종 선택
        st.subheader("🧐 내 자동차를 선택해서 친환경 자동차인지 확인해보세요!")
        all_models = sorted(df_model['model_name'].unique())
        selected_model = st.selectbox("차종을 선택하세요!", all_models)

        if st.button("동력유형 예측하기"):
            # 선택한 차종의 동력유형 (참고용 - 실제 예측과 비교)
            true_power = df_model[df_model['model_name'] == selected_model]['power_type'].iloc[0]

            # 해당 차종이 속한 동력유형의 평균 배기량/연비 계산
            avg_spec = df_spec[df_spec['vehicle_type'] == true_power][['engine_cc', 'fuel_efficiency']].mean()

            # 가상의 입력 점 생성 (평균값 사용 → 실제 모델은 평균 기반 예측)
            new_point = np.array([[avg_spec['engine_cc'], avg_spec['fuel_efficiency']]])

            # 스케일링 및 예측
            new_point_scaled = scaler.transform(new_point)
            pred_encoded = knn.predict(new_point_scaled)[0]
            pred_label = le.inverse_transform([pred_encoded])[0]
            pred_proba = knn.predict_proba(new_point_scaled)[0]

            # 확률 데이터프레임
            proba_df = pd.DataFrame({
                '동력유형': class_names,
                '확률 (%)': np.round(pred_proba * 100, 2)
            }).sort_values(by='확률 (%)', ascending=False)

            # 결과 표시
            col1, col2 = st.columns(2)
            
            # 모델 정확도 표시
            test_acc = accuracy_score(y_test, knn.predict(X_test_scaled))
            st.success(f"🚀 모델 정확도 (테스트 데이터): {test_acc:.2%}")
            
            with col1:
                st.metric("실제 동력유형 (데이터 기준)", true_power)
                st.metric("예측 동력유형 (KNN)", pred_label)
            with col2:
                st.metric("평균 배기량 (cc)", f"{avg_spec['engine_cc']:.1f}")
                st.metric("평균 연비 (km/L)", f"{avg_spec['fuel_efficiency']:.1f}")
                
            # 해당 유형의 대표 차종 리스트
            similar_models = sorted(df_model[df_model['power_type'] == pred_label]['model_name'].unique())
            st.info(f"**{pred_label} 대표 차종 예시**: {', '.join(similar_models[:10])}{'...' if len(similar_models) > 10 else ''}")

            # 시각화
            st.subheader("🔍 KNN 분류 시각화")

            fig, ax = plt.subplots(figsize=(12, 8))

            # 1. 결정 경계 배경
            x_min, x_max = X['engine_cc'].min() - 200, X['engine_cc'].max() + 200
            y_min, y_max = X['fuel_efficiency'].min() - 2, X['fuel_efficiency'].max() + 2
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, 50),
                np.arange(y_min, y_max, 0.5)
            )
            Z = knn.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

            # 2. 전체 학습 데이터 산점도
            scatter = ax.scatter(
                X['engine_cc'], X['fuel_efficiency'],
                c=y_encoded, cmap='coolwarm',
                edgecolors='k', s=50, alpha=0.5
            )

            # 3. 선택한 차종 평균 점만 크게 강조
            ax.scatter(
                new_point[0][0], new_point[0][1],
                color='lime', s=800, marker='X',
                edgecolors='darkgreen', linewidths=6,
                label=f'선택 차종 평균\n({selected_model})\n예측: {pred_label}'
            )

            # 축 및 제목
            ax.set_xlabel('배기량 (cc)', fontsize=14)
            ax.set_ylabel('연비 (km/L)', fontsize=14)
            ax.set_title('KNN 기반 동력유형 분류 - 결정 경계 시각화', fontsize=16, pad=20)
            ax.grid(True, alpha=0.3)

            # 범례 (동력유형 + 선택 차종만)
            handles, _ = scatter.legend_elements()
            ax.legend(
                handles + [
                    plt.Line2D([0], [0], marker='X', color='lime', markeredgecolor='darkgreen', markersize=20)
                ],
                list(class_names) + [f'선택 차종 (예측: {pred_label})'],
                title="동력유형 및 예측 점",
                loc='upper right',
                fontsize=12,
                framealpha=0.9
            )

            st.pyplot(fig)

    except Exception as e:
        st.error(f"모델 실행 중 오류: {str(e)}")
        st.info("테이블명이나 컬럼명을 확인해주세요: vehicle_classification, car_model_by_power_type")
        