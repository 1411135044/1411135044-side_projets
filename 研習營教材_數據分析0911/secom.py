import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import zipfile
import io
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import base64
from datetime import datetime
import json
import matplotlib.font_manager as fm
from scipy.stats import linregress
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if 'BASE_DIR' not in st.session_state:
    st.session_state['BASE_DIR'] = BASE_DIR



def display_dataset_info():
    """
    建立一個可展開的區塊，顯示 SECOM 資料集的詳細介紹。
    """
    st.markdown("""
    這是一個來自半導體製造過程的真實工業數據集，其主要目標是透過數據分析，找出影響產品良率的關鍵感測器訊號。
    """)

    # 使用欄位來美化排版
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **數據類型 (Dataset Characteristics):** 多變量 (Multivariate)
        - **領域 (Subject Area):** 電腦科學 (Computer Science)
        - **相關任務 (Associated Tasks):** 分類 (Classification), 因果發現 (Causal-Discovery)
        - **捐贈日期 (Donated on):** 2008-11-18
        """)
    
    with col2:
        st.markdown(f"""
        - **特徵類型 (Feature Type):** 實數 (Real)
        - **資料筆數 (# Instances):** {1567:,}
        - **特徵數量 (# Features):** 591
        - **存在缺失值 (Has Missing Values?):** 是 (Yes)
        """)

    st.markdown("""
    **資料結構:**
    數據包含兩個檔案：一個是 `secom.data`，包含了 1567 筆生產紀錄，每筆有 591 個感測器特徵；另一個是 `secom_labels.data`，包含了每筆紀錄的分類結果（-1 為 Pass, 1 為 Fail）和時間戳。數據中的缺失值以 'NaN' 表示。

    **分析目標:**
    應用**特徵選取 (feature selection)** 技術，從 591 個感測器訊號中，找出並排序出對產品**良率 (yield)** 影響最大的關鍵特徵，以幫助工程師提升產能並降低成本。
    """)
    
    st.link_button("前往 UCI 資料集原始頁面 🔗", "https://archive.ics.uci.edu/dataset/179/secom")

def display_Data_Preprocessing(summary_dict):
        st.markdown("""
        數據預處理是數據分析的基石，它確保了後續模型和統計分析的**準確性與可靠性**。
        本應用程式自動對原始 SECOM 數據執行以下關鍵步驟：
        """)
        st.markdown("##### **1. 缺失值填補：中位數策略**")
        st.markdown(f"""
        - **原理說明**: 缺失值 (NaN) 會干擾分析。本程序採用**中位數**填補，即用該特徵已有的數值中間值來替代缺失數據。
        - **使用原因**: 相較於均值，中位數受極端值（離群值）影響較小，能更穩健地反映數據的典型趨勢。
        - **提供信息**: 確保所有感測器數據在統計計算時都是完整的。
        - **推測線索**:
            * 雖然已填補，但某些特徵若有**高比例缺失值**，可能暗示其對應的感測器本身存在問題（例如：故障、間歇性讀取異常）或數據採集過程有缺陷。工程師應對此類感測器進行檢查。
        """)
        st.markdown("##### **2. 低變異特徵移除**")
        st.markdown(f"""
        - **原理說明**: 移除數值從未改變（變異數為零）的感測器特徵。
        - **使用原因**: 這些特徵不包含任何信息增益，無法用於區分良品和不良品，移除可降低模型複雜度並提升計算效率。
        - **提供信息**: 最終分析的特徵數量從 {summary_dict.get('original_features', 'N/A')} 個減少到 **{summary_dict.get('final_features', 'N/A')}** 個，其中移除了 **{summary_dict.get('removed_features', 'N/A')}** 個無用特徵。
        - **推測線索**:
            * 被移除的零變異特徵，可能意味著該感測器本身**不工作**、**讀數固定**，或者它所監測的製程參數在整個數據採集期間都**保持絕對恆定**。後者較為罕見，因此前兩種可能性更大，建議對這些感測器的狀態進行物理檢查。
        """)

# --- 1. 初始化設定與資料處理函式 (已快取) ---


def initialize_page():
    sns.set_theme(style="whitegrid", rc={"figure.autolayout": True, "axes.grid": False})
    
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'analysis_cards' not in st.session_state:
        st.session_state.analysis_cards = []
    # 初始化隨機森林參數到 session_state
    if 'n_estimators' not in st.session_state:
        st.session_state['n_estimators'] = 100
    if 'random_state' not in st.session_state:
        st.session_state['random_state'] = 42
    if 'n_jobs' not in st.session_state:
        st.session_state['n_jobs'] = -1


@st.cache_data
def load_secom_data(DATA_DIR = './secom_dataset'):
    """載入 SECOM 資料集，若不存在則下載。"""
    DATA_DIR = "secom_dataset"
    data_path = os.path.join(DATA_DIR, 'secom.data')
    labels_path = os.path.join(DATA_DIR, 'secom_labels.data')

    if not (os.path.exists(data_path) and os.path.exists(labels_path)):
        url = "https://archive.ics.uci.edu/static/public/179/secom.zip"
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extract('secom.data', path=DATA_DIR)
                z.extract('secom_labels.data', path=DATA_DIR)
        except Exception as e:
            st.error(f"自動下載 SECOM 資料集失敗: {e}")
            return None

    sensor_df = pd.read_csv(data_path, sep=' ', header=None)
    sensor_df.columns = [f'Sensor_{i+1}' for i in range(sensor_df.shape[1])]
    labels_df = pd.read_csv(labels_path, sep=' ', header=None, names=['Status', 'Timestamp'])
    labels_df['Status'] = labels_df['Status'].replace({-1: 0, 1: 1})
    return pd.concat([sensor_df, labels_df], axis=1)

@st.cache_data
def preprocess_data(df):
    """
    對數據進行預處理，並回傳清理後的資料、處理摘要文字，以及包含原始數值的字典。
    """
    df_processed = df.copy()
    
    # 移除 Timestamp 以專注於感測器數據
    X_raw = df_processed.drop(['Status', 'Timestamp'], axis=1, errors='ignore')
    original_feature_count = X_raw.shape[1]

    # 1. 移除低變異數特徵
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(X_raw)
    retained_cols_mask = selector.get_support()
    X_retained = X_raw.loc[:, retained_cols_mask]
    removed_feature_count = original_feature_count - X_retained.shape[1]
    
    # 2. 用中位數填補剩餘的缺失值
    for col in X_retained.columns:
        if X_retained[col].isnull().any():
            median_val = X_retained[col].median()
            X_retained[col] = X_retained[col].fillna(median_val)
    df_final = pd.concat([X_retained, df_processed[['Status', 'Timestamp']]], axis=1)
    
    # 3. 準備回傳的摘要資訊
    summary_text = (f"1. **缺失值處理**：所有特徵中的缺失值 (NaN) 皆已使用該特徵的**中位數**進行填補。\n"
                    f"2. **低變異特徵移除**：移除了 **{removed_feature_count}** 個數值從未改變的無用感測器特徵。\n"
                    f"3. **最終特徵數**：經過預處理後，用於分析的特徵數量為 **{X_retained.shape[1]}** 個。")
    
    summary_dict = {
        "original_features": original_feature_count,
        "removed_features": removed_feature_count,
        "final_features": X_retained.shape[1]
    }
    
    return df_final, summary_text, summary_dict

@st.cache_data
def rank_features(df, n_estimators=100, random_state=42, n_jobs=-1): # 增加參數
    """應用特徵選取技術排序特徵。"""
    X = df.drop(['Status', 'Timestamp'], axis=1)
    y = df['Status']
    
    f_scores, _ = f_classif(X, y)
    f_test_ranking = pd.DataFrame({'Feature': X.columns, 'F-Score': f_scores}).sort_values('F-Score', ascending=False).reset_index(drop=True)
    
    # 使用傳入的參數創建 RandomForestClassifier
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs).fit(X, y)
    rf_ranking = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    return f_test_ranking, rf_ranking

@st.cache_data
def get_automated_insights(df_raw, df_clean, ranked_features_df, top_n=20):
    """執行統計檢定並生成自動化洞見。"""
    top_features = ranked_features_df['Feature'].head(top_n).tolist()
    insights = []
    
    pass_group = df_raw[df_raw['Status'] == 0]
    fail_group = df_raw[df_raw['Status'] == 1]
    
    for feature in top_features:
        # 執行 t-test
        t_stat, p_value = ttest_ind(
            fail_group[feature].dropna(), 
            pass_group[feature].dropna(), 
            equal_var=False # 假設兩組變異數不相等
        )
        insights.append({'Feature': feature, 'P-Value': p_value})
        
    insights_df = pd.DataFrame(insights).sort_values('P-Value', ascending=True)
    return insights_df

# --- 3. UI 渲染函式 ---

def display_dashboard_and_drilldown(df_raw, ranked_df):
    """
    顯示全局特徵重要性概覽和單一特徵深入分析。
    """
    # 從 session_state 中獲取當前的排序方法和隨機森林參數
    ranking_method = st.session_state.get('ranking_method_radio', "隨機森林重要性")
    
    # 獲取隨機森林的動態參數
    current_n_estimators = st.session_state.get('n_estimators', 100)
    current_random_state = st.session_state.get('random_state', 42)
    current_n_jobs = st.session_state.get('n_jobs', -1)

    st.subheader(f"全局特徵重要性概覽 (分析依據: {ranking_method})")

    # 根據選擇的排序方法顯示說明和參數
    if ranking_method == "F-test 分數":
        st.markdown("""
        **F-test (ANOVA) 原理與輸入：**
        F-test 用於評估不同類別（這裡指產品的 Pass/Fail 狀態）之間，單一感測器特徵的平均值是否存在統計上的顯著差異。
        * **概念公式**: $F = \\frac{\\text{組間變異}}{\\text{組內變異}}$
        * **輸入值**: 每個**感測器特徵 (Sensor_X)** 的數值，以及對應的**產品良率狀態 (Status)**（0 為 Pass, 1 為 Fail）。
        * **說明**: F-test 計算每個感測器特徵在 Pass 和 Fail 兩組數據中的變異情況。F 值越高，表示該感測器特徵在 Pass 和 Fail 之間的分佈差異越大，對良率的區分度越好，因此被認為越重要。
        * **計算參數**:
            * **檢定類型**: 單因子變異數分析 (ANOVA F-statistic)。
            * **假設**: 資料獨立性，殘差常態性，以及各組變異數相等 (但此處 `scipy.stats.f_classif` 內部通常不需要指定 `equal_var` 參數，它主要用於 t-test。F-test 假設常態性與變異數同質性，雖然對常態性要求不嚴格，但對變異數同質性敏感，若不滿足可能需考慮 Welch's F-test，但 `f_classif` 預設用於分類任務特徵選擇)。
        """)
    elif ranking_method == "隨機森林重要性":
        st.markdown(f"""
        **隨機森林重要性原理與輸入：**
        隨機森林是一種機器學習模型，它透過評估每個感測器特徵在多個決策樹中對減少不純度的貢獻來計算其重要性。
        * **概念**: 每個特徵在隨機森林中的平均 Gini 不純度（或熵）減少量。
        * **輸入值**: 所有**感測器特徵 (Sensor_1, ..., Sensor_N)** 的數值作為輸入，**產品良率狀態 (Status)** 作為預測目標。
        * **說明**: 模型在建構決策樹時，會選擇能最大程度「純化」數據的特徵進行分裂。一個特徵被選為分裂點的次數越多、且其分裂效果越好，則其重要性分數越高。
        * **模型參數 (當前設定)**:
            * **決策樹數量 (n_estimators)**: **{current_n_estimators}** 棵樹。增加樹的數量通常會提高模型的穩定性和準確性。
            * **隨機種子 (random_state)**: **{current_random_state}**。確保每次運行結果的可重現性。
            * **並行計算 (n_jobs)**: **{current_n_jobs}**。使用所有可用的 CPU 核心進行並行計算，以加速模型訓練。
        """)

    # --- 新增 全局特徵重要性概覽的詳細說明 ---
    with st.expander("✨ 點此了解「全局特徵重要性概覽」的原理、目的與潛在線索"):
        st.markdown(f"""
        特徵重要性分析旨在從眾多感測器訊號中，識別出對產品良率（Pass/Fail）影響最大的關鍵因子。
        它幫助我們將有限的資源聚焦到最有價值的製程環節上。
        """)
        
        st.markdown("##### **1. 原理說明**")
        if ranking_method == "隨機森林重要性":
            st.write("隨機森林透過建立多棵決策樹，並統計每個特徵在樹中用於「劃分數據」的頻率和效果來衡量其重要性。")
        else: # F-test
            st.write("F-test（變異數分析）則評估單一特徵的數值在不同產品狀態（良品/不良品）之間是否存在顯著的統計差異。")
        
        st.markdown("##### **2. 提供的核心信息**")
        st.info(
            "圖表呈現了 Top 20 個感測器特徵及其對應的重要性分數，反映了它們與產品良率的**關聯強度**。"
            "分數越高，表明該特徵與良率的關係越緊密。"
        )

        st.markdown("##### **3. 可推測出的潛在線索**")
        st.write("您可以從中推斷出以下線索，引導進一步的製程優化和根因分析：")
        
        col_clue1, col_clue2 = st.columns(2)
        with col_clue1:
            st.markdown("###### 🎯 優先調查對象")
            st.success(
                "排名**靠前的特徵**，特別是**在不同分析方法（隨機森林與 F-test）下都保持高排名**的感測器，"
                "極有可能是影響良率的關鍵製程參數。它們應被列為優先調查和監控的對象。"
            )
        with col_clue2:
            st.markdown("###### 🔍 洞察製程敏感性")
            st.info(
                "這些重要特徵可能指向製程中對參數波動**高度敏感的環節**。"
                "理解這些敏感點有助於制定更精密的品質控制策略。"
            )
        
        st.warning(
            "**重要提示**：特徵重要性分數反映的是「統計關聯性」，而非直接的「因果關係」。"
            "深入的因果分析和製程優化仍需結合工程領域的專業知識進行驗證。"
        )
    st.markdown("---") # 分隔線


    # --- Altair 圖表生成 ---
    # 確保 top_20_features 是始終存在的有效 DataFrame
    top_20_features = ranked_df.head(20) if ranked_df is not None and not ranked_df.empty else pd.DataFrame({'Feature': ['無可用特徵'], '重要性分數': [0]})

    # 定義用於 Altair 的 X 軸列名，確保即使 ranked_df 為空也能找到正確的列名
    if ranked_df is not None and not ranked_df.empty:
        x_col = ranked_df.columns[1]
    else:
        x_col = '重要性分數' # Fallback column name for the default DataFrame


    chart = alt.Chart(top_20_features).mark_bar().encode(
        x=alt.X(f'{x_col}:Q', title="重要性分數"), # 使用動態的 x_col
        y=alt.Y('Feature:N', sort='-x', title="特徵 (感測器)"),
        tooltip=['Feature', x_col] # tooltip 也使用動態的 x_col
    ).properties(title=f"Top 20 關鍵特徵排序 (依據: {ranking_method})")
    
    # 直接傳遞 Altair Chart 物件給 st.altair_chart，無需轉換為 JSON 再解析
    st.altair_chart(chart, use_container_width=True)


    st.subheader("單一特徵深入分析 (Drill-Down)")
    
    with st.container(border=True): # 使用容器增加視覺區隔
        st.info(
            "此區塊允許您選擇單一感測器特徵，直觀地觀察其數值在「良品 (Pass)」與「不良品 (Fail)」之間的**分佈差異**。"
            "若兩者分佈明顯分離或重疊，可為製程問題的根因分析提供重要線索。"
        )
        
        st.markdown("---") # 分隔線
        
        st.markdown("##### **1. 原理說明**")
        st.write(
            "此處使用**核密度估計 (Kernel Density Estimation, KDE) 圖**來呈現數據分佈。"
            "KDE 圖透過平滑曲線展示數據的密度和形狀，能直觀反映出數值集中在哪裡、分佈是寬是窄，以及是否有不同的峰值（模態）。"
        )
        
        st.markdown("##### **2. 提供的核心信息**")
        st.info(
            "您可以直觀地觀察選定感測器在良品和不良品數據中的以下方面：\n"
            "- **分佈是否分離或重疊**：良品和不良品數據在數值範圍上是否存在明顯區隔。\n"
            "- **數值偏向**：不良品數據是否普遍偏高或偏低。\n"
            "- **分佈形狀**：是集中（窄峰）還是分散（寬峰），是否有不止一個峰值（多模態）。"
            "- **變異性**：不良品數據的分佈是否比良品更寬，表示其穩定性更差。"
        )
        
        st.markdown("##### **3. 可推測出的潛在線索**")
        st.write("透過觀察分佈圖，您可以推測出以下重要的製程線索：")
        
        col_dr_clue1, col_dr_clue2 = st.columns(2)
        with col_dr_clue1:
            st.markdown("###### 🔍 異常區間定位")
            st.success(
                "若良品和不良品的分佈**明顯分離**，不良品集中在某個特定數值區間，"
                "這直接指向該數值區間是**高風險區域**，對應的製程參數可能在此範圍內出現異常。"
            )
            st.markdown("###### 💡 製程穩定性問題")
            st.info(
                "若不良品的分佈**遠比良品寬泛或呈現多個峰值**，"
                "則暗示該感測器對應的製程可能不夠穩定，存在多種操作條件或故障模式。"
            )
        with col_dr_clue2:
            st.markdown("###### 🚨 參數偏移警示")
            st.warning(
                "若不良品的分佈整體向**某一方向偏移**（例如普遍偏高或偏低），"
                "可能意味著製程參數存在系統性偏差，需要檢查校準、設定或原材料穩定性。"
            )
            st.markdown("###### 🧪 潛在因果關係線索")
            st.error(
                "結合特徵重要性排名和顯著差異，若此處分佈呈現強烈區隔，"
                "該感測器是**良率問題關鍵根因**的可能性極高，應優先深入調查。"
            )
    st.markdown("---") # 分隔線 (修改結束)


    # 確保 selected_feature 在選項中，並提供安全預設值
    # FIX: 使用 top_20_features 來獲取選項，而不是 df_for_chart
    selected_feature_options = top_20_features['Feature'].tolist()
    if not selected_feature_options:
        selected_feature_options = ["無可用特徵"]
    selected_feature = st.selectbox(
        "從 Top 20 中選擇一個特徵進行深入分析：", 
        options=selected_feature_options, 
        key='drilldown_feature_select'
    )
    
    fig, ax = plt.subplots(figsize=(10, 4))
    # 確保 df_raw 不為 None 且 selected_feature 存在於 df_raw 中
    if df_raw is not None and selected_feature in df_raw.columns and 'Status' in df_raw.columns:
        sns.kdeplot(data=df_raw, x=selected_feature, hue='Status', fill=True, palette={0: 'skyblue', 1: 'salmon'}, common_norm=False, ax=ax)
    else:
        ax.text(0.5, 0.5, '無法顯示鑽取分析圖\n(數據或特徵不存在或Status列缺失)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
        ax.set_axis_off() # 關閉軸
    ax.set_title(f'"{selected_feature}" Pass (0) vs Fail (1)')
    st.pyplot(fig)
    plt.close(fig)

def display_interactive_comparison(df_raw):
    """
    渲染一個功能完整的「即時預覽，事後釘選」的互動式比較分析工作台。
    """
    st.write("在此處設定分析條件並即時預覽結果，再將滿意的分析圖表「釘選」至下方畫布。")

    numerical_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    if len(numerical_cols) < 2:
        st.warning("需要至少兩個數值型態的欄位才能進行此分析。")
        return

    # 新增互動式比較分析的說明區塊 (修改開始)
    with st.expander("✨ 點此了解「互動式比較分析」的原理、目的與潛在線索"):
        st.markdown("""
        「互動式比較分析」提供了一個高自由度的實驗場，讓您能夠深入探索**任意兩個感測器特徵之間的關係**。
        這對於理解製程中感測器之間的相互作用、識別潛在的連鎖反應或驗證特定假設至關重要。
        """)

        st.markdown("##### **1. 原理說明**")
        st.write(
            "此模塊主要利用**散佈圖 (Scatter Plot)** 視覺化兩個變數的關係。"
            "並結合**線性迴歸 (Linear Regression)** 分析來量化其線性關聯："
        )
        st.markdown(
            "- **皮爾森相關係數 (r)**: 衡量兩個變數間線性關係的強度和方向 (範圍從 -1 到 1)。"
            "- **R 平方值 (R-squared)**: 表示一個變數的變異能被另一個變數解釋的比例。"
            "- **P 值 (p-value)**: 判斷觀察到的線性關係是否具有統計顯著性。"
        )

        st.markdown("##### **2. 提供的核心信息**")
        st.info(
            "您可以觀察圖形中的點分佈模式，並參考量化指標來判斷：\n"
            "- 兩個感測器之間是否存在**線性關係**（正相關、負相關或無相關）。\n"
            "- 關係的**強度**和**統計顯著性**。\n"
            "- 是否存在**異常群體**或**聚類模式**，特別是與良品/不良品狀態相關聯的模式。"
        )

        st.markdown("##### **3. 可推測出的潛在線索**")
        st.write("您可以從中推斷出以下線索，進行更深層次的製程分析：")

        col_ia_clue1, col_ia_clue2 = st.columns(2)
        with col_ia_clue1:
            st.markdown("###### 🔗 聯動關係洞察")
            st.success(
                "如果兩個關鍵特徵之間存在**強相關性**，可能指向製程中的物理依賴或參數耦合效應。"
                "這有助於理解製程中參數的傳遞機制，甚至可以通過調整一個參數來影響另一個。"
            )
            st.markdown("###### 📉 故障模式協同分析")
            st.info(
                "觀察散佈圖中**良品和不良品是否在某個特定的二維區域內分離**，"
                "這可能指示兩種或多種感測器協同作用下的複合故障模式，需要綜合考量多個參數才能解決問題。"
            )
        with col_ia_clue2:
            st.markdown("###### 🧪 假設驗證與新發現")
            st.warning(
                "此模塊是驗證工程師對製程假設的利器。"
                "例如，假設「感測器A的變化會導致感測器B變化」，可以通過繪製其關係圖來快速驗證，甚至發現意想不到的關係。"
            )
            st.markdown("###### ⚠️ 非線性關係警示")
            st.error(
                "**重要提示**：線性迴歸僅捕捉線性關係。"
                "如果散佈圖呈現複雜的曲線或群集，可能存在**強烈的非線性關係**，此時線性相關係數的參考價值有限，建議考慮其他非線性分析方法。"
            )
    st.markdown("---") # 分隔線 (修改結束)


    # --- 1. 即時預覽區 ---
    with st.container(border=True):
        st.subheader("即時預覽與控制面板")

        # --- 控制面板 (恢復所有選項) ---
        col1, col2, col3 = st.columns(3)
        with col1:
            x_var = st.selectbox("選擇 X 軸變數:", options=numerical_cols, index=0, key="x_var_canvas")
        with col2:
            y_var = st.selectbox("選擇 Y 軸變數:", options=numerical_cols, index=1 if len(numerical_cols) > 1 else 0, key="y_var_canvas")
        with col3:
            chart_type = st.selectbox(
                "選擇圖表類型:",
                ["散佈圖與迴歸線 (regplot)", "聯合分佈圖 (jointplot)", "六角箱圖 (hexbin)", "二維密度圖 (kdeplot)", "純散佈圖 (scatterplot)"],
                key="chart_type_canvas"
            )
        
        # --- 即時預覽圖表與統計 ---
        df_filtered = df_raw[[x_var, y_var]].dropna()
        
        fig = None # 初始化圖表物件
        stats_dict = {} # 初始化統計數據字典
        
        if len(df_filtered) > 1:
            # --- 繪圖邏輯 (穩健版) ---
            try:
                if chart_type in ["散佈圖與迴歸線 (regplot)", "二維密度圖 (kdeplot)", "純散佈圖 (scatterplot)"]:
                    fig_temp, ax = plt.subplots(figsize=(8, 5))
                    if chart_type == "散佈圖與迴歸線 (regplot)":
                        sns.regplot(data=df_filtered, x=x_var, y=y_var, ax=ax, line_kws={"color": "red"}, scatter_kws={"alpha": 0.4})
                    elif chart_type == "二維密度圖 (kdeplot)":
                        sns.kdeplot(data=df_filtered, x=x_var, y=y_var, ax=ax, fill=True)
                    elif chart_type == "純散佈圖 (scatterplot)":
                        sns.scatterplot(data=df_filtered, x=x_var, y=y_var, ax=ax, alpha=0.6)
                    ax.set_title(f"{x_var} vs {y_var}")
                    fig = fig_temp
                    st.pyplot(fig)
                else: # Jointplot 和 Hexbin
                    kind = "reg" if chart_type == "聯合分佈圖 (jointplot)" else "hex"
                    g = sns.jointplot(data=df_filtered, x=x_var, y=y_var, kind=kind, height=7)
                    g.fig.suptitle(f"{x_var} vs {y_var}", y=1.02)
                    fig = g.fig
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"繪圖時發生錯誤: {e}")
            
            # --- 關係量化分析 (恢復此區塊) ---
            try:
                stats = linregress(df_filtered[x_var], df_filtered[y_var])
                stats_dict = {'r': stats.rvalue, 'r2': stats.rvalue**2, 'p': stats.pvalue}
                
                st.markdown("#### 關係量化分析")
                kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                kpi_col1.metric("皮爾森相關係數 (r)", f"{stats_dict['r']:.3f}")
                kpi_col2.metric("R 平方值 (R-squared)", f"{stats_dict['r2']:.3f}")
                kpi_col3.metric("P 值 (p-value)", f"{stats_dict['p']:.4f}")
                
                r_value, p_value, r_squared = stats_dict['r'], stats_dict['p'], stats_dict['r2']
                corr_desc = "無相關"
                if r_value > 0.7: corr_desc = "高度正相關"
                elif r_value > 0.4: corr_desc = "中度正相關"
                elif r_value > 0.1: corr_desc = "低度正相關"
                elif r_value < -0.7: corr_desc = "高度負相關"
                elif r_value < -0.4: corr_desc = "中度負相關"
                elif r_value < -0.1: corr_desc = "低度負相關"
                sig_desc = f"且此關聯性在統計上是**顯著的** (因 p < 0.05)" if p_value < 0.05 else "但此關聯性在統計上並**不顯著** (因 p >= 0.05)"
                st.info(f"**分析結論**：`{x_var}` 與 `{y_var}` 之間存在 **{corr_desc}**，{sig_desc}。`{x_var}` 的變化可以解釋 `{y_var}` 約 **{r_squared:.1%}** 的變異。")

                if st.button("➕ 將此分析釘選至下方畫布", use_container_width=True):
                    if fig:
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches='tight')
                        img_b64 = base64.b64encode(buf.getvalue()).decode()
                        new_card = {"title": f"`{x_var}` vs `{y_var}`", "chart_type": chart_type, "figure_b64": img_b64, "stats": stats_dict}
                        st.session_state.analysis_cards.append(new_card)
                    else:
                        st.warning("無法釘選，因為預覽圖表未能成功生成。")
            except ValueError:
                st.warning("數據內容無法進行線性迴歸計算。")
            
            plt.close('all')
        else:
            st.warning("所選欄位組合的有效數據不足，無法進行分析。")

    # --- 分析畫布顯示區 ---
    st.write("---")
    col_header, col_btn = st.columns([4,1])
    with col_header:
        st.subheader("分析畫布 (已釘選的圖表)")
    with col_btn:
        if st.button("🗑️ 清空所有圖表", use_container_width=True):
            st.session_state.analysis_cards = []
            st.rerun()

    if not st.session_state.analysis_cards:
        st.info("目前畫布是空的。請在上方「即時預覽區」設定分析條件後，點擊「新增圖表至下方畫布」。")
    else:
        for i in range(0, len(st.session_state.analysis_cards), 2):
            c1, c2 = st.columns(2)
            with c1:
                card1 = st.session_state.analysis_cards[i]
                with st.container(border=True):
                    st.markdown(f"##### {card1['title']} ({card1['chart_type'].split(' ')[0]})")
                    st.image(base64.b64decode(card1['figure_b64']))
                    s1 = card1['stats']
                    st.write(f"r: {s1['r']:.3f} | R²: {s1['r2']:.3f} | p-value: {s1['p']:.3g}")
            if i + 1 < len(st.session_state.analysis_cards):
                with c2:
                    card2 = st.session_state.analysis_cards[i + 1]
                    with st.container(border=True):
                        st.markdown(f"##### {card2['title']} ({card2['chart_type'].split(' ')[0]})")
                        st.image(base64.b64decode(card2['figure_b64']))
                        s2 = card2['stats']
                        st.write(f"r: {s2['r']:.3f} | R²: {s2['r2']:.3f} | p-value: {s2['p']:.3g}")

def display_automated_insights(df_raw, insights_df):
    st.subheader("依統計顯著性排序的異常訊號報告")
    st.info("此報告自動找出良品與不良品數據分佈差異最顯著的特徵，並由上至下排序。")
    
    # 修改並新增 自動化洞見說明區塊 (修改開始)
    st.markdown("""
    ---
    #### **異常訊號評判標準說明**
    此報告依據各感測器訊號在良品與不良品之間分佈的**統計顯著性 (Statistical Significance)** 進行排序。統計顯著性透過 **P 值 (p-value)** 來衡量。
    * **P 值 (p-value)**：表示如果「良品與不良品之間沒有差異」這個假設（即「虛無假設」）成立的情況下，觀察到目前或更極端差異的機率。
    * **P 值算法與輸入值**：
        * 這裡的 P 值是透過對每個感測器特徵的**良品組數據**和**不良品組數據**執行**獨立樣本 t-test**（獨立樣本 t 檢定）計算得出的。
        * **t-test 概念**: t-test 用於比較兩個獨立樣本的平均值是否存在統計上的顯著差異。其計算公式為：
            $t = \\frac{(\\bar{X}_1 - \\bar{X}_2)}{\\sqrt{\\frac{s_1^2}{n_1} + \\frac{s_2^2}{n_2}}}$
            其中：
            * $\\bar{X}_1, \\bar{X}_2$ 是兩個組的樣本平均值。
            * $s_1^2, s_2^2$ 是兩個組的樣本變異數。
            * $n_1, n_2$ 是兩個組的樣本大小。
        * **輸入值**：
            * **組 1 (不良品組)**：選定感測器特徵在**所有不良品 (Status = 1)** 生產紀錄中的數值。
            * **組 2 (良品組)**：選定感測器特徵在**所有良品 (Status = 0)** 生產紀錄中的數值。
            * **假設變異數不相等 (equal_var=False)**。
        * 透過計算出的 t 值和自由度，可以查表或利用統計軟體得出 P 值。P 值越小，表示兩組（良品與不良品）在該感測器數值上的差異越顯著。
    """)

    with st.expander("✨ 點此了解「自動化洞見報告」的目的與潛在線索"):
        st.markdown("""
        此報告自動篩選出良品與不良品間數據分佈差異最顯著的感測器，作為**根因分析的優先起點**。
        它能幫助工程師快速聚焦於最可能影響產品良率的關鍵環節，大幅節省人工排查時間。
        """)

        st.markdown("##### **1. 提供的核心信息**")
        st.info(
            "報告呈現了根據 P 值由小到大排序的感測器訊號列表，"
            "P 值越小，表示該感測器在良品與不良品之間的分佈差異越不可能由隨機造成，統計顯著性越高。"
        )

        st.markdown("##### **2. 可推測出的潛在線索**")
        st.write("您可以從中推斷出以下線索，引導進一步的製程優化和問題排查：")

        col_ai_clue1, col_ai_clue2 = st.columns(2)
        with col_ai_clue1:
            st.markdown("###### 🚨 高優先級問題點")
            st.error(
                "P 值**高度顯著 (< 0.001)** 的特徵是**最緊急的調查對象**。"
                "它們極有可能是導致良率問題的直接或間接根源，應立即結合領域知識進行深入檢查。"
            )
            st.markdown("###### ✅ 關注潛在風險")
            st.warning(
                "P 值**顯著 (< 0.05)** 但非高度顯著的特徵，也值得持續關注。"
                "它們可能在特定條件下（如不同批次、季節性變化）影響良率，是潛在的風險點。"
            )
        with col_ai_clue2:
            st.markdown("###### 🔍 節省排查時間")
            st.info(
                "此報告將數百個感測器中的重點自動篩選出來，"
                "大幅縮短了工程師從原始數據中尋找異常的時間，直接指向最有價值的分析方向。"
            )
            st.markdown("###### 🧪 驗證與行動")
            st.success(
                "針對排名靠前的異常訊號，建議進行**製程參數檢查、設備校準、操作流程審核**，"
                "並追溯相關批次的生產記錄，以驗證數據與實際問題的因果關係。"
            )
        
        st.info(
            "**多重比較提示**：由於對多個感測器進行了獨立的統計檢定，隨機出現「顯著」結果的機率會增加（即假陽性）。"
            "對於高度嚴謹的結論，建議對 P 值進行額外的**多重比較校正**（例如 Bonferroni 校正），或透過現場實驗再次驗證。"
        )
    st.markdown("---") # 分隔線 (修改結束)


    # 視覺化 P 值判斷準則 (保持不變)
    st.markdown("##### **P 值判斷準則**")
    col_v1, col_v2, col_v3 = st.columns(3)

    with col_v1:
        st.error("**高度顯著差異**")
        st.markdown("<p style='color:red;'><b>P < 0.001</b></p>", unsafe_allow_html=True)
        st.write("觀察到的差異極不可能由隨機造成。此訊號在良品與不良品之間存在非常強的統計差異，應列為**首要調查對象**。")

    with col_v2:
        st.warning("**顯著差異**")
        st.markdown("<p style='color:orange;'><b>P < 0.05</b></p>", unsafe_allow_html=True)
        st.write("觀察到的差異不太可能由隨機造成。此訊號在兩組樣本中存在統計上的顯著差異，值得進一步關注。")

    with col_v3:
        st.success("**無顯著差異**")
        st.markdown("<p style='color:green;'><b>P ≥ 0.05</b></p>", unsafe_allow_html=True)
        st.write("觀察到的差異可能是由於隨機機會造成的。此訊號在良品與不良品之間未表現出統計上的顯著差異，其變化對良率影響較小或不明顯。")
    
    st.markdown("---") # 分隔線

    for _, row in insights_df.iterrows():
        feature = row['Feature']
        p_value = row['P-Value']

        with st.container(border=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(8, 3))
                # 確保 df_raw 不為 None 且 feature 存在於 df_raw 中
                if df_raw is not None and feature in df_raw.columns and 'Status' in df_raw.columns:
                    sns.kdeplot(data=df_raw, x=feature, hue='Status', fill=True, palette={0: 'skyblue', 1: 'salmon'}, common_norm=False, ax=ax)
                else:
                    ax.text(0.5, 0.5, f'無法顯示特徵 "{feature}" 圖表\n(數據或Status列缺失)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
                    ax.set_axis_off()
                ax.get_legend().remove()
                ax.set_title("")
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                st.markdown(f"#### `{feature}`")
                if p_value < 0.001:
                    st.error(f"**高度顯著差異** (p < 0.001)")
                    st.write("此訊號在良品與不良品之間存在極其顯著的統計差異，是根本原因分析的**首要調查對象**。")
                elif p_value < 0.05:
                    st.warning(f"**顯著差異** (p = {p_value:.4f})")
                    st.write("此訊號在兩組樣本中存在統計上的顯著差異，值得關注。")
                else:
                    st.success(f"無顯著差異 (p = {p_value:.4f})")
                    st.write("此訊號在良品與不良品之間未表現出統計上的顯著差異。")
# --- 4. 報告生成 ---
def display_report_export_section(df_raw, df_clean, f_test_ranking, rf_ranking, summary_text, summary_dict):

    # --- 從 session_state 和變數中，收集所有分析的當前狀態 ---
    ranking_method = st.session_state.get('ranking_method_radio', "隨機森林重要性")
    ranked_df = rf_ranking if ranking_method == "隨機森林重要性" else f_test_ranking
    
    # 確保 ranked_df 不為空，避免後續取值錯誤
    if ranked_df.empty:
        st.warning("排名列表為空，無法生成報告。")
        return

    # 確保 drilldown_feature 有一個安全的值，避免索引錯誤
    drilldown_feature = st.session_state.get('drilldown_feature_select', ranked_df['Feature'].iloc[0] if not ranked_df.empty else "N/A")

    numerical_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    x_var = st.session_state.get('x_var_canvas', numerical_cols[0] if numerical_cols else "N/A")
    y_var = st.session_state.get('y_var_canvas', numerical_cols[1] if len(numerical_cols)>1 else "N/A")
    chart_type = st.session_state.get('chart_type_canvas', "散佈圖與迴歸線 (regplot)")
    comparison_selections = (x_var, y_var, chart_type)
    
    # 計算互動比較區的統計數據
    regression_stats = {}
    if x_var != "N/A" and y_var != "N/A":
        df_filtered = df_raw[[x_var, y_var]].dropna()
        if len(df_filtered) > 1:
            try:
                stats = linregress(df_filtered[x_var], df_filtered[y_var])
                r_value, p_value, r_squared = stats.rvalue, stats.pvalue, stats.rvalue**2
                
                # 自動生成文字結論
                corr_desc = "無相關"
                if r_value > 0.7: corr_desc = "高度正相關"
                elif r_value > 0.4: corr_desc = "中度正相關"
                elif r_value > 0.1: corr_desc = "低度正相關"
                elif r_value < -0.7: corr_desc = "高度負相關"
                elif r_value < -0.4: corr_desc = "中度負相關"
                elif r_value < -0.1: corr_desc = "低度負相關"
                sig_desc = "統計上顯著" if p_value < 0.05 else "統計上不顯著"
                conclusion_text = f"`{x_var}` 與 `{y_var}` 之間存在 {corr_desc} (r={r_value:.2f})，此關聯性在 {sig_desc} (p={p_value:.3f})。`{x_var}` 的變化可以解釋 `{y_var}` 約 {r_squared:.1%} 的變異。"
                
                regression_stats = {'r': r_value, 'r2': r_squared, 'p': p_value, 'conclusion': conclusion_text}
            except ValueError:
                regression_stats = {'conclusion': '數據無法進行線性迴歸計算。'}
        else: # Add this else block for when df_filtered is not long enough
            regression_stats = {'conclusion': '所選欄位數據不足，無法進行迴歸計算。'}
    else: # If x_var or y_var is N/A
        regression_stats = {'conclusion': '未選擇足夠的數值欄位進行迴歸計算。'}


    insights_df = get_automated_insights(df_raw, df_clean, ranked_df)
    pinned_cards = st.session_state.get('analysis_cards', [])

        

# --- 5. 主執行函式 ---
# ... (其他函式保持不變)

def display_analysis_section(df_raw, df_clean, f_test_ranking, rf_ranking, summary_text, summary_dict):
    """
    功能區二：資料分析專區，呈現所有分析並呼叫匯出功能 (最終版)。
    """
    st.header("2. 資料分析專區", divider='gray')
    
    # 統一控制面板 - 選擇排序方法
    ranking_method = st.radio("選擇排序方法以驅動下方分析：", ("隨機森林重要性", "F-test 分數"), horizontal=True, key='ranking_method_radio')
    
    # 動態顯示隨機森林參數調整選項
    st.markdown("---")
    st.markdown("#### **演算法參數設定**")
    if ranking_method == "隨機森林重要性":
        st.write("您可以調整隨機森林模型的參數來觀察特徵重要性的變化。")
        col_param1, col_param2, col_param3 = st.columns(3)
        
        with col_param1:
            # 使用 session_state 保持參數值
            st.session_state['n_estimators'] = st.number_input(
                "決策樹數量 (n_estimators):", 
                min_value=10, 
                max_value=1000, 
                value=st.session_state.get('n_estimators', 100), # 預設值為100
                step=10,
                help="隨機森林中決策樹的數量。數量越多通常模型越穩定，但計算量也越大。"
            )
        
        with col_param2:
            st.session_state['random_state'] = st.number_input(
                "隨機種子 (random_state):", 
                min_value=0, 
                max_value=1000, 
                value=st.session_state.get('random_state', 42), # 預設值為42
                step=1,
                help="用於隨機數生成的種子。設定後可確保每次運行結果相同，方便重現。"
            )
            
        with col_param3:
            # n_jobs 較特殊，通常只有 -1 (所有核心) 或 1 (單核心)
            st.session_state['n_jobs'] = st.selectbox(
                "並行計算核心數 (n_jobs):", 
                options=[-1, 1], # -1 表示使用所有可用核心
                index=0 if st.session_state.get('n_jobs', -1) == -1 else 1,
                help="指定用於模型訓練的 CPU 核心數。-1 表示使用所有可用核心。"
            )

        # 添加一個按鈕來觸發重新計算，只有在參數改變時才需要
        if st.button("應用隨機森林參數並重新計算", key='apply_rf_params'):
            # 清除 rank_features 的快取，強制重新運行
            st.cache_data.clear() # 清除所有 cache_data 的快取
            st.rerun() # 重新運行應用程式以應用新參數
    elif ranking_method == "F-test 分數": # F-test 沒有可調整的常見參數
        st.info("F-test (ANOVA) 為統計檢定方法，通常無須額外參數設定。")
    st.markdown("---")
    
    # 根據選擇，設定 ranked_df
    # 在這裡呼叫 rank_features 時，傳入來自 session_state 的參數
    if ranking_method == "隨機森林重要性":
        # 這裡需要重新計算 rank_features 以使用最新的參數
        # 注意：每次 reruns() 時都會執行這裡
        f_test_ranking_recalc, rf_ranking_recalc = rank_features(
            df_clean,
            n_estimators=st.session_state.get('n_estimators', 100),
            random_state=st.session_state.get('random_state', 42),
            n_jobs=st.session_state.get('n_jobs', -1)
        )
        ranked_df = rf_ranking_recalc
        # 如果用戶切回 F-test，則直接使用之前計算好的 f_test_ranking
        # 避免不必要的重新計算
        # 但由於 st.cache_data.clear()，這裡每次都會重新計算，確保最新狀態
    else:
        ranked_df = f_test_ranking
    
    # 三種分析模式頁籤
    tab_dashboard, tab_compare, tab_insights = st.tabs(["📊 儀表板與鑽取分析", "🔬 互動式比較分析", "💡 自動化洞見報告"])

    with tab_dashboard:
        # 將正確的 ranked_df 傳遞給 display_dashboard_and_drilldown
        display_dashboard_and_drilldown(df_raw, ranked_df)
    with tab_compare:
        display_interactive_comparison(df_raw)
    with tab_insights:
        insights_df = get_automated_insights(df_raw, df_clean, ranked_df)
        display_automated_insights(df_raw, insights_df)
 
    # 在所有分析頁籤之後，呼叫匯出功能區塊，並將 summary_text 傳遞下去
    if 'insights_df' in locals():
        display_report_export_section(df_raw, df_clean, f_test_ranking, rf_ranking, summary_text, summary_dict)

def render_sidebar():
    """
    在 Streamlit 側邊欄中渲染所有功能的說明手冊。
    """
    st.sidebar.title("📖 功能說明手冊")
    st.sidebar.markdown("---")

    # 區塊一：預處理摘要
    with st.sidebar.expander("1. 資料預處理摘要", expanded=False):
        st.write("""
        此區塊會自動對載入的 SECOM 資料集進行基礎的數據清洗，並顯示摘要：
        - **缺失值處理**: 使用各欄位的中位數填補空值。
        - **無用特徵移除**: 自動刪除數值從未改變的感測器特徵，因為它們不帶有任何分析價值。
        """)

    # 區塊二：資料分析專區
    with st.sidebar.expander("2. 資料分析專區", expanded=True):
        st.write("""
        這是核心的互動分析區域，包含三個功能頁籤。您可以在頂部切換特徵排序的**分析依據**（統計檢定 vs. 機器學習模型），這個選擇會影響「儀表板」和「自動化洞見」的排序。
        """)
        st.markdown("##### 📊 儀表板與鑽取分析")
        st.write("""
        - **目的**: 提供「由總到分」的快速診斷。
        - **操作**: 首先查看 Top 20 特徵的全局重要性長條圖，然後從下拉選單中選擇您感興趣的單一特徵，下方會立即顯示該特徵在良品與不良品中的數據分佈對比圖。
        """)

        st.markdown("##### 🔬 互動式比較分析")
        st.write("""
        - **目的**: 提供一個高自由度的「分析畫布」，用於探索任意兩個數值變數間的關係。
        - **操作**:
            1.  在**控制面板**選擇 X 軸、Y 軸變數和圖表類型。
            2.  在下方的**即時預覽區**查看圖表和量化統計結果。
            3.  若對結果滿意，點擊 **「➕ 釘選至畫布」** 按鈕，將該分析卡片存至下方畫布。
            4.  重複以上步驟，即可在畫布上並排比較多個分析結果。
            5.  點擊 **「🗑️ 清空所有圖表」** 可重置畫布。
        """)

        st.markdown("##### 💡 自動化洞見報告")
        st.write("""
        - **目的**: 由程式自動找出最值得關注的異常訊號。
        - **內容**: 此頁籤會對排名靠前的特徵自動執行統計檢定，並依照**統計顯著性**（P 值由小到大）排序，優先呈現良/劣品數據分佈差異最大的特徵及其視覺化圖表。
        """)
    
    st.sidebar.markdown("---")


def display_report_page(df_raw=None, df_clean=None, f_test_ranking=None, rf_ranking=None, summary_text="無摘要", summary_dict=None):
    """顯示整合報告頁面，直接呈現所有處理過的分析結果"""
    st.set_page_config(page_title="SECOM 分析報告", page_icon="📊", layout="wide")
    st.title("半導體製程分析報告 (SECOM Dataset)")
    st.markdown(f"**報告生成日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 側邊欄返回按鈕
    with st.sidebar:
        st.info("如果要列印成 PDF，點擊右上角的3個\"點點\"按鈕，再點擊 Print 按鈕即可")
        st.info("列印 PDF 時 建議隱藏側邊欄")
        if st.button("返回主應用", key="back_to_main", use_container_width=True):
            st.session_state['page'] = 'main'
            st.rerun()

    st.divider()

    # 從 session_state 獲取數據，若未提供則使用預設值
    df_raw = df_raw if df_raw is not None else st.session_state.get('df_raw', None)
    df_clean = df_clean if df_clean is not None else st.session_state.get('df_clean', None)
    f_test_ranking = f_test_ranking if f_test_ranking is not None else st.session_state.get('f_test_ranking', None)
    rf_ranking = rf_ranking if rf_ranking is not None else st.session_state.get('rf_ranking', None)
    summary_text = summary_text if summary_text != "無摘要" else st.session_state.get('summary_text', "無摘要")
    summary_dict = summary_dict if summary_dict is not None else st.session_state.get('summary_dict', {})

    # 檢查數據是否完整
    if df_raw is None or df_clean is None:
        st.error("數據未正確載入，請返回主頁面重新載入數據。")
        return

    # 提取報告狀態
    ranking_method = st.session_state.get('ranking_method_radio', "隨機森林重要性")
    ranked_df = rf_ranking if ranking_method == "隨機森林重要性" else f_test_ranking
    if ranked_df is None:
        st.error("無法生成特徵排名數據，請確保數據已正確載入並處理。")
        ranked_df = pd.DataFrame({'Feature': ['N/A'], 'Score': [0]})

    insights_df = st.session_state.get('insights_df', None)
    pinned_cards = st.session_state.get('analysis_cards', [])
    numerical_cols = df_raw.select_dtypes(include=np.number).columns.tolist() if df_raw is not None else []
    x_var = st.session_state.get('x_var_canvas', numerical_cols[0] if numerical_cols else "N/A")
    y_var = st.session_state.get('y_var_canvas', numerical_cols[1] if len(numerical_cols) > 1 else "N/A")
    chart_type = st.session_state.get('chart_type_canvas', "散佈圖與迴歸線 (regplot)")

    # 計算迴歸統計
    regression_stats = {}
    if x_var != "N/A" and y_var != "N/A" and df_raw is not None:
        df_filtered = df_raw[[x_var, y_var]].dropna()
        if len(df_filtered) > 1:
            try:
                stats = linregress(df_filtered[x_var], df_filtered[y_var])
                r_value, p_value, r_squared = stats.rvalue, stats.pvalue, stats.rvalue**2
                corr_desc = "無相關"
                if r_value > 0.7: corr_desc = "高度正相關"
                elif r_value > 0.4: corr_desc = "中度正相關"
                elif r_value > 0.1: corr_desc = "低度正相關"
                elif r_value < -0.7: corr_desc = "高度負相關"
                elif r_value < -0.4: corr_desc = "中度負相關"
                elif r_value < -0.1: corr_desc = "低度負相關"
                sig_desc = "統計上顯著" if p_value < 0.05 else "統計上不顯著"
                conclusion_text = f"`{x_var}` 與 `{y_var}` 之間存在 {corr_desc} (r={r_value:.2f}, p={p_value:.3f}, R²={r_squared:.1%})。"
                regression_stats = {'r': r_value, 'r2': r_squared, 'p': p_value, 'conclusion': conclusion_text}
            except ValueError:
                regression_stats = {'conclusion': '無法進行線性迴歸計算。'}
        else:
            regression_stats = {'conclusion': '數據不足，無法進行迴歸計算。'}
    else:
        regression_stats = {'conclusion': '未選擇有效數值欄位。'}

    # 1. 數據集概述
    st.header("1. 數據集概述", divider="gray")
    display_dataset_info()
    st.subheader("數據預覽")
    st.dataframe(df_raw.head(100))  # 直接顯示前100筆，符合原始main的預覽

    # 2. 預處理摘要
    st.header("2. 預處理摘要", divider="gray")
    st.info(summary_text)
    display_Data_Preprocessing(summary_dict)

    # 3. 儀表板與鑽取分析結果
    st.header("3. 儀表板與鑽取分析結果", divider="gray")
    st.markdown(f"**特徵排名 (依據: {ranking_method})**")
    if not ranked_df.empty:
        st.dataframe(ranked_df.head(5)[['Feature', ranked_df.columns[1]]])
        chart = alt.Chart(ranked_df.head(20)).mark_bar().encode(
            x=alt.X(f'{ranked_df.columns[1]}:Q', title="重要性分數"),
            y=alt.Y('Feature:N', sort='-x', title="特徵"),
            tooltip=['Feature', ranked_df.columns[1]]
        ).properties(title=f"Top 20 特徵排名")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("無可用排名數據。")

    # 4. 自動化洞見報告結果
    st.header("4. 自動化洞見報告結果", divider="gray")
    display_automated_insights(df_raw, insights_df)

    # 5. 互動式比較分析結果
    st.header("5. 互動式比較分析結果", divider="gray")
    st.markdown(f"**當前選擇**: X={x_var}, Y={y_var}, 圖表類型={chart_type}  ")
    st.markdown(f"**迴歸分析**: {regression_stats.get('conclusion', '無結論')}")
    if pinned_cards:
        st.subheader("釘選圖表")
        for card in pinned_cards:
            with st.container(border=True):
                st.markdown(f"**{card['title']}** ({card['chart_type']})")
                st.image(base64.b64decode(card['figure_b64']))
                s = card['stats']
                st.write(f"r: {s['r']:.3f} | R²: {s['r2']:.3f} | p-value: {s['p']:.3g}")
    else:
        st.info("無釘選圖表。")
def main():
    st.set_page_config(page_title="半導體製程分析儀表板", page_icon="🏭", layout="wide")
    

    # 初始化 session_state
    if 'page' not in st.session_state:
        st.session_state['page'] = 'main'
        
    # 頁面切換邏輯
    if st.session_state['page'] == 'main':
        st.title("半導體製程分析儀表板 (SECOM Dataset)")
        render_sidebar()
        initialize_page()
        with st.expander("ℹ️ 點此查看 SECOM 資料集詳細介紹"):
            display_dataset_info()

        # 步驟一：載入資料
        df_raw = load_secom_data()
        if df_raw is None:
            if st.button("點此下載 SECOM 資料集"):
                pass
            st.stop()

        # 步驟二：執行所有後端分析與計算
        with st.spinner("正在進行數據預處理與特徵工程計算..."):
            df_clean, summary_text, summary_dict = preprocess_data(df_raw)
            if df_clean is None:
                st.error("數據預處理失敗，請檢查數據完整性。")
                st.stop()
            f_test_ranking, rf_ranking = rank_features(df_clean, 
                                                    n_estimators=st.session_state.get('n_estimators', 100),
                                                    random_state=st.session_state.get('random_state', 42),
                                                    n_jobs=st.session_state.get('n_jobs', -1))
            if f_test_ranking is None or rf_ranking is None:
                st.error("特徵排名計算失敗，請檢查數據或參數設置。")
                st.stop()

        # 確保所有變數存入 session_state
        st.session_state['df_raw'] = df_raw
        st.session_state['df_clean'] = df_clean
        st.session_state['f_test_ranking'] = f_test_ranking
        st.session_state['rf_ranking'] = rf_ranking
        st.session_state['summary_text'] = summary_text
        st.session_state['summary_dict'] = summary_dict
        # 計算並儲存 insights_df
        ranking_method = st.session_state.get('ranking_method_radio', "隨機森林重要性")
        ranked_df = rf_ranking if ranking_method == "隨機森林重要性" else f_test_ranking
        st.session_state['insights_df'] = get_automated_insights(df_raw, df_clean, ranked_df)

        # 步驟三：顯示預處理摘要
        st.header("1. 資料預處理摘要", divider='gray')
        with st.expander("點此查看原始數據預覽 (前100筆)"):
            st.dataframe(df_raw.head(100))
        st.markdown("#### 預處理總結")
        st.info(summary_text)

        with st.expander("✨ 點此了解數據預處理的原理、目的與潛在線索"):
            display_Data_Preprocessing(summary_dict)
            

        # 增加下載處理後 CSV 檔案的按鈕
        st.markdown("---")
        st.subheader("處理後數據下載")
        st.write("您可以選擇下載全部或部分經過預處理後的數據。")
        st.warning("如果下載失敗並出現 'Request failed with status code 413' 錯誤，表示檔案過大，請嘗試選擇更少的 'Top N' 特徵進行下載。")

        download_option = st.radio(
            "選擇下載數據的範圍：",
            ("下載所有處理後的特徵", "下載 Top N 重要特徵數據"),
            key="download_csv_option",
            horizontal=True
        )

        if download_option == "下載所有處理後的特徵":
            csv_data = df_clean.to_csv(index=False).encode('utf-8')
            file_name = "secom_processed_all_features_data.csv"
            mime_type = "text/csv"
            download_label = "下載所有特徵數據 (CSV)"
            help_text = "點擊下載包含所有處理後特徵的 SECOM 數據。"
        else:
            ranking_method = st.session_state.get('ranking_method_radio', "隨機森林重要性")
            current_ranked_df = rf_ranking if ranking_method == "隨機森林重要性" else f_test_ranking
            if current_ranked_df is None or current_ranked_df.empty:
                st.warning("無法獲取重要特徵排名，請確保數據已載入並處理。")
                st.stop()

            st.info(f"當前 'Top N' 特徵是根據 **{ranking_method}** 進行排序的。")
            max_slider_value = min(50, df_clean.shape[1] - 2)
            top_n = st.slider(
                "選擇要下載的 Top N 個重要特徵：",
                min_value=5,
                max_value=max_slider_value,
                value=min(20, max_slider_value),
                step=5,
                help="選擇包含在 CSV 中的最重要特徵數量。如果仍無法下載，請嘗試更小的值。"
            )
            top_features_cols = current_ranked_df['Feature'].head(top_n).tolist()
            cols_to_download = [col for col in top_features_cols if col in df_clean.columns]
            if 'Status' in df_clean.columns:
                cols_to_download.append('Status')
            if 'Timestamp' in df_clean.columns:
                cols_to_download.append('Timestamp')
            final_cols_ordered = [col for col in df_clean.columns if col in cols_to_download]
            df_download = df_clean[final_cols_ordered]
            csv_data = df_download.to_csv(index=False).encode('utf-8')
            method_for_filename = "RandomForest" if ranking_method == "隨機森林重要性" else "Ftest"
            file_name = f"secom_processed_top_{top_n}_features_by_{method_for_filename}_data.csv"
            mime_type = "text/csv"
            download_label = f"下載 Top {top_n} ({ranking_method}) 特徵數據 (CSV)"
            help_text = f"點擊下載包含 Top {top_n} 個根據 {ranking_method} 排序的重要特徵和良率狀態的 SECOM 數據。"

        st.download_button(
            label=download_label,
            data=csv_data,
            file_name=file_name,
            mime=mime_type,
            help=help_text
        )
        st.markdown("---")

        # 顯示分析區並添加報告按鈕
        display_analysis_section(df_raw, df_clean, f_test_ranking, rf_ranking, summary_text, summary_dict)
        
        st.divider()
        if st.button("查看整合報告", key="view_report", use_container_width=True):
            st.session_state['page'] = 'report'
            st.rerun()

        st.caption("""
        國立臺中科技大學人工智慧應用工程學士學位學程 Copyright © 2022 NTCUST Bachelor Degree Program of Artificial Intelligence     
        地址 : 404 臺中市三民路三段129號資訊樓8樓2805室   
        電話 : 04 - 2219 - 6308   
        傳真 : 04 - 2219 - 6301    
        信箱 : ai01@nutc.edu.tw
        """)
    else:
        # 報告頁面：從 session_state 獲取數據
        display_report_page()


if __name__ == "__main__":
    main()