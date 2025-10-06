import streamlit as st
import pandas as pd
import os
import json
import zipfile
import logging
import glob
from pygwalker.api.streamlit import StreamlitRenderer
# --- 基本設定 ---
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Kaggle 分析工具", layout="wide")
BASE_DIR = st.session_state.get('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))

# --- 認證函數 ---
@st.cache_resource
def init_kaggle_api():
    local_kaggle_path = "kaggle.json"
    if os.path.exists(os.path.join(BASE_DIR, local_kaggle_path)):
        try:
            with open(os.path.join(BASE_DIR, local_kaggle_path), 'r') as f:
                credentials = json.load(f)
            os.environ['KAGGLE_USERNAME'] = credentials['username']
            os.environ['KAGGLE_KEY'] = credentials['key']
        except Exception as e:
            st.error(f"讀取本地 `kaggle.json` 檔案時發生錯誤: {e}")
            st.session_state['api_authenticated'] = False
            return None

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        st.session_state['api_authenticated'] = True
        return api
    except Exception as e:
        st.error(f"Kaggle API 認證失敗，請檢查您的憑證設定。")
        st.session_state['api_authenticated'] = False
        return None

def code_finder_page(api):
    st.title("🔎 尋找相關程式碼 (Kernels)")
    st.info("您可以輸入一個資料集的 Ref，來尋找其他開發者是如何分析這個資料集的。")

    # --- 步驟一：輸入資料集 Ref，尋找相關程式碼 ---
    st.subheader("步驟一：輸入資料集 Ref，尋找相關程式碼")
    dataset_ref = st.text_input(
        "輸入資料集 Ref：", 
        placeholder="例如: harshsingh2209/supply-chain-analysis",
        key="code_finder_dataset_ref"
    )
    st.caption("說明：輸入資料集的 Ref（如 'username/dataset-name'），可從 Kaggle API 工具頁面複製。")
    if st.button("尋找程式碼", key="find_kernels_button"):
        st.caption("說明：點擊後，系統將顯示與該資料集相關的程式碼清單（Kaggle Kernels）。")
        if dataset_ref:
            st.session_state.current_dataset_ref = dataset_ref  # 新增：儲存當前資料集 Ref，用於後續路徑
            with st.spinner(f"正在尋找與 '{dataset_ref}' 相關的程式碼..."):
                try:
                    kernels = api.kernels_list(dataset=dataset_ref, sort_by='hotness', page_size=50)
                    if kernels:
                        kernel_data = [{
                            'Ref': k.ref,
                            'Title': k.title,
                            'Author': k.author,
                        } for k in kernels]
                        st.session_state.kernels_df = pd.DataFrame(kernel_data)
                    else:
                        st.warning("找不到與此資料集相關的程式碼。")
                        st.session_state.kernels_df = pd.DataFrame()
                except Exception as e:
                    st.error(f"尋找程式碼時發生錯誤: {e}")
        else:
            st.warning("請先輸入資料集 Ref。")
    
    if 'kernels_df' in st.session_state and not st.session_state.kernels_df.empty:
        st.dataframe(st.session_state.kernels_df, use_container_width=True)
        
        # --- 步驟二：下載 Kernel ---
        st.markdown("---")
        st.subheader("步驟二：從上方結果複製 Ref，下載程式碼")
        kernel_ref_to_download = st.text_input(
            "輸入程式碼 Ref：",
            placeholder="例如: some-user/awesome-analysis-notebook",
            key="kernel_ref_to_download"
        )
        st.caption("說明：Ref 是程式碼的唯一識別碼，格式如 'username/kernel-name'，可從上方表格複製。")
        if st.button("下載程式碼 (.ipynb)", key="download_kernel_button"):
            st.caption("說明：點擊後，程式碼將下載為 .ipynb 檔案至 'kaggle_downloads/{dataset_ref}/kernels' 資料夾。")
            if kernel_ref_to_download and 'current_dataset_ref' in st.session_state:
                base_path = 'kaggle_downloads'
                dataset_path = os.path.join(base_path, st.session_state.current_dataset_ref)
                kernel_download_path = os.path.join(dataset_path, 'kernels')  # 新增：kernels 子資料夾
                os.makedirs(kernel_download_path, exist_ok=True)
                with st.spinner(f"正在下載 '{kernel_ref_to_download}'..."):
                    try:
                        api.kernels_pull(kernel_ref_to_download, path=kernel_download_path)
                        st.success(f"程式碼 '{kernel_ref_to_download}.ipynb' 已成功下載到 `{kernel_download_path}` 資料夾！")
                        
                        # 新增：步驟三 - 選擇程式碼並導向 Colab
                        st.markdown("---")
                        st.subheader("步驟三：運行程式碼在 Google Colab")
                        # 列出 kernels 資料夾下的 .ipynb 檔案
                        ipynb_files = glob.glob(os.path.join(kernel_download_path, '*.ipynb'))
                        if ipynb_files:
                            display_files = [os.path.basename(f) for f in ipynb_files]  # 只顯示檔名
                            selected_ipynb = st.selectbox("選擇一個已下載的程式碼檔案運行：", display_files, key="selected_kernel_file")
                            st.caption("說明：選擇檔案後，點擊下方按鈕開啟 Colab，並上傳該檔案運行學習數據分析。")
                            if selected_ipynb:
                                selected_path = os.path.join(kernel_download_path, selected_ipynb)
                                st.info(f"""
                                您選擇了 '{selected_ipynb}'！現在可以將它上傳到 Google Colab 運行。
                                
                                **步驟指南：**
                                1. 點擊下方按鈕開啟 Google Colab（新分頁）。
                                2. 在 Colab 介面，選擇「檔案 > 上傳筆記本」或直接拖曳 '{selected_ipynb}' 檔案上傳（檔案位於 {selected_path}）。
                                3. 上傳後，即可編輯、運行程式碼（Colab 支援 GPU/TPU，適合學習 ML 模型）。
                                4. 如果程式碼需要 Kaggle 資料集，可在 Colab 中使用 Kaggle API 重新下載（類似本程式認證步驟）。
                                
                                **提示：** Colab 是免費的雲端 Jupyter Notebook，適合初學者練習數據分析。
                                """)
                                st.link_button("開啟 Google Colab 並上傳檔案", "https://colab.research.google.com/", help="點擊開啟 Colab 新分頁")
                        else:
                            st.warning("kernels 資料夾中尚未有 .ipynb 檔案，請先下載。")
                    except Exception as e:
                        st.error(f"下載程式碼時發生錯誤: {e}")
            else:
                st.warning("請先輸入要下載的程式碼 Ref，或確認已輸入資料集 Ref。")

# --- 主介面：Kaggle API 功能 ---
def main_page(api):
    st.title("Kaggle api")
    st.subheader("探索熱門資料集")
    
    DATASET_CATEGORIES = {
        "--- 請選擇一個探索分類 ---": "",
        "健康與醫療 (Health & Medical)": "health medical",
        "金融與經濟 (Finance & Economics)": "finance economics",
        "電腦視覺 (Computer Vision)": "computer vision images",
        "自然語言處理 (NLP)": "nlp text data",
        "氣候與環境 (Climate & Environment)": "climate environment",
        "教育 (Education)": "education",
        "社群媒體 (Social Media)": "social media"
    }

    def fetch_trending_datasets():
        selected_cat_name = st.session_state.category_selector
        if selected_cat_name == list(DATASET_CATEGORIES.keys())[0]:
            st.session_state.trending_datasets_df = pd.DataFrame()
            return
        search_term = DATASET_CATEGORIES[selected_cat_name]
        with st.spinner(f"正在擷取 '{selected_cat_name}' 分類的熱門資料集..."):
            try:
                datasets = api.dataset_list(search=search_term, sort_by='hottest')
                if datasets:
                    datasets = datasets[:10]
                    dataset_data = [{'Ref': d.ref, 'Title': d.title, 'URL': f"https://www.kaggle.com/{d.ref}"} for d in datasets]
                    st.session_state.trending_datasets_df = pd.DataFrame(dataset_data)
                else:
                    st.session_state.trending_datasets_df = pd.DataFrame()
            except Exception as e:
                st.error(f"擷取熱門資料集時發生錯誤: {e}")
                st.session_state.trending_datasets_df = pd.DataFrame()

    st.selectbox(
        "選擇一個分類，立即查看該領域最熱門的前10個資料集：",
        options=list(DATASET_CATEGORIES.keys()),
        key="category_selector",
        on_change=fetch_trending_datasets
    )
    st.caption("說明：選擇分類後，系統會顯示該領域最熱門的前10個資料集，點擊表格中的連結可前往 Kaggle 查看詳情。")
    
    if 'trending_datasets_df' in st.session_state and not st.session_state.trending_datasets_df.empty:
        st.subheader(f"📈 '{st.session_state.category_selector}' 分類下的熱門資料集")
        st.data_editor(
            st.session_state.trending_datasets_df,
            column_config={"URL": st.column_config.LinkColumn("Kaggle 連結", display_text="🔗 前往")},
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")
    with st.expander("🔍 手動搜尋 (點此展開)"):
        with st.form(key="dataset_search_form"):
            manual_search_term = st.text_input("輸入關鍵字進行精準搜尋：", placeholder="例如：'supply-chain-dataset', 'customer churn', 'sentiment'")
            st.caption("說明：輸入關鍵字（如 'supply-chain-dataset'）後點擊搜尋，結果會顯示在下方表格。支援多關鍵字搜尋，使用空格分隔。")
            submit_button = st.form_submit_button(label="搜尋")
            st.caption("說明：點擊搜尋後，系統將從 Kaggle 查詢符合關鍵字的資料集。")

        if submit_button and manual_search_term:
            with st.spinner(f"正在搜尋 '{manual_search_term}'..."):
                try:
                    datasets = api.dataset_list(search=manual_search_term)
                    if datasets:
                        dataset_data = [{'Ref': d.ref, 'Title': d.title} for d in datasets]
                        st.session_state.manual_search_df = pd.DataFrame(dataset_data)
                    else:
                        st.warning(f"找不到關於 '{manual_search_term}' 的資料集。")
                        st.session_state.manual_search_df = pd.DataFrame()
                except Exception as e:
                    st.error(f"搜尋時發生錯誤: {e}")
        
        if 'manual_search_df' in st.session_state and not st.session_state.manual_search_df.empty:
            st.subheader("手動搜尋結果")
            st.dataframe(st.session_state.manual_search_df, use_container_width=True)

    st.markdown("---")
    st.subheader("📥 下載資料集")
    dataset_ref_to_download = st.text_input("從上方任一結果中複製 Ref 貼到此處進行下載：", key="data_ref_unified")
    st.caption("說明：Ref 是資料集的唯一識別碼，格式如 'username/dataset-name'，可從上方表格複製。")
    if st.button("下載完整資料集", key="download_dataset_button"):
        st.caption("說明：點擊後，資料集將下載並解壓縮至 'kaggle_downloads/{Ref}' 子資料夾。")
        if dataset_ref_to_download:
            base_path = 'kaggle_downloads'
            download_path = os.path.join(base_path, dataset_ref_to_download)  # 新增：使用 Ref 作為子資料夾名稱
            os.makedirs(download_path, exist_ok=True)
            with st.spinner(f"正在下載並解壓縮 '{dataset_ref_to_download}' 的所有檔案..."):
                try:
                    api.dataset_download_files(dataset_ref_to_download, path=download_path, unzip=True)
                    st.success(f"資料集 '{dataset_ref_to_download}' 已成功下載並解壓縮到 `{download_path}` 資料夾！")
                except Exception as e:
                    st.error(f"下載資料集時發生錯誤: {e}")
        else:
            st.warning("請先輸入要下載的資料集 Ref。")
    pygwalker_page()
    code_finder_page(api)
    
# --- Pygwalker 資料探索介面 ---
def pygwalker_page():
    st.title("📊 Pygwalker")
    st.info("請先使用左側的 **Kaggle API 工具** 下載 CSV 檔案，然後在此處選擇檔案進行互動式分析。")

    download_folder = 'kaggle_downloads'
    try:
        supported_formats = ['*.csv', '*.xlsx']
        files = []
        for fmt in supported_formats:
            files.extend(glob.glob(os.path.join(download_folder, "**", fmt), recursive=True))
    except FileNotFoundError:
        files = []

    if not files:
        st.warning(f"在 `{download_folder}` 資料夾中找不到任何支援的檔案（CSV、Excel、JSON）。")
        return

    display_files = [os.path.relpath(f, download_folder) for f in files]
    selected_display_file = st.selectbox("選擇一個已下載的檔案進行分析：", display_files)
    st.caption("說明：支援 CSV、Excel、JSON 格式。若檔案無法載入，請檢查格式或編碼。")

    if selected_display_file:
        file_path = os.path.join(download_folder, selected_display_file)
        with st.spinner(f"正在讀取 '{selected_display_file}'..."):
            try:
                if file_path.endswith('.csv'):
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(file_path, encoding='latin1')
                elif file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                elif file_path.endswith('.json'):
                    df = pd.read_json(file_path)
                else:
                    st.error("不支援的檔案格式")
                    return
                st.subheader(f"正在分析: `{selected_display_file}`")
                renderer = StreamlitRenderer(df)
                renderer.explorer()
            except Exception as e:
                st.error(f"載入資料時發生錯誤: {e}")
                return

# --- 主程式執行區塊 ---
def main():
    if 'api_authenticated' not in st.session_state:
        st.session_state.api_authenticated = False
    
    api = init_kaggle_api()
    
    if not st.session_state.get('api_authenticated', False):
        st.error("Kaggle API 認證失敗，請上傳您的 kaggle.json 檔案。")
        uploaded_file = st.file_uploader("上傳 kaggle.json 檔案", type="json")
        if uploaded_file:
            try:
                # 將上傳檔案保存到指定路徑
                local_kaggle_path = "kaggle.json"
                with open(os.path.join(BASE_DIR, local_kaggle_path), 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                # 讀取 JSON 並設定環境變數
                with open(os.path.join(BASE_DIR, local_kaggle_path), 'r') as f:
                    credentials = json.load(f)
                os.environ['KAGGLE_USERNAME'] = credentials['username']
                os.environ['KAGGLE_KEY'] = credentials['key']
                
                # 初始化 API
                api = init_kaggle_api()
                if api:
                    st.session_state['api_authenticated'] = True
                    st.success("kaggle.json 讀取成功！API 已認證。")
                else:
                    st.error("kaggle.json 無效，認證失敗。")
            except Exception as e:
                st.error(f"讀取 kaggle.json 時發生錯誤: {e}")
    
    if st.session_state.get('api_authenticated', False) and api:
        st.toast("Kaggle API 已認證成功！")
        main_page(api)

if __name__ == "__main__":
    main()