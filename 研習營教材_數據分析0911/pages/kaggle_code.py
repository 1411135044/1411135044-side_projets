import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import textwrap
import nbformat

# 使用 secom0717_CP.py 定義的 BASE_DIR
BASE_DIR = st.session_state.get('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))

def convert_ipynb_to_streamlit(ipynb_path, output_py_path, data_path=None):
    # 讀取 Jupyter Notebook
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    streamlit_code = [
        "import streamlit as st",
        "import pandas as pd",
        "import plotly.express as px",
        "import plotly.graph_objects as go",
        "import json",
        "",
        "# 功能說明",
        "# 本程式將 Jupyter Notebook 轉換為 Streamlit 應用，",
        "# 完整呈現所有 Markdown 文字、程式碼和圖表（例如 Plotly 餅圖）。",
        "# 用途：展示供應鏈分析，特別是按運輸模式的缺陷率分佈。",
        "# 限制：",
        "# 1. 需要提供資料集 CSV 檔案（例如從 Kaggle 下載），否則使用模擬資料。",
        "# 2. 目前僅支援 Plotly 圖表，其他圖表庫（如 Matplotlib）需額外處理。",
        "# 3. 若 Notebook 包含複雜依賴（如 scikit-learn），需確保環境已安裝。",
        "",
        "# 設置頁面配置",
        'st.set_page_config(page_title="供應鏈分析", layout="wide")',
        "",
        "# 建立共用變數環境，模擬 Jupyter Notebook 的行為",
        "session_vars = {}",
        ""
    ]
    
    if data_path:
        streamlit_code.append(f"# 載入資料集")
        streamlit_code.append(f"try:")
        streamlit_code.append(f"    data = pd.read_csv('{data_path}')")
        streamlit_code.append(f"    session_vars['data'] = data")
        streamlit_code.append(f"except Exception as e:")
        streamlit_code.append(f"    st.error(f'載入資料失敗: {{e}}')")
        streamlit_code.append(f"    data = pd.DataFrame({{")
        streamlit_code.append(f"        'Transportation modes': ['Air', 'Sea', 'Road', 'Rail'],")
        streamlit_code.append(f"        'Defect rates': [2.5, 3.2, 1.8, 2.1]")
        streamlit_code.append(f"    }})")
        streamlit_code.append(f"    session_vars['data'] = data")
        streamlit_code.append("")
    
    cell_index = 0
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            markdown_content = ''.join(cell['source']).rstrip()
            streamlit_code.append(f"# Markdown 內容")
            streamlit_code.append(f'st.markdown("""')
            streamlit_code.append(markdown_content)
            streamlit_code.append(f'""")')
            streamlit_code.append("")
        
        elif cell['cell_type'] == 'code':
            code_content = ''.join(cell['source']).rstrip()
            streamlit_code.append(f"# 程式碼展示")
            streamlit_code.append(f'st.markdown(\"### 程式碼\")')
            streamlit_code.append(f'st.code(\"\"\"')
            streamlit_code.append(f"{code_content}")
            streamlit_code.append(f'\"\"\", language=\"python\")')
            streamlit_code.append("")
            
            streamlit_code.append("try:")
            safe_code = repr(code_content)
            streamlit_code.append("    old_figures = {k: id(v) for k, v in session_vars.items() if isinstance(v, go.Figure)}")  # 記錄舊 Figure 的 id
            streamlit_code.append(f"    exec({safe_code}, globals(), session_vars)")
            
            # 顯示文字輸出 (print 或 DataFrame)
            if 'outputs' in cell and cell['outputs']:
                for output in cell['outputs']:
                    if output.get('output_type') == 'stream':
                        text = ''.join(output.get('text', '')).strip()
                        if text:
                            streamlit_code.append(f"    st.text({repr(text)})")
                    elif output.get('output_type') == 'execute_result':
                        text_data = output.get('data', {}).get('text/plain', '')
                        if text_data:
                            text = ''.join(text_data).strip()
                            # 檢查是否為 DataFrame
                            if 'DataFrame' in text:
                                streamlit_code.append(f"    st.dataframe(session_vars.get(list(session_vars.keys())[-1]))")
                            else:
                                streamlit_code.append(f"    st.text({repr(text)})")
            
            # 處理圖表輸出，優先使用 Notebook outputs 的 JSON 重建圖表（若有）
            plotly_displayed = False
            if 'outputs' in cell and cell['outputs']:
                for output in cell['outputs']:
                    if output.get('output_type') == 'display_data' and 'application/vnd.plotly.v1+json' in output.get('data', {}):
                        fig_data = output['data']['application/vnd.plotly.v1+json']
                        streamlit_code.append(f"    fig_data = {repr(fig_data)}")
                        streamlit_code.append("    fig = go.Figure(fig_data)")
                        streamlit_code.append(f"    st.plotly_chart(fig, use_container_width=True, key='plotly_{cell_index}')")
                        plotly_displayed = True
            
            # 如果 Notebook 沒有存圖表輸出，從 session_vars 找新增加或變更的圖表（避免累積重複）
            if not plotly_displayed:
                streamlit_code.append(f"    _cell_id = {cell_index}")
                streamlit_code.append("    for var_name, var_value in session_vars.items():")
                streamlit_code.append("        if isinstance(var_value, go.Figure) and (var_name not in old_figures or id(var_value) != old_figures[var_name]):")
                streamlit_code.append(f"            st.plotly_chart(var_value, use_container_width=True, key=f'{{var_name}}_{{id(var_value)}}_{{_cell_id}}')")
            
            streamlit_code.append("except Exception as e:")
            streamlit_code.append(f"    st.error(f'執行程式碼時發生錯誤: {{e}}')")
            streamlit_code.append("")
            cell_index += 1

    os.makedirs(os.path.dirname(output_py_path), exist_ok=True)
    with open(output_py_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(streamlit_code))

    return output_py_path
#直接轉換的程式碼會有縮排錯誤要再檢查轉換邏輯

def main():
    st.title("Jupyter Notebook 轉 Streamlit")
    st.info("選擇.ipynb 檔案或上傳新檔案，並選擇資料集（可選），即可自動生成 Streamlit 程式，儲存至 ./pages/ 並支援分頁切換。")
    st.warning("""
    這程式能快速把 Kaggle Notebook「一鍵轉換」成 Streamlit App，適合用在 展示分析結果 或 分享資料故事。\n
    但是:\n   
    1.檔案路徑 還是要手動維護（特別是 kaggle_downloads 與資料集）。\n
    2.只支援 Plotly 圖表，其他圖表要額外寫轉換。\n
    3.如果 Notebook 太複雜（多資料來源 / 機器學習 / 特殊套件），就需要自己改程式碼。\n""")
    kaggle_downloads_folder = os.path.join(BASE_DIR,"kaggle_downloads")
    ipynb_files = glob.glob(os.path.join(kaggle_downloads_folder, "**", "*.ipynb"), recursive=True)
    ipynb_files = ipynb_files or ["無可用 .ipynb 檔案"]

    st.subheader("選擇或上傳 .ipynb 檔案")
    selected_ipynb = st.selectbox(
    "選擇 .ipynb 檔案：",
    [os.path.basename(f) for f in ipynb_files],
    key="ipynb_selector"
)
    ipynb_map = {os.path.basename(f): f for f in ipynb_files}

    ipynb_file = st.file_uploader("或上傳新的 .ipynb 檔案", type="ipynb", key="ipynb_uploader")
    
    # 添加 checkbox 判斷是否需要資料集
    use_dataset = st.checkbox("使用資料集", value=False, key="use_dataset")
    
    # 根據 checkbox 顯示資料集選擇元件
    data_path = None
    if use_dataset:
        data_folder = os.path.join(BASE_DIR, "kaggle_downloads")
        data_files = glob.glob(os.path.join(data_folder, "**", "*.csv"), recursive=True)
        data_files = [os.path.relpath(f, data_folder) for f in data_files] or ["無可用 CSV 檔案"]
        selected_data = st.selectbox("選擇資料集（CSV，可選）：", data_files, key="data_selector")
        data_file = st.file_uploader("或上傳新的 CSV 檔案", type="csv", key="data_uploader")
        
        if selected_data != "無可用 CSV 檔案":
            data_path = selected_data
        elif data_file:
            data_path = os.path.join(BASE_DIR, "temp_data.csv")
            with open(data_path, "wb") as f:
                f.write(data_file.getvalue())
    
    ipynb_path = None
    if selected_ipynb in ipynb_map:
        ipynb_path = ipynb_map[selected_ipynb]
    elif ipynb_file:
        ipynb_path = os.path.join(BASE_DIR, "temp_notebook.ipynb")
        with open(ipynb_path, "wb") as f:
            f.write(ipynb_file.getvalue())

    
    if ipynb_path:
        output_filename = os.path.basename(ipynb_path).replace(".ipynb", "_streamlit.py")
        output_py_path = os.path.join(BASE_DIR, "pages", output_filename)
        
        if st.button("轉換為 Streamlit"):
            try:
                output_path = convert_ipynb_to_streamlit(ipynb_path, output_py_path, data_path)
                if 'generated_file' not in st.session_state:
                    st.session_state['generated_file'] = None
                st.session_state['generated_file'] = output_path
                st.success(f"轉換成功！已生成: {output_py_path}")
                st.rerun()  # 觸發重新渲染以切換到新分頁
            except Exception as e:
                st.error(f"轉換失敗: {e}")
    
    st.markdown("---")
    st.subheader("已轉換的 Streamlit 頁面")
    pages_folder = os.path.join(BASE_DIR, "pages")
    page_files = glob.glob(os.path.join(pages_folder, "*.py"))
    if 'page_files' not in st.session_state:
        st.session_state['page_files'] = []
    st.session_state['page_files'] = page_files
    
    if page_files:
        st.write("以下是 ./pages/ 中已生成的 Streamlit 頁面：")
        for page in page_files:
            if os.path.basename(page) != "secom0717_CP.py":  # 排除不相關檔案
                page_name = os.path.basename(page).replace(".py", "")
                st.page_link(f"pages/{os.path.basename(page)}", label=page_name)
    else:
        st.info("尚未在 ./pages/ 中生成任何頁面，請先執行轉換。")

if __name__ == "__main__":
    main()