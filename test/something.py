# from 12, pattern value function is not hugely different
#%%
import numpy as np
from collections import Counter
import gradio as gr
import time
import pandas as pd
import copy
import tempfile
#%%

def data_to_excel(datas):
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        with pd.ExcelWriter(tmp.name) as writer:
            dt = pd.DataFrame(datas)
            dt.to_excel(writer, index = False, sheet_name = 'Sheet1')
        excel_download = gr.File(value = tmp.name, visible = True)
    return [excel_download]
def show(datas):
    with gr.Blocks() as demo:
        gr.Markdown('Excel表格生成')
        button = gr.Button(value = 'Click to turn data to Excel~!')
        with gr.Row():
            excel_download = gr.File(interactive=False, visible = False)
        try:
            button.click(data_to_excel, [datas], [excel_download])
        except Exception as e:
            print(e)
    demo.launch()
datas = [1]
show(datas)
#%%
