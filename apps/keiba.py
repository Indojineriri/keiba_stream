import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objs as go
from plotly.subplots import make_subplots

sns.set()
# タイトル設定
st.title('競馬データ分析')

# タブのように機能するラジオボタンを作成
tab = st.radio("競馬場を選択してください:", ('東京', '京都'))

# 選択された競馬場に応じてデータをロード
if tab == '東京':
    df = pd.read_pickle('apps/merged_tokyo_turf.pickle')
    
elif tab == '京都':
    df = pd.read_pickle('apps/merged_kyoto_turf.pickle')
    
def calc_binned_avg(feature, target, bins):
    df_binned = df.copy()
    df_binned['binned'] = pd.cut(df_binned[feature], bins=bins, include_lowest=True)
    return df_binned.groupby('binned')[target].mean().reset_index()
#
conversion_dict = {
        '枠番':'枠 番',  '斤量':'斤量',  '馬体重':'Weight',  '平均着差':'Goal difference_5R',  '平均賞金額':'Bonus_5R',
        '平均Last3F':'Last3F_5R',  '前走スピード指数':'speed_1R'
        }

setting_length = st.selectbox(
    'コースの長さを選択してください:',
    options=df['course_len'].unique(), # course_lenのユニークな値を選択肢として提供
    format_func=lambda x: f'{int(x*100)}m' # 選択肢をメートル単位で表示
)
idx=df['course_len']==setting_length
df=df[idx].copy()
#縦軸、横軸の設定
feature = st.selectbox(
    '比較する特徴量を選択してください:',
#    options=df.columns.drop('rank') # 'rank'を除外したすべての特徴量
    options=['枠番','斤量','馬体重','平均着差','平均賞金額','平均Last3F','前走スピード指数']
)
#マップの変換
feature=feature.map(conversion_dict)

# ビンの作成
num_bins = st.number_input('分割数を入力してください', min_value=1, max_value=100, value=10)
range_min, range_max = st.slider(
    'グラフのレンジを選択してください',
    min_value=float(df[feature].min()),
    max_value=float(df[feature].max()),
    value=(float(df[feature].min()), float(df[feature].max()))
)

# ビンの作成、ここではユーザーが指定した範囲を使用
bins = np.linspace(start=range_min, stop=range_max, num=num_bins)

# 平均順位の計算
avg_rank = calc_binned_avg(feature, 'rank', bins)

# プロット作成
fig_BS = make_subplots(rows=1, cols=1, subplot_titles=[feature])

# 棒グラフの追加
fig_BS.add_trace(go.Bar(x=avg_rank['binned'].astype(str), y=avg_rank['rank']), row=1, col=1)

# レイアウト調整
fig_BS.update_layout(barmode='group')

# Streamlitにグラフを表示
st.plotly_chart(fig_BS)
