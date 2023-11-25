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
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
import lightgbm as lgb
#jp_font_path = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
#jp_font = font_manager.FontProperties(fname=[f for f in jp_font_path if 'IPAexGothic' in f][0])
sns.set()
fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
print(any('IPAexGothic' in font for font in fonts))
#パラメータ系の設定
plt.rcParams['font.family'] = 'IPAexGothic'  # 日本語フォントに設定
conversion_dict = {
        '枠番':'Waku',  '斤量':'斤量',  '馬体重':'Weight',  '平均着差':'Goal difference_5R',  '平均賞金額':'Bonus_5R',
        '平均Last3F':'Last3F_5R', '平均位置取り':'first_corner_5R' ,'前走スピード指数':'speed_1R'
        }
selection=['枠番','斤量','平均着差','平均賞金額','平均Last3F','平均位置取り','前走スピード指数']
default_selections = ['平均着差', '平均賞金額', '平均Last3F', '平均位置取り']
params = {'objective': 'binary',
'random_state': 100,
'feature_pre_filter': False,
'lambda_l1': 0.0,
'lambda_l2': 0.0,
'num_leaves': 31,
'feature_fraction': 0.6,
'bagging_fraction': 0.5111969317302304,
'bagging_freq': 1,
'min_child_samples': 20,
'num_iterations': 1000,}

page=st.sidebar.radio('ページを選択してください',['データ分析','データ可視化','京阪杯','ジャパンC'])

if page=='データ分析':
    # タブのように機能するラジオボタンを作成
    st.title('競馬データ分析')
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
        options=selection
    )
    #マップの変換
    converted_features = conversion_dict.get(feature, feature)

    # ビンの作成
    num_bins = st.number_input('分割数を入力してください', min_value=1, max_value=100, value=10)
    range_min, range_max = st.slider(
        'グラフのレンジを選択してください',
        min_value=float(df[converted_features].min()),
        max_value=float(df[converted_features].max()),
        value=(float(df[converted_features].min()), float(df[converted_features].max()))
    )

    # ビンの作成、ここではユーザーが指定した範囲を使用
    bins = np.linspace(start=range_min, stop=range_max, num=num_bins)

    # 平均順位の計算
    avg_rank = calc_binned_avg(converted_features, 'rank', bins)

    # プロット作成
    fig_BS = make_subplots(rows=1, cols=1, subplot_titles=[converted_features])

    # 棒グラフの追加
    fig_BS.add_trace(go.Bar(x=avg_rank['binned'].astype(str), y=avg_rank['rank']), row=1, col=1)

    # レイアウト調整
    fig_BS.update_layout(
        yaxis_title='複勝率',
        barmode='group'
    )

    # Streamlitにグラフを表示
    st.plotly_chart(fig_BS)
    
elif page=='データ可視化':
    st.title('東京京都11/24 7-12レース')
    tab = st.radio("競馬場を選択してください:", ('東京', '京都'))
    if tab == '東京':
        df_learn_t  = pd.read_pickle('apps/merged_tokyo_turf.pickle')
        df_learn_d=pd.read_pickle('apps/merged_tokyo_durt.pickle')
        df=pd.read_pickle('apps/calin_tokyo.pickle')
    elif tab == '京都':
        df_learn_t= pd.read_pickle('apps/merged_kyoto_turf.pickle')
        df_learn_d=pd.read_pickle('apps/merged_kyoto_durt.pickle')
        df=pd.read_pickle('apps/calin_kyoto.pickle')
    #レースIDの取得
    race_id_unique=df.index.unique()
    race_id = st.selectbox(
        '予想するレースを選択してください:',
    #options=df.columns.drop('rank') # 'rank'を除外したすべての特徴量
        options=race_id_unique
    )
    idx=df.index==race_id
    df_use=df[idx].copy()
    st.subheader("データ可視化（2軸で傾向見れるよーん）")
    default_feature1 = selection.index('平均位置取り') if '平均位置取り' in selection else 0
    default_feature2 = selection.index('平均Last3F') if '平均Last3F' in selection else 0
    feature1 = st.selectbox('第1の特徴量を選択してください:', selection,index=default_feature1)
    feature2 = st.selectbox('第2の特徴量を選択してください:', selection,index=default_feature2)
    
    converted_features1=conversion_dict.get(feature1, feature1)
    converted_features2=conversion_dict.get(feature2, feature2)
    # 散布図の作成
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_use, x=converted_features1, y=converted_features2)
    plt.title(f"scatter: {converted_features1} vs {converted_features2}")
    # 馬名の追加
    for i in range(df_use.shape[0]):
        plt.text(x=df_use[converted_features1].iloc[i], y=df_use[converted_features2].iloc[i], s=df_use['馬 番'].iloc[i],
            fontdict=dict(color='red', size=12),
            bbox=dict(facecolor='yellow', alpha=1))

    st.pyplot(plt)
    
    #学習の開始
    st.subheader("機械学習予想（LightGBMでやってるよーん）")
    #特徴量の選択
    selected_features = st.multiselect('特徴量を選択してください:', selection, default=default_selections)
    ml_features = [conversion_dict.get(feature, feature) for feature in selected_features]
    #コース長さの設定
    course_length=df_use['course_len'].mean()
    print(df_use)
    
    #芝かダートの選択
    if 'race_type_芝' in df_use.columns:
        df_learn = df_learn_t
    else:
    # 'race_type_芝' が存在しない場合の処理（必要に応じて）
    # 例: df_learn を別のデフォルト値に設定
        df_learn = df_learn_d # または別の適当なデフォルト値
        
    idx=df_learn['course_len']==course_length
    #学習用のXを設定
    X=df_learn[idx]
    X = X[ml_features]
    #学習用のYを設定
    y=df_learn[idx]
    y=y['rank']
    #LGBMでの学習の実施
    lgb_clf = lgb.LGBMClassifier(**params)
    lgb_clf.fit(X, y)
    #学習させるべきデータ
    probabilities = lgb_clf.predict_proba(df_use[ml_features])[:, 1]
    pred=df_use[['馬 番','馬名']].copy()
    pred['複勝確率%']=probabilities*100
    st.dataframe(pred)
    
elif page=='京阪杯':
    df_learn=pd.read_pickle('apps/merged_keihan.pickle')
    df_use=pd.read_pickle('apps/calin_keihan.pickle')
    
    st.subheader("データ可視化（2軸で傾向見れるよーん）")
    default_feature1 = selection.index('平均位置取り') if '平均位置取り' in selection else 0
    default_feature2 = selection.index('平均Last3F') if '平均Last3F' in selection else 0
    feature1 = st.selectbox('第1の特徴量を選択してください:', selection,index=default_feature1)
    feature2 = st.selectbox('第2の特徴量を選択してください:', selection,index=default_feature2)
    converted_features1=conversion_dict.get(feature1, feature1)
    converted_features2=conversion_dict.get(feature2, feature2)
    # 散布図の作成
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_use, x=converted_features1, y=converted_features2)
    plt.title(f"scatter: {converted_features1} vs {converted_features2}")
    # 馬名の追加
    for i in range(df_use.shape[0]):
        plt.text(x=df_use[converted_features1].iloc[i], y=df_use[converted_features2].iloc[i], s=df_use['馬 番'].iloc[i],
            fontdict=dict(color='red', size=12),
            bbox=dict(facecolor='yellow', alpha=1))

    st.pyplot(plt)
    
    #学習の開始
    st.subheader("機械学習予想（LightGBMでやってるよーん）")
    #特徴量の選択
    selected_features = st.multiselect('特徴量を選択してください:', selection, default=default_selections)
    ml_features = [conversion_dict.get(feature, feature) for feature in selected_features]
    #学習用のXを設定
    X=df_learn
    X = X[ml_features]
    #学習用のYを設定
    y=df_learn
    y=y['rank']
    #LGBMでの学習の実施
    lgb_clf = lgb.LGBMClassifier(**params)
    lgb_clf.fit(X, y)
    #学習させるべきデータ
    probabilities = lgb_clf.predict_proba(df_use[ml_features])[:, 1]
    pred=df_use[['馬 番','馬名']].copy()
    pred['複勝確率%']=probabilities*100
    st.dataframe(pred)
    
elif page=='ジャパンC':
    df_learn=pd.read_pickle('apps/merged_japan.pickle')
    df_use=pd.read_pickle('apps/calin_japan.pickle')
    print(df_use)
    st.subheader("データ可視化（2軸で傾向見れるよーん）")
    default_feature1 = selection.index('平均位置取り') if '平均位置取り' in selection else 0
    default_feature2 = selection.index('平均Last3F') if '平均Last3F' in selection else 0
    feature1 = st.selectbox('第1の特徴量を選択してください:', selection,index=default_feature1)
    feature2 = st.selectbox('第2の特徴量を選択してください:', selection,index=default_feature2)
    converted_features1=conversion_dict.get(feature1, feature1)
    converted_features2=conversion_dict.get(feature2, feature2)
    st.dataframe(df_use)
    # 散布図の作成
#    sns.set_style("whitegrid")
#    plt.figure(figsize=(10, 6))
#    sns.scatterplot(data=df_use, x=converted_features1, y=converted_features2)
#    plt.title(f"scatter: {converted_features1} vs {converted_features2}")
#    # 馬名の追加
#    for i in range(df_use.shape[0]):
#        plt.text(x=df_use[converted_features1].iloc[i], y=df_use[converted_features2].iloc[i], s=df_use['馬 番'].iloc[i],
#            fontdict=dict(color='red', size=12),
#            bbox=dict(facecolor='yellow', alpha=1))
#
#    st.pyplot(plt)
    
    #学習の開始
    st.subheader("機械学習予想（LightGBMでやってるよーん）")
    #特徴量の選択
    selected_features = st.multiselect('特徴量を選択してください:', selection, default=default_selections)
    ml_features = [conversion_dict.get(feature, feature) for feature in selected_features]
    #学習用のXを設定
    X=df_learn
    X = X[ml_features]
    #学習用のYを設定
    y=df_learn
    y=y['rank']
    #LGBMでの学習の実施
    lgb_clf = lgb.LGBMClassifier(**params)
    lgb_clf.fit(X, y)
    #学習させるべきデータ
    probabilities = lgb_clf.predict_proba(df_use[ml_features])[:, 1]
    pred=df_use[['馬 番','馬名']].copy()
    pred['複勝確率%']=probabilities*100
    st.dataframe(pred)