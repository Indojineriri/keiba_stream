import streamlit as st

# 座席の状態を切り替える関数
def toggle_seat_status():
    # ボタンが押された座席のキーを取得
    button_key = st.session_state['last_clicked']
    # 座席の状態をトグル
    st.session_state['seats'][button_key] = not st.session_state['seats'][button_key]

# 列数と行数の入力
num_rows = st.number_input("行数を入力", min_value=1, max_value=10, value=3)
num_cols = st.number_input("列数を入力", min_value=1, max_value=10, value=5)

# 座席の状態を格納する辞書を初期化
if 'seats' not in st.session_state:
    st.session_state['seats'] = {}
if 'last_clicked' not in st.session_state:
    st.session_state['last_clicked'] = None

# 座席グリッドの生成
for row in range(num_rows):
    cols = st.columns(num_cols)
    for col in range(num_cols):
        # ボタンのキーを作成
        button_key = f"seat_{row}_{col}"
        
        # 座席の状態が未定義の場合はFalse(空席)で初期化
        if button_key not in st.session_state['seats']:
            st.session_state['seats'][button_key] = False
        
        # 座席ボタンの表示と状態の切り替え
        label = "○" if st.session_state['seats'][button_key] else "×"
        if cols[col].button(label, key=button_key):
            # 最後にクリックされたボタンのキーを記録
            st.session_state['last_clicked'] = button_key
            # 座席の状態を切り替える
            toggle_seat_status()