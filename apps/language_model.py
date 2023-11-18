from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)
output = chat.predict_messages([HumanMessage(content="日本の総理大臣は誰ですか？")])

print(output)