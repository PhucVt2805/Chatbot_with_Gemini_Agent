import os
import tempfile
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMMathChain
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, PromptTemplate, HumanMessagePromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Thiết lập các biến môi trường cần thiết
os.environ['GOOGLE_API_KEY'] = 'GOOGLE_API_KEY'
os.environ['TAVILY_API_KEY'] = 'TAVILY_API_KEY'

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Chat với Gemini & Tools",
    page_icon=":brain:",
    layout="centered",
)

st.header("Tải lên tệp thông tin của bạn")

# Hàm chuyển đổi vai trò người dùng cho Streamlit
def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

# Sidebar để chọn các công cụ sử dụng
st.sidebar.title("Bạn muốn sử dụng những công cụ nào?")
tools = []

# Khởi tạo mô hình LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Tiêu đề của chatbot trên trang
st.title("🤖 Gemini - ChatBot")

# Phần cấu hình công cụ tìm kiếm dựa trên PDF
retriever_toggle = st.sidebar.checkbox('Tìm kiếm dựa vào pdf tải lên')
if retriever_toggle:
    uploaded_files = st.sidebar.file_uploader("Tải lên file PDF của bạn", type=['pdf'], accept_multiple_files=True)
    temp_dir = tempfile.mkdtemp()
    topic = st.sidebar.text_input('Nội dung chính của các file PDF')
    if uploaded_files and st.sidebar.button("Xử lí tập tin PDF", type='primary'):
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(bytes_data)
            st.sidebar.write(f"Đã xử lí xong tập tin: {uploaded_file.name}.")
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        loader = PyPDFDirectoryLoader(path=temp_dir, extract_images=True)
        documents = loader.load_and_split(text_splitter=splitter)
        vector = FAISS.from_documents(documents=documents, embedding=embeddings)
        retriever = vector.as_retriever()
        retriever_tool = create_retriever_tool(
            retriever,
            "docs_search",
            f"Thông tin liên quan đến chủ đề {topic}. Đối với bất kỳ câu hỏi nào liên quan đến chủ đề này, bạn phải sử dụng công cụ này!",
        )
        tools.append(retriever_tool)

# Phần cấu hình công cụ tìm kiếm trên Wikipedia
wikipedia_toggle = st.sidebar.checkbox('Tìm kiếm trên Wikipedia')
if wikipedia_toggle:
    wiki = WikipediaAPIWrapper(lang='vi')
    wiki_tool = Tool(
        name="Wikipedia",
        func=wiki.run,
        description="""Một công cụ hữu ích để tìm kiếm trên internet để tìm thông tin về sự kiện, danh nhân, kỉ niệm, đối tượng...
        Lưu ý: Khi tìm kiếm trên Wikipedia hãy đưa ra những từ khoá chính trong câu.""",
    )
    tools.append(wiki_tool)

# Phần cấu hình công cụ toán học
math_toggle = st.sidebar.checkbox('Công cụ toán học')
if math_toggle:
    math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    math_tool = Tool.from_function(
        name="Calculator",
        func=math_chain.run,
        description="""Một công cụ hữu ích để thực hiện các phép toán học
        Nếu có bất cứ câu hỏi nào về Toán học, bạn phải sử dụng công cụ này!""",
    )
    tools.append(math_tool)

# Công cụ tìm kiếm web
web = TavilySearchResults()
web_tool = Tool(
    name="Web",
    func=web.run,
    description="""Một công cụ hữu ích để tìm kiếm trên internet các đường dẫn cho người dùng tham khảo thêm thông tin.
    Lưu ý: Khi sử dụng tool chỉ sử dụng từ khoá chính.
    Khi đưa ra câu trả lời cho câu hỏi của người dùng, bạn phải sử dụng công cụ này để cung cấp thêm đường dẫn tham khảo cho người dùng!""",
)
tools.append(web_tool)

# Cấu hình prompt cho chatbot
prompt = ChatPromptTemplate.from_messages(
    [SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template="""Bạn là 1 trợ lí hữu ích ảo và bạn cần phải đưa ra thông tin chính xác và hữu ích nhất cho người sử dụng.
            Hãy ưu tiên sử dụng những Tools mà tôi đã cung cấp. 
            Khi bạn sử dụng một công cụ, hãy đảm bảo rằng bạn cung cấp tên của công cụ và kết quả của nó trong câu trả lời của bạn. 
            Nếu các công cụ không hữu ích, hãy sử dụng kiến thức của riêng bạn để trả lời.""")),
     MessagesPlaceholder(variable_name='chat_history', optional=True),
     HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
     MessagesPlaceholder(variable_name='agent_scratchpad'),
    ]
)

# Hiển thị thông báo cho người dùng
st.info("Vui lòng làm mới trình duyệt nếu bạn cần đặt lại phiên", icon="🚨")

# Tạo agent và agent_executor để xử lý yêu cầu của người dùng
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Khởi tạo lịch sử trò chuyện nếu chưa có
if "chat_histories" not in st.session_state:
    st.session_state['chat_histories'] = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn trong lịch sử trò chuyện khi trang được tải lại
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ô nhập liệu cho tin nhắn của người dùng
if prompt := st.chat_input("Câu hỏi của bạn?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Truy vấn trợ lý bằng lịch sử trò chuyện mới nhất
    result = agent_executor.invoke(
        {'input': prompt},
        config={"configurable": {"session_id": "<foo>"}}
    )

    # Hiển thị phản hồi của trợ lý
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = result["output"]
        message_placeholder.markdown(full_response + "|")
    message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
