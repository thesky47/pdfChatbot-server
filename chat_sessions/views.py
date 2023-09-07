import uuid
from django.shortcuts import get_object_or_404
from rest_framework import generics, views
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .models import ChatSession
from .serializers import ChatSessionSerializer, ConversationSerializer
from django.conf import settings
import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
import pinecone 
from langchain.memory import PostgresChatMessageHistory, ConversationBufferWindowMemory, CombinedMemory,   ConversationSummaryMemory

def get_filename_from_path(file_path):
    # Use os.path.basename to get the filename from the file path
    return os.path.basename(file_path)

def upload_file(file, user_fname, session_id):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    for doc in documents:
        doc.metadata.update({"session_id" : session_id})
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data

    pinecone.init(
        api_key=settings.PINECONE_API_KEY,  
        environment=settings.PINECONE_ENV, 
    )   
    index_name = "langchain-demo"

    if index_name not in pinecone.list_indexes():
        # we create a new indexs
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536  
        )

    return Pinecone.from_documents(
        docs, embeddings, index_name=index_name 
    )#namespace=user_fname
    # # define retriever
    # retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # # create a chatbot chain. Memory is managed externally.
    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=ChatOpenAI(model_name=llm_name, temperature=0), 
    #     chain_type=chain_type, 
    #     retriever=retriever, 
    #     return_source_documents=True,
    #     return_generated_question=True,
    # )
    # return qa 


class ChatSessionListView(generics.ListCreateAPIView):
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer
    permission_classes = [IsAuthenticated]

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)
    
    def create(self, request, *args, **kwargs):
        pdf_file = request.data.get('pdf_file')
        
        if not pdf_file:
            return Response({'detail': 'PDF file is required.'}, status=status.HTTP_400_BAD_REQUEST)

        # Generate title, id, and set the creator
        title = pdf_file.name
        creator = request.user

        chat_session_data = {
            'title': title,
            'pdf_file': pdf_file,
            'description': request.data.get('description', "No description"),
            'creator': creator.id,
        }

        serializer = ChatSessionSerializer(data=chat_session_data)

        if serializer.is_valid(raise_exception=True):
            session = serializer.save()
            try:
                db = upload_file(f".{session.pdf_file.url}", f"{session.creator.first_name}_{session.creator.id}", str(session.id))
                history = PostgresChatMessageHistory(
                    connection_string=f'postgresql://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_NAME")}',
                    session_id=str(session.id),
                )
            except Exception as e:
                print(e)
                return Response({"error" : f'{e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



class ConverseView(views.APIView):
    serializer_class = ConversationSerializer
    permission_classes = [IsAuthenticated]

    template = """You are a chatbot having a conversation with a human.
    Given the following extracted parts of a long document and a question, create a final answer.
    ```
    {context}
    ```

    ```
    Summary of Conversation:
    {history}

    ```

    ```
    Current conversation:
    {chat_history}
    ````

    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "history", "context"], template=template
    )
    
    def post(self, request, session_id):
        session = get_object_or_404(ChatSession, id=session_id)
        
        if query := request.data.get("userMessage", None):
            history = PostgresChatMessageHistory(
            connection_string=f'postgresql://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_NAME")}',
                session_id=str(session_id),
            )

            curr_memory = ConversationBufferWindowMemory(chat_memory=history, memory_key="chat_history", input_key="human_input", k=3)
            summary_memory = ConversationSummaryMemory(chat_memory=history, llm=OpenAI(), input_key="human_input")
            # Combined
            memory = CombinedMemory(memories=[curr_memory, summary_memory])
            
            pinecone.init(
                api_key=settings.PINECONE_API_KEY,  
                environment=settings.PINECONE_ENV, 
            )   

            index = pinecone.Index("langchain-demo")
            embeddings = OpenAIEmbeddings()
            vector_store = Pinecone(index, embeddings.embed_query, "text")
            retriever = vector_store.as_retriever(search_kwargs={"filter" : {"session_id" : str(session.id)} })
            matched_docs = retriever.get_relevant_documents(query)

            # matched_docs = PyPDFLoader("insightsonindia.com-UN Resolution on Kashmir in 1947.pdf").load()
            chain = load_qa_chain(
                OpenAI(temperature=0), 
                chain_type="stuff", 
                memory=memory, 
                prompt=self.prompt,
                verbose=True
            )
            response = chain({"input_documents": matched_docs, "human_input": query}, return_only_outputs=True)

            print(memory.memories[0].buffer)
            print()
            print(memory.memories[1].buffer)
            return Response({"response" : response, "History" :memory.memories[0].buffer, "Summary" :  memory.memories[1].buffer})


class ChatSessionDetailView(generics.RetrieveAPIView):
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer
    permission_classes = [IsAuthenticated]



