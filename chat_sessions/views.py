import uuid
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .models import ChatSession
from .serializers import ChatSessionSerializer
from django.conf import settings
from PyPDF2 import PdfReader
import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
import pinecone 
from langchain.memory import PostgresChatMessageHistory

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
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536  
        )

    return Pinecone.from_documents(
        docs, embeddings, index_name=index_name, namespace=user_fname
    )
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
        serializer = self.get_serializer(queryset, fields=('id', 'title', 'description'), many=True)
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
            'creator': creator.id
        }

        serializer = ChatSessionSerializer(data=chat_session_data)

        if serializer.is_valid():
            session = serializer.save()
            try:
                db = upload_file(session.pdf_file.url, session.creator.first_name, str(session.id))
                history = PostgresChatMessageHistory(
                    connection_string=f'postgresql://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/chat_history',
                    session_id=str(session.id),
                )
            except Exception as e:
                print(e)
                return Response({"error" : f'{e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ChatSessionDetailView(generics.RetrieveAPIView):
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer
    permission_classes = [IsAuthenticated]



