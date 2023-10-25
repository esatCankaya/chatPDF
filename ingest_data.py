from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle


print("Loading data...")
loader = UnstructuredFileLoader("your.pdf")
raw_documents = loader.load()


print("Splitting text...")
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=600,
    chunk_overlap=100,
    length_function=len,
)
documents = text_splitter.split_documents(raw_documents)


print("Creating vectorstore...")
embeddings = OpenAIEmbeddings()
store_name = 'name_your_store'
if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_documents(documents, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
