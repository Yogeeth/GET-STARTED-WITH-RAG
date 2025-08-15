import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')

chroma_cli = chromadb.PersistentClient(path='./data/chroma.db')
image_loader = ImageLoader()
embedding_func = OpenCLIPEmbeddingFunction()

collection = chroma_cli.get_or_create_collection(
    'multimodal_collection',
    embedding_function=embedding_func,
    data_loader=image_loader
)

# collection.add(
#     ids=['6','7','8'],
#     uris=['./Images/india.png','./Images/india1.png','./Images/india2.png'],
#     metadatas=[{'category':'abstract'},{'category':'abstract'},{'category':'abstract'}]
# )
# collection.update(
#     ids=['0','1'],
#     uris=['./Images/fire.png','./Images/love.png'],
#     metadatas=[{'category':'abstract'},{'category':'abstract'}]
# )
print(collection.count())

def print_query_results(query_list: list, query_results: dict) -> None:
    result_count = len(query_results["ids"][0])

    for i in range(len(query_list)):
        print(f"Results for query: {query_list[i]}")

        for j in range(result_count):
            id = query_results["ids"][i][j]
            distance = query_results["distances"][i][j]
            data = query_results["data"][i][j]
            document = query_results["documents"][i][j]
            metadata = query_results["metadatas"][i][j]
            uri = query_results["uris"][i][j]

            print(
                f"id: {id}, distance: {distance}, metadata: {metadata}, document: {document}"
            )

            # Display image, the physical file must exist at URI.
            # (ImageLoader loads the image from file)
            print(f"data: {uri}")
            plt.imshow(data)
            plt.axis("off")
            plt.show()

while True:
    i=input("Choose [ ai, fire, India(3), Love, Maths, Peace ] : ")
    n=input("relavant results : ")
    if i=='end':
        break
    query=collection.query(
        query_texts=i,
        n_results=int(n),
        include=['documents','distances','metadatas','data','uris']
    )
    print_query_results([i],query)

# plt.imshow(query['data'][0][0])
# plt.axis("off")
# plt.show()