from app.deps import get_vs
from app.ingestion.loader import split_docs, load_docs


def main():
    docs = split_docs(load_docs("./data/docs"))
    vs = get_vs()
    vs.add_documents(docs)
    try:
        vs.persist()

    except Exception as e:
        print(e)
    print(f"Indexed {len(docs)} chunks into Chroma.")

if __name__ == '__main__':
    main()
