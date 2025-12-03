from app.deps import get_vs
from app.ingestion.loader import split_docs, load_docs


def main():
    docs = split_docs(load_docs("./data/docs"))
    vs = get_vs() # 获取Chroma数据库的实例
    vs.add_documents(docs) # 调用add_documents方法,对docs中的每个Document
    # 使用嵌入模型将其page_content转为向量,并将向量+原始文本+元数据 存入向量数据库
    try:
        vs.persist() # 调用persist方法将向量数据库的内容持久化到本地磁盘

    except Exception as e:
        print(e)
    print(f"Indexed {len(docs)} chunks into Chroma.")
    # 告知用户已成功将多少个文本块（chunks）索引到 Chroma 向量数据库中

if __name__ == '__main__':
    main()
