import chromadb
from ollama import chat
from ollama import ChatResponse

client = chromadb.Client()


collection = client.create_collection("medical-diagnosis")
collection.add(
    documents=[
        "Febre alta, dores no corpo, dor de cabeça e manchas vermelhas na pele podem ser sintomas de dengue, uma doença viral transmitida por mosquitos.",
        "Dor de garganta, febre, dificuldade para engolir e inchaço nas amígdalas podem ser sintomas de amigdalite, uma inflamação das amígdalas causada por vírus ou bactérias.",
        "Tosse persistente, falta de ar e dor no peito podem ser sintomas de pneumonia, uma infecção que afeta os pulmões.",
        "Fadiga, náuseas, dor no abdômen superior e olhos amarelados podem ser sintomas de hepatite, uma inflamação do fígado causada por vírus ou outros fatores.",
        "Dor nas articulações, rigidez matinal e inchaço nas articulações podem indicar artrite reumatoide, uma doença autoimune que afeta as articulações.",
    ],
    ids=['dengue', 'amigdalite', 'pneumonia', 'hepatite', 'artrite-reumatoide'],
)


while True:
    question = input("\n\nDescreva seus sintomas ou digite 'sair' para encerrar: ").strip()
    if question.lower() == "sair":
        print("Obrigado por usar o programa")
        break

    
    results = collection.query(
        query_texts=[question],
        n_results=1,
    )

    if results["documents"]:
        related_document = results["documents"][0][0]

        
        response: ChatResponse = chat(model='llama3.2', messages=[
            {
                'role': 'user',
                'content': f"""
                Com base nos seguintes sintomas fornecidos:\n{question}\n
                Use as informações relevantes abaixo para oferecer uma resposta:\n{related_document}
                """
            }
        ])

        
        print(f"\nResposta baseada nos sintomas fornecidos:\n{response.message.content}")
    else:
        print("Desculpe, não consegui identificar uma condição médica baseada nos sintomas fornecidos. Por favor, tente descrever de outra forma ou consulte um médico.")
