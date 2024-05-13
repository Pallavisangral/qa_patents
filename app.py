from flask import Flask, request, jsonify
from qa_results import patent_retrieval


app = Flask(__name__)
patent_api = patent_retrieval()
retrieval_chain = patent_api.get_retriever()

@app.route('/')
def home():
    return "Welcome to Patent Retrieval API"

@app.route('/api/patent/retrieve', methods=['GET'])
def retrieve_patent():
    # query = request.args.get('query')
    # output = retrieval_chain.invoke({"input": query})
    # print(output['answer'])
    # return output['answer']
    query = request.args.get('query')
    output = retrieval_chain.invoke({"input": query})
    return jsonify({"answer": output['answer']})



# if __name__ == '__main__':
#     app.run(debug=True)

