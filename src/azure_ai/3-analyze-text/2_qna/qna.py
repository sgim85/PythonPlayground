'''
Sourece: https://github.com/MicrosoftLearning/mslearn-ai-language/blob/main/Instructions/Labs/02-qna.md

One of the most common conversational scenarios is providing support through a knowledge base of frequently asked questions (FAQs). 
Many organizations publish FAQs as documents or web pages, which works well for a small set of question and answer pairs, but large documents can be difficult and time-consuming to search.

Azure AI Language includes a question answering capability that enables you to create a knowledge base of question and answer pairs that can be queried using natural language input, and is most commonly used as a resource that a bot can use to look up answers to questions submitted by users.
In this exercise, you'll use the Azure AI Language Python SDK for text analytics to implement a simple question answering application.

Package to install:
azure-ai-language-questionanswering
'''

from dotenv import load_dotenv
import os

# import namespaces
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.questionanswering import QuestionAnsweringClient


def main():
    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')
        ai_project_name = os.getenv('QA_PROJECT_NAME')
        ai_deployment_name = os.getenv('QA_DEPLOYMENT_NAME')

        # Create client using endpoint and key
        credential = AzureKeyCredential(ai_key)
        ai_client = QuestionAnsweringClient(endpoint=ai_endpoint, credential=credential)


        # Submit a question and display the answer
        user_question = ''
        while True:
            user_question = input('\nQuestion:\n')
            if user_question.lower() == "quit":                
                break
            response = ai_client.get_answers(question=user_question,
                                            project_name=ai_project_name,
                                            deployment_name=ai_deployment_name)
            for candidate in response.answers:
                print(candidate.answer)
                print("Confidence: {}".format(candidate.confidence))
                print("Source: {}".format(candidate.source))



    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()
