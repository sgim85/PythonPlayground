'''
Develop a vision-enabled chat app
Use Azure AI Foundry to build a generative AI app that supports image input.

Source: https://github.com/MicrosoftLearning/mslearn-ai-vision/blob/main/Instructions/Labs/08-gen-ai-vision.md

In this exercise, you use the Phi-4-multimodal-instruct generative AI model to generate responses to prompts that include images. 
You'll develop an app that provides AI assistance with fresh produce in a grocery store by using Azure AI Foundry and the Azure AI Model Inference service.
'''

import os
from urllib.request import urlopen, Request
import base64
from pathlib import Path
from dotenv import load_dotenv

# Add references
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI


def main(): 

    # Clear the console
    os.system('cls' if os.name=='nt' else 'clear')
        
    try: 
    
        # Get configuration settings 
        load_dotenv()
        project_endpoint = os.getenv("PROJECT_CONNECTION")
        model_deployment =  os.getenv("MODEL_DEPLOYMENT")


        # Initialize the project client
        project_client = AIProjectClient(            
                credential=DefaultAzureCredential(
                    exclude_environment_credential=True,
                    exclude_managed_identity_credential=True
                ),
                endpoint=project_endpoint,
            )

        

        # Get a chat client
        openai_client = project_client.get_openai_client(api_version="2024-10-21")
        



        # Initialize prompts
        system_message = "You are an AI assistant in a grocery store that sells fruit. You provide detailed answers to questions about produce."
        prompt = ""

        # Loop until the user types 'quit'
        while True:
            prompt = input("\nAsk a question about the image\n(or type 'quit' to exit)\n")
            if prompt.lower() == "quit":
                break
            elif len(prompt) == 0:
                    print("Please enter a question.\n")
            else:
                print("Getting a response ...\n")


                # Get a response to image input
                image_url = "https://github.com/MicrosoftLearning/mslearn-ai-vision/raw/refs/heads/main/Labfiles/gen-ai-vision/orange.jpeg"
                image_format = "jpeg"
                request = Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
                image_data = base64.b64encode(urlopen(request).read()).decode("utf-8")
                data_url = f"data:image/{image_format};base64,{image_data}"

                response = openai_client.chat.completions.create(
                    model=model_deployment,
                    messages=[
                        {"role": "system", "content": system_message},
                        { "role": "user", "content": [  
                            { "type": "text", "text": prompt},
                            { "type": "image_url", "image_url": {"url": data_url}}
                        ] } 
                    ]
                )
                print(response.choices[0].message.content)
                    


    except Exception as ex:
        print(ex)


if __name__ == '__main__': 
    main()