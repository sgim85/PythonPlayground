'''
Source: https://github.com/MicrosoftLearning/mslearn-ai-agents/blob/main/Instructions/02-build-ai-agent.md

In Azure, you will use the Azure AI Agent Service to develop an agent that uses built-in tools.

Use Azure AI Agent Service SDK to create a client application that uses an AI agent. 
The agent can use the built-in Code Interpreter tool to run dynamic Python code to perform statistical analyses.

Prerequisite: Create a project in Azure Foundry. A model, e.g. gpt-4o, will be deployed automatically. If model isn't automatically created as well, create & deploy it.

What the code below does:
Connects to the AI Foundry project.
Uploads the data file and creates a code interpreter tool that can access it.
Creates a new agent that uses the code interpreter tool and has explicit instructions to use Python as necessary for statistical analysis.
Runs a thread with a prompt message from the user along with the data to be analyzed.
Checks the status of the run in case there's a failure
Retrieves the messages from the completed thread and displays the last one sent by the agent.
Displays the conversation history
Deletes the agent and thread when they're no longer required.

Libraries to install:
python-dotenv
azure-identity
azure-ai-projects>=2.0.0b1
'''

import os
from dotenv import load_dotenv
from typing import Any
from pathlib import Path


# Add references
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, CodeInterpreterTool, CodeInterpreterToolAuto


def main(): 

    # Clear the console
    os.system('cls' if os.name=='nt' else 'clear')

    # Load environment variables from .env file
    load_dotenv()
    project_endpoint= os.getenv("PROJECT_ENDPOINT")
    model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

    # Display the data to be analyzed
    script_dir = Path(__file__).parent  # Get the directory of the script
    file_path = script_dir / 'data.txt'

    with file_path.open('r') as file:
        data = file.read() + "\n"
        print(data)

    # Connect to the AI Project and OpenAI clients
    with (
        DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True) as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as openai_client
    ):
    

        # Upload the data file and create a CodeInterpreterTool
        file = openai_client.files.create(
            file=open(file_path, "rb"), purpose="assistants"
        )
        print(f"Uploaded {file.filename}")

        code_interpreter = CodeInterpreterTool(
            container=CodeInterpreterToolAuto(file_ids=[file.id])
        )
        

        # Define an agent that uses the CodeInterpreterTool
        agent = project_client.agents.create_version(
            agent_name="data-agent",
            definition=PromptAgentDefinition(
                model=model_deployment,
                instructions="You are an AI agent that analyzes the data in the file that has been uploaded. Use Python to calculate statistical metrics as necessary.",
                tools=[code_interpreter],
            ),
        )
        print(f"Using agent: {agent.name}")
        

        # Create a conversation for the chat session
        conversation = openai_client.conversations.create()
        

        # Loop until the user types 'quit'
        while True:
            # Get input text
            user_prompt = input("Enter a prompt (or type 'quit' to exit): ")
            if user_prompt.lower() == "quit":
                break
            if len(user_prompt) == 0:
                print("Please enter a prompt.")
                continue

            # Send a prompt to the agent
            openai_client.conversations.items.create(
                conversation_id=conversation.id,
                items=[{"type": "message", "role": "user", "content": user_prompt}],
            )

            response = openai_client.responses.create(
                conversation=conversation.id,
                extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
                input="",
            )
            

            # Check the response status for failures
            if response.status == "failed":
                print(f"Response failed: {response.error}")
            

            # Show the latest response from the agent
            print(f"Agent: {response.output_text}")
            

        # Get the conversation history
        print("\nConversation Log:\n")
        items = openai_client.conversations.items.list(conversation_id=conversation.id)
        for item in items:
            if item.type == "message":
                print(f"item.content[0].type = {item.content[0].type}")
                role = item.role.upper()
                content = item.content[0].text
                print(f"{role}: {content}\n")


        # Clean up
        openai_client.conversations.delete(conversation_id=conversation.id)
        print("Conversation deleted")

        project_client.agents.delete(agent_name=agent.name, agent_version=agent.version)
        print("Agent deleted")



if __name__ == '__main__': 
    main()