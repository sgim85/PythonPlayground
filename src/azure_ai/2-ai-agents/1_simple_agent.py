'''
Create Azure AI Agent in Azure Foundry (using Foundry new experience). Agent name = "expense-agent"
Added a Knowledge datasource via a company file explaining reimbursements amounts and limits.

Libs to install:
azure-ai-projects>=2.0.0b1
azure-identity

'''

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

myEndpoint = "https://project58956529-resource.services.ai.azure.com/api/projects/user1-58956529-2107"

project_client = AIProjectClient(
    endpoint=myEndpoint,
    credential=DefaultAzureCredential(),
)

myAgent = "expense-agent"
# Get an existing agent
agent = project_client.agents.get(agent_name=myAgent)
print(f"Retrieved agent: {agent.name}")

openai_client = project_client.get_openai_client()


# Reference the agent to get a response
response = openai_client.responses.create(
    input=[{"role": "user", "content": "Tell me what you can help with."}],
    extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
)

print(f"Response output: {response.output_text}")