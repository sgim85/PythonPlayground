"""
Create a knowledge mining solution
Use Azure AI Search to extract key information from documents and make it easier to search and analyze.

Source: https://github.com/MicrosoftLearning/mslearn-ai-information-extraction/blob/main/Instructions/Labs/05-knowledge-mining.md

In this exercise, you use AI Search to index a set of documents maintained by Margie's Travel, a fictional travel agency.
The indexing process involves using AI skills to extract key information to make them searchable, and generating a knowledge store containing data assets for further analysis.
"""

from dotenv import load_dotenv
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient


def main():

    # Clear the console
    os.system("cls" if os.name == "nt" else "clear")

    try:
        # Get config settings
        load_dotenv()
        search_endpoint = os.getenv("SEARCH_ENDPOINT")
        query_key = os.getenv("QUERY_KEY")
        index = os.getenv("INDEX_NAME")

        # Get a search client
        search_client = SearchClient(
            search_endpoint, index, AzureKeyCredential(query_key
        )

        # Loop until the user types 'quit'
        while True:
            # Get query text
            query_text = input("Enter a query (or type 'quit' to exit): ")
            if query_text.lower() == "quit":
                break
            if len(query_text) == 0:
                print("Please enter a query.")
                continue

            # Clear the console
            os.system("cls" if os.name == "nt" else "clear")

            # Search the index
            found_documents = search_client.search(
                search_text=query_text,
                select=["metadata_storage_name", "locations", "people", "keyphrases"],
                order_by=["metadata_storage_name"],
                include_total_count=True,
            )

            # Parse the results
            print(f"\nSearch returned {found_documents.get_count()} documents:")
            for document in found_documents:
                print(f"\nDocument: {document['metadata_storage_name']}")
                print(" - Locations:")
                for location in document["locations"]:
                    print(f"   - {location}")
                print(" - People:")
                for person in document["people"]:
                    print(f"   - {person}")
                print(" - Key phrases:")
                for phrase in document["keyphrases"]:
                    print(f"   - {phrase}")

    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()
